"""
04_train.py - LoRA / QLoRA supervised fine-tuning for the Indian-legal RAG assistant.

Works for: Qwen3 dense (4B/8B/14B/32B), Qwen3-30B-A3B (MoE), Qwen3-VL-*-Instruct (dense or MoE).
Text-only training: the vision tower (if any) stays frozen and untouched.

Design choices (why this recipe):
  * Loss ONLY on the assistant tokens (prompt + evidence masked) -> learns behaviour, not to parrot context.
  * LoRA on attention projections (+ dense MLP for non-MoE). MoE experts/router are frozen: stable, cheap,
    and what you want when the goal is grounding/citation *behaviour*, not new knowledge.
  * rsLoRA scaling, cosine LR, warmup, grad clipping, bf16, gradient checkpointing.
  * Length-grouped batches (less padding), eval loss + early stopping + best-checkpoint restore.
  * Fully resumable (--resume) - safe on pre-emptible/rented GPUs.
  * Examples longer than --max-len are DROPPED (never truncated: truncation would cut the answer).

Single GPU (H200 / 96 GB card, bf16 LoRA):
    python finetune/04_train.py --model /path/to/Qwen3-VL-30B-A3B-Instruct
Smaller card (48 GB), 4-bit:
    python finetune/04_train.py --model ... --quant 4bit
Multi-GPU data parallel:
    torchrun --nproc_per_node=2 finetune/04_train.py --model ...
"""

import argparse
import inspect
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling import (  # noqa: E402
    add_mm_ids, chat_prompt_text, load_model, load_tokenizer, probe_text_only,
)

FT_DIR = Path(__file__).resolve().parent
IM_END = "<|im_end|>"


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
class SFTDataset(Dataset):
    """Pre-tokenised examples; labels are -100 on the prompt, real ids on the answer."""

    def __init__(self, path, tokenizer, max_len, limit=0, name="data"):
        self.items, dropped, total_tok, label_tok = [], 0, 0, 0
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if limit:
            rows = rows[:limit]
        for r in rows:
            user = r["messages"][0]["content"]
            answer = r["messages"][1]["content"].strip()
            p_ids = tokenizer(chat_prompt_text(tokenizer, user), add_special_tokens=False)["input_ids"]
            a_ids = tokenizer(answer + IM_END, add_special_tokens=False)["input_ids"]
            if len(p_ids) + len(a_ids) > max_len:
                dropped += 1
                continue
            ids = np.asarray(p_ids + a_ids, dtype=np.int32)
            lab = np.asarray([-100] * len(p_ids) + a_ids, dtype=np.int32)
            self.items.append((ids, lab))
            total_tok += len(ids)
            label_tok += len(a_ids)
        n = max(1, len(self.items))
        print(f"[{name}] {len(self.items)} examples kept, {dropped} dropped (> {max_len} tokens) | "
              f"mean len {total_tok / n:.0f} tok | trained-on tokens {100 * label_tok / max(1, total_tok):.1f}%")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        ids, lab = self.items[i]
        return {"input_ids": torch.from_numpy(ids.astype(np.int64)),
                "labels": torch.from_numpy(lab.astype(np.int64))}


class Collator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        longest = max(len(f["input_ids"]) for f in feats)
        longest = (longest + 7) // 8 * 8  # tensor-core friendly
        ids = torch.full((len(feats), longest), self.pad_id, dtype=torch.long)
        lab = torch.full((len(feats), longest), -100, dtype=torch.long)
        att = torch.zeros((len(feats), longest), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n], lab[i, :n], att[i, :n] = f["input_ids"], f["labels"], 1
        return add_mm_ids({"input_ids": ids, "labels": lab, "attention_mask": att})


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def make_training_args(**kw):
    """Build TrainingArguments, tolerating renamed/removed args across transformers versions."""
    from transformers import TrainingArguments
    params = inspect.signature(TrainingArguments.__init__).parameters
    if "eval_strategy" not in params and "eval_strategy" in kw:
        kw["evaluation_strategy"] = kw.pop("eval_strategy")
    dropped = [k for k in kw if k not in params]
    for k in dropped:
        kw.pop(k)
    if dropped:
        print(f"[warn] TrainingArguments ignored unsupported args: {dropped}")
    return TrainingArguments(**kw)


def lora_target_regex(is_moe: bool) -> str:
    attn = r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    if is_moe:  # experts are fused/frozen; adapt attention only
        return rf".*layers\.\d+\.{attn}"
    return rf".*layers\.\d+\.({attn}|mlp\.(gate_proj|up_proj|down_proj))"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local path or HF id of the BASE model")
    ap.add_argument("--train", default=str(FT_DIR / "data" / "train.jsonl"))
    ap.add_argument("--val", default=str(FT_DIR / "data" / "val.jsonl"))
    ap.add_argument("--out", default=str(FT_DIR / "outputs" / "legal-lora"))
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--attn", default="auto", help="auto | flash_attention_2 | sdpa | eager")
    ap.add_argument("--max-len", type=int, default=8192)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=2, help="per-device micro batch")
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--eval-samples", type=int, default=300, help="cap val examples used during training")
    ap.add_argument("--patience", type=int, default=4, help="early-stopping patience (evals)")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--report-to", default="none", help="none | tensorboard | wandb")
    ap.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in --out")
    ap.add_argument("--limit", type=int, default=0, help="debug: only use first N train examples")
    ap.add_argument("--max-steps", type=int, default=-1, help="debug: stop after N steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, Trainer, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tok = load_tokenizer(args.model)
    train_ds = SFTDataset(args.train, tok, args.max_len, args.limit, "train")
    val_ds = SFTDataset(args.val, tok, args.max_len, args.eval_samples, "val")
    if len(train_ds) == 0:
        sys.exit("No training examples. Did you run 03_build_dataset.py?")

    model, arch = load_model(args.model, args.quant, args.attn)
    if arch["is_vl"]:
        probe_text_only(model, tok)

    grad_ckpt = not args.no_grad_ckpt
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    elif grad_ckpt:
        model.enable_input_require_grads()  # needed so checkpointed blocks get grads with frozen embeddings

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_target_regex(arch["is_moe"]), use_rslora=True,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    eff_bs = args.batch_size * args.grad_accum * int(os.environ.get("WORLD_SIZE", 1))
    steps_per_epoch = max(1, len(train_ds) // eff_bs)
    print(f"effective batch {eff_bs} | ~{steps_per_epoch} steps/epoch | ~{int(steps_per_epoch * args.epochs)} total")

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    use_tf32 = use_bf16

    targs = make_training_args(
        output_dir=args.out,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_tf32,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        group_by_length=True,
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=3,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0 if os.name == "nt" else 2,
        remove_unused_columns=False,
        report_to=args.report_to,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(tok.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    resume = None
    if args.resume and Path(args.out).exists():
        resume = get_last_checkpoint(args.out)
        print(f"Resuming from: {resume}")

    trainer.train(resume_from_checkpoint=resume)

    final_dir = Path(args.out) / "final"
    trainer.save_model(str(final_dir))  # LoRA adapter only (best checkpoint)
    tok.save_pretrained(str(final_dir))
    (final_dir / "train_config.json").write_text(
        json.dumps({**vars(args), **arch, "train_examples": len(train_ds)}, indent=2), encoding="utf-8")
    metrics = trainer.evaluate()
    print("final eval:", metrics)
    print(f"\nAdapter saved to {final_dir}\nNext:\n  python finetune/05_eval_and_merge.py "
          f"--model {args.model} --adapter {final_dir} --merge-to {Path(args.out) / 'merged'}")


if __name__ == "__main__":
    main()
