"""
train.py - LoRA / QLoRA supervised fine-tuning, built for a SHORT-WALLTIME Slurm cluster.

Designed so a job can be killed / time out at any moment and the NEXT job continues exactly where it stopped:

  * checkpoints (weights + optimizer + scheduler + RNG + data position) every --ckpt-every steps
  * auto-resume from the newest checkpoint in --out (disable with --no-resume)
  * --time-budget-min : stops cleanly (saves a checkpoint first) BEFORE the Slurm wall-time is hit
  * SIGUSR1 / SIGTERM (Slurm --signal, scancel, pre-emption) -> save a checkpoint at the next step, then exit
  * progress.json    : machine-readable status (step, epoch, losses, ETA, best ckpt, status=running/paused_*/completed/failed)
  * --sync-dir       : after every checkpoint the small LoRA adapter + progress.json are copied there
                       (put it in /home: /scratch is purged without warning)

Training recipe: loss ONLY on assistant tokens, LoRA on all attention + MLP projections (rsLoRA), cosine LR with warmup,
grad clipping, bf16, gradient checkpointing, length-grouped batches, eval loss + early stopping + best-checkpoint restore.
Examples longer than --max-len are DROPPED (never truncated: truncation would cut the answer).

Typical (48 GB RTX PRO 6000 slice, 8B model, bf16 LoRA):
    python train.py --model /path/to/Qwen3-8B --out /scratch/$USER/legal_lora --sync-dir ~/legal_lora_out
"""

import argparse
import inspect
import json
import math
import os
import shutil
import signal
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

T_START = time.time()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from modeling import (  # noqa: E402
    add_mm_ids, chat_prompt_text, load_model, load_tokenizer, probe_text_only,
)

IM_END = "<|im_end|>"
STOP = {"reason": None}  # set by signal handlers, read by RunMonitor


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------
class SFTDataset(Dataset):
    """Pre-tokenised examples; labels are -100 on the prompt, real ids on the answer."""

    def __init__(self, path, tokenizer, max_len, limit=0, name="data"):
        self.items, self.lengths = [], []
        dropped, total_tok, label_tok = 0, 0, 0
        rows = [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        if limit:
            rows = rows[:limit]
        t0 = time.time()
        for i, r in enumerate(rows):
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
            self.lengths.append(len(ids))
            total_tok += len(ids)
            label_tok += len(a_ids)
            if (i + 1) % 10000 == 0:
                print(f"  [{name}] tokenised {i + 1}/{len(rows)} ({time.time() - t0:.0f}s)", flush=True)
        n = max(1, len(self.items))
        print(f"[{name}] {len(self.items)} examples kept, {dropped} dropped (> {max_len} tokens) | "
              f"mean len {total_tok / n:.0f} tok | trained-on tokens {100 * label_tok / max(1, total_tok):.1f}%",
              flush=True)

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


class LengthGroupedEpochSampler(Sampler):
    """
    Shuffle -> cut into mega-batches -> sort each by length -> split into micro-batches -> shuffle micro-batches.
    Less padding (faster) without making batches monotonic in length. Deterministic per (seed, epoch), so a
    resumed run replays exactly the same order and the Trainer can skip the batches already consumed.
    (Re-implemented because newer transformers silently dropped `group_by_length`.)
    """

    def __init__(self, lengths, micro_bs, seed, mega_mult=50):
        self.lengths, self.micro_bs, self.seed, self.mega = lengths, max(1, micro_bs), seed, max(1, micro_bs) * mega_mult
        self.epoch, self._explicit = 0, False
        print(f"[sampler] length-grouped sampler active (micro_bs={self.micro_bs}, mega={self.mega})", flush=True)

    def set_epoch(self, epoch):
        self.epoch, self._explicit = int(epoch), True

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(len(self.lengths), generator=g).tolist()
        out = []
        for i in range(0, len(perm), self.mega):
            chunk = sorted(perm[i:i + self.mega], key=lambda j: -self.lengths[j])
            batches = [chunk[k:k + self.micro_bs] for k in range(0, len(chunk), self.micro_bs)]
            for b in torch.randperm(len(batches), generator=g).tolist():
                out.extend(batches[b])
        if not self._explicit:
            self.epoch += 1
        return iter(out)


# ----------------------------------------------------------------------
# Progress file / time budget / checkpoint sync
# ----------------------------------------------------------------------
def atomic_write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=float), encoding="utf-8")
    os.replace(tmp, path)


def _install_signal_handlers():
    def handler(signum, _frame):
        STOP["reason"] = {getattr(signal, "SIGUSR1", -1): "signal_usr1", signal.SIGTERM: "signal_term"}.get(
            signum, f"signal_{signum}")
        print(f"\n[signal] {STOP['reason']} received -> will checkpoint at the next step and exit", flush=True)

    for name in ("SIGUSR1", "SIGTERM", "SIGINT"):
        if hasattr(signal, name):
            signal.signal(getattr(signal, name), handler)


def make_monitor_cls():
    from transformers import TrainerCallback

    class RunMonitor(TrainerCallback):
        """Writes progress.json, enforces the time budget / signals, mirrors adapters to --sync-dir."""

        def __init__(self, out_dir, sync_dir, budget_s, meta):
            self.out_dir, self.sync_dir, self.budget_s, self.meta = Path(out_dir), sync_dir, budget_s, dict(meta)
            self.stop_reason = None
            self.start_step = 0
            self.prev = {}
            p = self.out_dir / "progress.json"
            if p.exists():
                try:
                    self.prev = json.loads(p.read_text(encoding="utf-8"))
                except Exception:  # noqa: BLE001
                    self.prev = {}
            self.prev_elapsed = float(self.prev.get("total_elapsed_sec", 0.0))
            self.sessions = int(self.prev.get("sessions", 0)) + 1

        # ---- snapshot -------------------------------------------------
        def snapshot(self, state, status, note="", extra=None):
            session = time.time() - T_START
            done_this = max(0, state.global_step - self.start_step)
            remaining = max(0, (state.max_steps or 0) - state.global_step)
            eta = (session / done_this * remaining) if done_this > 0 else None
            logs = state.log_history or []
            train_logs = [l for l in logs if "loss" in l]
            evals_by_step = {l["step"]: {"step": l["step"], "epoch": round(float(l.get("epoch", 0)), 4),
                                         "eval_loss": float(l["eval_loss"])}
                             for l in logs if "eval_loss" in l}
            evals = [evals_by_step[k] for k in sorted(evals_by_step)]
            d = {
                "status": status,
                "note": note,
                "global_step": state.global_step,
                "max_steps": state.max_steps,
                "percent_done": round(100.0 * state.global_step / max(1, state.max_steps or 1), 2),
                "epoch": round(float(state.epoch or 0.0), 4),
                "num_train_epochs": self.meta.get("epochs"),
                "latest_train_loss": float(train_logs[-1]["loss"]) if train_logs else None,
                "latest_learning_rate": float(train_logs[-1]["learning_rate"]) if train_logs and "learning_rate" in train_logs[-1] else None,
                "latest_grad_norm": float(train_logs[-1]["grad_norm"]) if train_logs and "grad_norm" in train_logs[-1] else None,
                "latest_eval_loss": evals[-1]["eval_loss"] if evals else None,
                "best_eval_loss": float(state.best_metric) if state.best_metric is not None else None,
                "best_checkpoint": state.best_model_checkpoint,
                "last_checkpoint": self._last_ckpt(),
                "eval_history": evals,
                "train_loss_tail": [{"step": l["step"], "loss": float(l["loss"])} for l in train_logs[-30:]],
                "sessions": self.sessions,
                "session_elapsed_sec": round(session, 1),
                "total_elapsed_sec": round(self.prev_elapsed + session, 1),
                "eta_this_session_sec": round(eta, 0) if eta else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "config": self.meta,
            }
            if extra:
                d.update(extra)
            atomic_write_json(self.out_dir / "progress.json", d)
            if self.sync_dir:
                try:
                    atomic_write_json(Path(self.sync_dir) / "progress.json", d)
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] could not write progress to sync dir: {e}", flush=True)

        def _last_ckpt(self):
            cks = sorted(self.out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1])) \
                if self.out_dir.exists() else []
            return cks[-1].name if cks else None

        # ---- callbacks -------------------------------------------------
        def on_train_begin(self, args, state, control, **kw):
            self.start_step = state.global_step
            print(f"[monitor] session #{self.sessions} starting at step {state.global_step}/{state.max_steps}", flush=True)
            self.snapshot(state, "running", note=f"session {self.sessions} started")

        def on_log(self, args, state, control, **kw):
            self.snapshot(state, "running")

        def on_evaluate(self, args, state, control, **kw):
            self.snapshot(state, "running", note="evaluated")

        def on_save(self, args, state, control, **kw):
            ck = self.out_dir / f"checkpoint-{state.global_step}"
            if self.sync_dir and ck.exists():
                try:
                    dst = Path(self.sync_dir) / "latest_adapter"
                    dst.mkdir(parents=True, exist_ok=True)
                    for f in ck.glob("adapter_*"):
                        shutil.copy2(f, dst / f.name)
                    atomic_write_json(dst / "step.json", {"global_step": state.global_step,
                                                          "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                except Exception as e:  # noqa: BLE001
                    print(f"[warn] adapter sync failed: {e}", flush=True)
            self.snapshot(state, "running", note=f"checkpoint-{state.global_step} saved")

        def on_step_end(self, args, state, control, **kw):
            reason = STOP["reason"]
            if reason is None and self.budget_s and (time.time() - T_START) >= self.budget_s:
                reason = "time_budget"
            if reason and not self.stop_reason:
                self.stop_reason = reason
                print(f"[monitor] stopping at step {state.global_step}: {reason} (saving checkpoint)", flush=True)
            if self.stop_reason:
                control.should_save = True
                control.should_training_stop = True
            return control

    return RunMonitor


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
        print(f"[warn] TrainingArguments ignored unsupported args: {dropped}", flush=True)
    return TrainingArguments(**kw)


def lora_target_regex(is_moe: bool) -> str:
    attn = r"self_attn\.(q_proj|k_proj|v_proj|o_proj)"
    if is_moe:  # experts are fused/frozen; adapt attention only
        return rf".*layers\.\d+\.{attn}"
    return rf".*layers\.\d+\.({attn}|mlp\.(gate_proj|up_proj|down_proj))"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="local path of the BASE model")
    ap.add_argument("--train", default=str(HERE / "data" / "train.jsonl"))
    ap.add_argument("--val", default=str(HERE / "data" / "val.jsonl"))
    ap.add_argument("--out", default=str(HERE / "outputs" / "legal-lora"), help="checkpoints (use /scratch)")
    ap.add_argument("--sync-dir", default=None, help="mirror adapter + progress.json here (use /home)")
    ap.add_argument("--quant", choices=["none", "4bit"], default="none")
    ap.add_argument("--attn", default="auto", help="auto | flash_attention_2 | sdpa | eager")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.03)
    ap.add_argument("--batch-size", type=int, default=4, help="per-device micro batch")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--ckpt-every", type=int, default=200, help="checkpoint + eval every N optimizer steps")
    ap.add_argument("--keep-ckpts", type=int, default=3)
    ap.add_argument("--eval-samples", type=int, default=400, help="cap val examples used during training")
    ap.add_argument("--patience", type=int, default=5, help="early-stopping patience (evals)")
    ap.add_argument("--logging-steps", type=int, default=10)
    ap.add_argument("--time-budget-min", type=float, default=0,
                    help="stop cleanly after this many minutes (set to wall-time minus ~15). 0 = unlimited")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints in --out")
    ap.add_argument("--report-to", default="none", help="none | tensorboard | wandb")
    ap.add_argument("--limit", type=int, default=0, help="debug: only use first N train examples")
    ap.add_argument("--max-steps", type=int, default=-1, help="debug: stop after N steps")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    _install_signal_handlers()

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import EarlyStoppingCallback, Trainer, set_seed
    from transformers.trainer_utils import get_last_checkpoint

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.sync_dir:
        Path(args.sync_dir).mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"GPU: {torch.cuda.get_device_name(0)} | {torch.cuda.get_device_properties(0).total_memory / 2**30:.1f} GiB",
              flush=True)

    tok = load_tokenizer(args.model)
    print("Tokenising datasets ...", flush=True)
    train_ds = SFTDataset(args.train, tok, args.max_len, args.limit, "train")
    val_ds = SFTDataset(args.val, tok, args.max_len, args.eval_samples, "val")
    if len(train_ds) == 0:
        sys.exit("No training examples found.")

    model, arch = load_model(args.model, args.quant, args.attn)
    if arch["is_vl"]:
        probe_text_only(model, tok)

    grad_ckpt = not args.no_grad_ckpt
    if args.quant == "4bit":
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=grad_ckpt,
            gradient_checkpointing_kwargs={"use_reentrant": False})
    elif grad_ckpt:
        model.enable_input_require_grads()  # checkpointed blocks need grads through frozen embeddings

    lora_cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        target_modules=lora_target_regex(arch["is_moe"]), use_rslora=True,
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    world = int(os.environ.get("WORLD_SIZE", 1))
    eff_bs = args.batch_size * args.grad_accum * world
    steps_per_epoch = max(1, math.ceil(len(train_ds) / eff_bs))
    total_steps = args.max_steps if args.max_steps > 0 else int(math.ceil(steps_per_epoch * args.epochs))
    warmup_steps = max(5, int(args.warmup_ratio * total_steps))
    print(f"effective batch {eff_bs} | {steps_per_epoch} steps/epoch | ~{total_steps} total | warmup {warmup_steps}",
          flush=True)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    targs = make_training_args(
        output_dir=str(out_dir),
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=max(1, args.batch_size),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,          # explicit int: warmup_ratio was removed in newer transformers
        weight_decay=0.01,
        max_grad_norm=1.0,
        bf16=use_bf16,
        fp16=use_fp16,
        tf32=use_bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused" if torch.cuda.is_available() else "adamw_torch",
        logging_steps=args.logging_steps,
        logging_first_step=True,
        eval_strategy="steps",
        eval_steps=args.ckpt_every,
        save_strategy="steps",
        save_steps=args.ckpt_every,          # must equal eval_steps for load_best_model_at_end
        save_total_limit=args.keep_ckpts,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=args.report_to,
        ddp_find_unused_parameters=False,
        seed=args.seed,
    )

    class GroupedTrainer(Trainer):
        def _get_train_sampler(self, *a, **k):
            try:
                return LengthGroupedEpochSampler(train_ds.lengths, self.args.per_device_train_batch_size, args.seed)
            except Exception as e:  # noqa: BLE001
                print(f"[warn] length-grouped sampler unavailable ({e}); using default sampler", flush=True)
                return super()._get_train_sampler(*a, **k)

    meta = {k: v for k, v in vars(args).items()}
    meta.update(arch)
    meta.update({"train_examples": len(train_ds), "val_examples": len(val_ds), "effective_batch": eff_bs,
                 "steps_per_epoch": steps_per_epoch, "warmup_steps": warmup_steps})
    budget_s = args.time_budget_min * 60 if args.time_budget_min > 0 else 0
    RunMonitor = make_monitor_cls()
    monitor = RunMonitor(out_dir, args.sync_dir, budget_s, meta)

    trainer = GroupedTrainer(
        model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds,
        data_collator=Collator(tok.pad_token_id),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), monitor],
    )

    resume = None
    if not args.no_resume:
        resume = get_last_checkpoint(str(out_dir)) if out_dir.exists() else None
    print(f"Resuming from: {resume}" if resume else "Starting fresh (no checkpoint found)", flush=True)

    try:
        trainer.train(resume_from_checkpoint=resume)
    except BaseException as e:  # noqa: BLE001
        monitor.snapshot(trainer.state, "failed", note=f"{type(e).__name__}: {str(e)[:400]}")
        raise

    st = trainer.state
    finished = st.global_step >= st.max_steps
    if monitor.stop_reason and not finished:
        monitor.snapshot(st, f"paused_{monitor.stop_reason}",
                         note="checkpoint saved; resubmit the same job to continue")
        print(f"\n[paused] {monitor.stop_reason} at step {st.global_step}/{st.max_steps}. "
              f"Resubmit the job to continue.", flush=True)
        return

    note = "early stopping triggered" if not finished else "reached the last step"
    final_dir = out_dir / "final"
    trainer.save_model(str(final_dir))  # LoRA adapter (best checkpoint)
    tok.save_pretrained(str(final_dir))
    (final_dir / "train_config.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")
    metrics = trainer.evaluate()
    print("final eval:", metrics, flush=True)
    if args.sync_dir:
        dst = Path(args.sync_dir) / "final_adapter"
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(final_dir, dst)
    monitor.snapshot(st, "completed", note=note,
                     extra={"final_eval": {k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))},
                            "final_adapter": str(final_dir),
                            "peak_gpu_mem_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)
                            if torch.cuda.is_available() else None})
    print(f"\nDONE. Adapter saved to {final_dir}" + (f" and {args.sync_dir}/final_adapter" if args.sync_dir else ""),
          flush=True)


if __name__ == "__main__":
    main()
