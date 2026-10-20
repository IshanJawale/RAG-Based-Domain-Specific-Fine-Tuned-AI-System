"""
merge_adapter.py - merge a trained LoRA adapter into the base model (bf16, CPU-friendly) and save a standalone model.
Run it on your laptop (needs ~2x model size in RAM: ~34 GB for an 8B model - close other apps) or on the HPC.

    python merge_adapter.py --model Models/Qwen3-8B --adapter legal_lora_out/final_adapter --out Models/Qwen3-8B-legal
    python merge_adapter.py --model ... --adapter ... --out ... --val data/val.jsonl --sample 3   # also print sample outputs

Then add the merged folder to AVAILABLE_MODELS in src/llm.py.
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from modeling import chat_prompt_text, detect_arch, load_tokenizer  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--val", default=None)
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()

    from peft import PeftModel
    arch = detect_arch(args.model)
    if arch["is_vl"]:
        from transformers import AutoModelForImageTextToText as Auto
    else:
        from transformers import AutoModelForCausalLM as Auto
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        model = Auto.from_pretrained(args.model, dtype=torch.bfloat16, device_map={"": dev})
    except TypeError:
        model = Auto.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map={"": dev})
    tok = load_tokenizer(args.model)

    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload()
    model.config.use_cache = True
    Path(args.out).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.out, safe_serialization=True, max_shard_size="5GB")
    tok.save_pretrained(args.out)
    print(f"Merged model saved to {args.out}")

    if args.val and args.sample:
        rows = [json.loads(l) for l in open(args.val, encoding="utf-8") if l.strip()][: args.sample]
        model.eval()
        for r in rows:
            enc = tok(chat_prompt_text(tok, r["messages"][0]["content"]), return_tensors="pt",
                      add_special_tokens=False).to(model.device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=256, do_sample=False)
            print("\n--- Q:", r["messages"][0]["content"][:300])
            print("--- GOLD :", r["messages"][1]["content"][:400])
            print("--- MODEL:", tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)[:400])


if __name__ == "__main__":
    main()
