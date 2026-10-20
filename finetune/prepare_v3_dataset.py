"""
prepare_v3_dataset.py - Convert Indian-legal-data-v3 into train.jsonl and val.jsonl

Converts {"instruction": ..., "output": ...} format into chat messages:
{"messages": [{"role": "user", "content": ...}, {"role": "assistant", "content": ...}]}

Features:
- Samples a configurable number of high-quality examples (default: 40,000 for fast, high-coverage training, or all 192k)
- Deduplicates and filters empty/garbled text
- Creates a 95% train / 5% val split with fixed seed
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "finetuning_data" / "Indian-legal-data-v3" / "Context Below 4096" / "below_4096_tokens.jsonl"
OUT_DIR = Path(__file__).resolve().parent / "data"

def main():
    parser = argparse.ArgumentParser(description="Prepare Indian-legal-data-v3 for 04_train.py")
    parser.add_argument("--input", default=str(INPUT_PATH), help="Path to below_4096_tokens.jsonl")
    parser.add_argument("--out-dir", default=str(OUT_DIR), help="Output directory for train/val.jsonl")
    parser.add_argument("--max-samples", type=int, default=40000, help="Max total samples to use (0 = all 192k)")
    parser.add_argument("--val-ratio", type=float, default=0.05, help="Fraction for validation split")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        sys.exit(f"Error: Could not find input dataset at {in_path}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading from {in_path} ...")
    samples = []
    seen = set()

    with open(in_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue

            inst = str(row.get("instruction", "")).strip()
            out = str(row.get("output", "")).strip()

            if len(inst) < 10 or len(out) < 20:
                continue

            h = hash((inst[:120], out[:120]))
            if h in seen:
                continue
            seen.add(h)

            item = {
                "id": f"v3_{idx}",
                "messages": [
                    {"role": "user", "content": inst},
                    {"role": "assistant", "content": out}
                ]
            }
            samples.append(item)

    print(f"Loaded {len(samples)} clean, unique samples.")

    rng = random.Random(args.seed)
    rng.shuffle(samples)

    if args.max_samples > 0 and len(samples) > args.max_samples:
        samples = samples[:args.max_samples]
        print(f"Subsampled to {len(samples)} examples as requested.")

    val_count = max(100, int(len(samples) * args.val_ratio))
    val_set = samples[:val_count]
    train_set = samples[val_count:]

    train_path = out_dir / "train.jsonl"
    val_path = out_dir / "val.jsonl"

    with open(train_path, "w", encoding="utf-8") as f:
        for s in train_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    with open(val_path, "w", encoding="utf-8") as f:
        for s in val_set:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"\n[Done] Successfully saved:")
    print(f"  Train: {train_path} ({len(train_set):,} examples)")
    print(f"  Val  : {val_path} ({len(val_set):,} examples)")

if __name__ == "__main__":
    main()
