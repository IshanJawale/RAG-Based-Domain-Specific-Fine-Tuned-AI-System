"""
Merge a LoRA adapter into its Qwen3-VL base and convert it to the Q4_K_M GGUF that llama-server runs.

    python scripts/build_gguf.py 4b
    python scripts/build_gguf.py 8b
    python scripts/build_gguf.py 4b --base   # no adapter: plain Instruct model

Writes models/gguf/legal-qwen3-vl-{4b,8b}.Q4_K_M.gguf (or qwen3-vl-{4b,8b}-instruct.Q4_K_M.gguf with --base),
the files src/llm.py serves. Only needed after retraining an adapter: setup.sh downloads the finished GGUFs.
Needs the base model and adapter in models/, and tools/llama.cpp (converter + llama-quantize).
Merging runs on CPU in bf16. The GGUF is text-only (the legal chat never sends images).
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
LLAMA = ROOT / "tools" / "llama.cpp"
QUANTIZE = LLAMA / ("llama-quantize.exe" if os.name == "nt" else "llama-quantize")
SIZES = {"4b": "Qwen3-VL-4B-Instruct", "8b": "Qwen3-VL-8B-Instruct"}


def main(size: str, base_only: bool):
    base_dir = MODELS / SIZES[size]
    adapter_dir = MODELS / f"legal-lora-vl-{size}-adapter"
    merged_dir = MODELS / f"_merged-vl-{size}"
    name = f"qwen3-vl-{size}-instruct" if base_only else f"legal-qwen3-vl-{size}"
    out = MODELS / "gguf" / f"{name}.Q4_K_M.gguf"

    print(f"{'Loading' if base_only else 'Merging ' + adapter_dir.name + ' into'} {base_dir.name} ...", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(str(base_dir), dtype=torch.bfloat16, device_map="cpu", low_cpu_mem_usage=True)
    if not base_only:
        model = PeftModel.from_pretrained(model, str(adapter_dir)).merge_and_unload()
    model.save_pretrained(str(merged_dir), safe_serialization=True)
    for f in base_dir.iterdir():  # tokenizer / processor / template files
        if f.suffix in {".json", ".txt", ".jinja"} and f.name != "model.safetensors.index.json" and not (merged_dir / f.name).exists():
            shutil.copy(f, merged_dir / f.name)
    del model

    f16 = merged_dir / "f16.gguf"
    print("Converting to GGUF and quantizing to Q4_K_M ...", flush=True)
    subprocess.run([sys.executable, str(LLAMA / "convert_hf_to_gguf.py"), str(merged_dir), "--outtype", "f16", "--outfile", str(f16)], check=True)
    out.parent.mkdir(exist_ok=True)
    subprocess.run([str(QUANTIZE), str(f16), str(out), "Q4_K_M"], check=True)
    shutil.rmtree(merged_dir)
    print(f"Done: {out}")


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if a != "--base"]
    if len(args) != 1 or args[0] not in SIZES:
        sys.exit(__doc__)
    main(args[0], "--base" in sys.argv)
