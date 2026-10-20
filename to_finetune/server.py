"""
server.py - Production-Ready Legal LLM Inference Server for HPC Cluster
Exposes OpenAI-compatible REST API endpoints (/v1/chat/completions, /v1/models)
and a lightweight /generate and /health endpoint over FastAPI & Uvicorn.
"""

import argparse
import asyncio
import gc
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

# Path Resolvers
CLUSTER_USER = os.environ.get("USER", "ishan.jawale.pg26")
USER_HOME = Path(os.environ.get("HOME", f"/home/{CLUSTER_USER}"))
SCRATCH_DIR = Path(os.environ.get("SCRATCH", f"/scratch/{CLUSTER_USER}"))

def resolve_base_model_path(tag: str, explicit: Optional[str] = None) -> Path:
    if explicit and Path(explicit).exists():
        return Path(explicit)
    size_str = "4B" if "4b" in tag.lower() else "8B"
    candidates = [
        SCRATCH_DIR / "models" / f"Qwen3-VL-{size_str}-Instruct",
        Path(f"/scratch/ishan.jawale.pg26/models/Qwen3-VL-{size_str}-Instruct"),
        USER_HOME / "models" / f"Qwen3-VL-{size_str}-Instruct",
        SCRATCH_DIR / "models" / f"Qwen3-{size_str}",
    ]
    for c in candidates:
        if c.exists() and ((c / "config.json").exists() or any(c.glob("*.json"))):
            return c
    raise FileNotFoundError(f"Could not locate base model for {size_str}. Searched: {[str(c) for c in candidates]}")

def resolve_adapter_path(tag: str, explicit: Optional[str] = None) -> Optional[Path]:
    if explicit:
        p = Path(explicit)
        if p.exists():
            return p
        return None
    tag_lower = tag.lower()
    if tag_lower in ["none", "base", ""]:
        return None

    if "1ep" in tag_lower:
        candidates = [
            USER_HOME / "legal_lora_out" / "legal-lora-vl-4b-1ep" / "final_adapter",
            SCRATCH_DIR / "legal_lora" / "legal-lora-vl-4b-1ep" / "final",
            Path("/home/ishan.jawale.pg26/legal_lora_out/legal-lora-vl-4b-1ep/final_adapter"),
        ]
    elif "4ep" in tag_lower:
        candidates = [
            SCRATCH_DIR / "legal_lora" / "legal-lora-vl-4b-4ep" / "final",
            USER_HOME / "legal_lora_out" / "legal-lora-vl-4b-4ep" / "final_adapter",
            USER_HOME / "legal_lora_out" / "legal-lora-vl-4b-4ep" / "checkpoint-2800",
        ]
    elif "8b" in tag_lower:
        candidates = [
            USER_HOME / "legal_lora_out" / "legal-lora-vl-8b" / "final_adapter",
            SCRATCH_DIR / "legal_lora" / "legal-lora-vl-8b" / "final",
            Path("/home/ishan.jawale.pg26/legal_lora_out/legal-lora-vl-8b/final_adapter"),
        ]
    else:  # default 4b 2-epoch
        candidates = [
            USER_HOME / "legal_lora_out" / "legal-lora-vl-4b" / "final_adapter",
            SCRATCH_DIR / "legal_lora" / "legal-lora-vl-4b" / "final",
            Path("/home/ishan.jawale.pg26/legal_lora_out/legal-lora-vl-4b/final_adapter"),
        ]

    for c in candidates:
        if c.exists() and ((c / "adapter_config.json").exists() or (c / "adapter_model.bin").exists() or (c / "adapter_model.safetensors").exists()):
            return c
    return None

# ======================================================================
# GLOBAL STATE & FASTAPI APP
# ======================================================================
app = FastAPI(
    title="Legal LLM HPC Inference Server",
    description="High-throughput on-premise inference server running fine-tuned Qwen3-VL models on Plaksha HPC.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None
processor = None
tokenizer = None
model_name_active = "qwen3-vl-legal"
inference_lock = asyncio.Lock()


# ======================================================================
# PYDANTIC SCHEMAS (OpenAI Compatible)
# ======================================================================
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: Optional[str] = "qwen3-vl-legal"
    messages: List[ChatMessage]
    max_tokens: Optional[int] = Field(default=1024, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=1.1, ge=1.0, le=2.0)
    stream: Optional[bool] = False

class SimpleGenerateRequest(BaseModel):
    prompt: str
    max_tokens: Optional[int] = Field(default=512, ge=1, le=4096)
    temperature: Optional[float] = Field(default=0.2, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=0.9, ge=0.0, le=1.0)


# ======================================================================
# MODEL LOADING FUNCTION
# ======================================================================
def initialize_model(model_tag: str, adapter_tag: str, explicit_base: Optional[str], explicit_adapter: Optional[str], quant: str):
    global model, processor, tokenizer, model_name_active
    from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from peft import PeftModel

    base_path = resolve_base_model_path(model_tag, explicit_base)
    adapter_path = resolve_adapter_path(adapter_tag, explicit_adapter)

    print("=" * 70)
    print(f"Initializing Model on Cluster:")
    print(f"  Base Model Path: {base_path}")
    print(f"  Adapter Path:    {adapter_path if adapter_path else 'None (Base Model Only)'}")
    print(f"  Device:          CUDA ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 70)

    cfg = AutoConfig.from_pretrained(str(base_path), trust_remote_code=False)
    is_vl = "vl" in cfg.model_type.lower()
    AutoCls = AutoModelForImageTextToText if is_vl else AutoModelForCausalLM

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    load_kwargs = {
        "torch_dtype": dtype,
        "device_map": {"": 0} if torch.cuda.is_available() else {"": "cpu"},
        "low_cpu_mem_usage": True,
    }

    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )

    print("Loading base weights...")
    t0 = time.time()
    loaded_model = AutoCls.from_pretrained(str(base_path), **load_kwargs)

    if adapter_path:
        print(f"Attaching LoRA adapter from {adapter_path}...")
        loaded_model = PeftModel.from_pretrained(loaded_model, str(adapter_path))
        print("Merging LoRA weights for zero-overhead inference...")
        try:
            loaded_model = loaded_model.merge_and_unload()
            print("Successfully merged LoRA adapter into base model weights!")
        except Exception as e:
            print(f"Notice: Running directly via PeftModel (merge skipped: {e})")

    loaded_model.eval()

    try:
        loaded_proc = AutoProcessor.from_pretrained(str(base_path))
    except Exception:
        loaded_proc = None

    try:
        loaded_tok = AutoTokenizer.from_pretrained(str(base_path))
        if loaded_tok.pad_token is None:
            loaded_tok.pad_token = loaded_tok.eos_token
    except Exception:
        loaded_tok = loaded_proc.tokenizer if hasattr(loaded_proc, "tokenizer") else None

    # Warmup
    print("Executing warmup inference...")
    warmup_text = "Summarize the legal principle of prima facie evidence in Indian civil law."
    inputs = (loaded_proc(text=[warmup_text], return_tensors="pt") if hasattr(loaded_proc, "tokenizer") else loaded_tok(warmup_text, return_tensors="pt")).to(device)
    with torch.no_grad():
        _ = loaded_model.generate(**inputs, max_new_tokens=15, do_sample=False)

    model = loaded_model
    processor = loaded_proc
    tokenizer = loaded_tok
    active_tag = f"qwen3-vl-{model_tag.lower()}"
    if adapter_path:
        active_tag += f"-lora-{adapter_tag.lower()}"
    model_name_active = active_tag

    print(f"Model ready in {time.time() - t0:.2f}s! Active Identifier: {model_name_active}")
    print("=" * 70)


# ======================================================================
# ENDPOINTS
# ======================================================================
@app.get("/health")
@app.get("/")
def health():
    vram_gb = round(torch.cuda.memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    vram_peak = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    return {
        "status": "online",
        "active_model": model_name_active,
        "device": gpu_name,
        "vram_allocated_gb": vram_gb,
        "vram_peak_gb": vram_peak,
        "host": os.environ.get("HOSTNAME", "compute-node"),
    }

@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [
            {
                "id": model_name_active,
                "object": "model",
                "created": int(time.time()),
                "owned_by": "plaksha-hpc",
                "root": model_name_active,
            },
            {
                "id": "qwen3-vl-legal",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "plaksha-hpc",
                "root": model_name_active,
            }
        ]
    }

from transformers import LogitsProcessor, LogitsProcessorList
import re


def clean_v3_meta_artifacts(text: str) -> str:
    """Strip teacher-prompt meta-commentary leaked from Indian-legal-data-v3."""
    if not text:
        return ""
    meta_patterns = [
        r"(?:#+\s*)?This improved answer is significantly more detailed[\s\S]*$",
        r"(?:#+\s*)?The answer remains strictly grounded in the original context[\s\S]*$",
        r"(?:#+\s*)?The answer does not remove correct information[\s\S]*$",
        r"(?:#+\s*)?This response is strictly grounded in[\s\S]*$",
    ]
    cleaned = text
    for pat in meta_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()
    cleaned = re.sub(r"\s*#+\s*$", "", cleaned).strip()
    return cleaned


class SentenceBoundaryEOSLogitsProcessor(LogitsProcessor):
    """
    Prevents premature mid-sentence <eos> / <|im_end|> emission caused by the
    512-token truncation artifact present in ~35% of Indian-legal-data-v3.
    Allows EOS freely at any valid sentence boundary (., !, ?, }, `).
    """
    def __init__(self, prompt_len: int, max_new_tokens: int, tok, eos_ids: list[int]):
        self.prompt_len = prompt_len
        self.max_new_tokens = max_new_tokens
        self.tok = tok
        self.eos_ids = [eid for eid in eos_ids if eid is not None]

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        gen_len = input_ids.shape[1] - self.prompt_len
        if gen_len < 10 or gen_len >= self.max_new_tokens - 15:
            return scores

        # Only decode tail if an EOS token is about to be sampled
        top_tok = int(torch.argmax(scores[0]).item())
        if top_tok in self.eos_ids:
            tail_text = self.tok.decode(input_ids[0, -4:], skip_special_tokens=True).rstrip()
            if not tail_text.endswith((".", "!", "?", '"', "'", ")", "]", "}", "`")):
                for eid in self.eos_ids:
                    scores[:, eid] = -float("inf")
        return scores


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    # Convert messages list to chat format
    messages = [{"role": m.role, "content": m.content} for m in req.messages]

    # Format using tokenizer / processor chat template
    tok = tokenizer if tokenizer is not None else getattr(processor, "tokenizer", None)
    if hasattr(tok, "apply_chat_template"):
        prompt_text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        # Fallback manual format
        prompt_lines = []
        for m in messages:
            prompt_lines.append(f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>")
        prompt_lines.append("<|im_start|>assistant\n")
        prompt_text = "\n".join(prompt_lines)

    device = next(model.parameters()).device
    inputs = (processor(text=[prompt_text], return_tensors="pt") if hasattr(processor, "tokenizer") else tok(prompt_text, return_tensors="pt")).to(device)

    prompt_tokens = inputs["input_ids"].shape[1]
    eos_ids = [tok.eos_token_id]
    im_end_id = tok.convert_tokens_to_ids("<|im_end|>") if hasattr(tok, "convert_tokens_to_ids") else None
    if isinstance(im_end_id, int) and im_end_id not in eos_ids:
        eos_ids.append(im_end_id)

    logits_proc = LogitsProcessorList([
        SentenceBoundaryEOSLogitsProcessor(prompt_tokens, req.max_tokens, tok, eos_ids)
    ])
    t0 = time.time()

    async with inference_lock:
        with torch.no_grad():
            pad_id = tok.pad_token_id if tok and tok.pad_token_id is not None else tok.eos_token_id
            gen_out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=(req.temperature > 0.0),
                temperature=max(0.01, req.temperature) if req.temperature > 0.0 else None,
                top_p=req.top_p if req.temperature > 0.0 else None,
                repetition_penalty=req.repetition_penalty,
                pad_token_id=pad_id,
                logits_processor=logits_proc,
            )

    elapsed = time.time() - t0
    new_toks = gen_out[0][prompt_tokens:]
    completion_text = clean_v3_meta_artifacts(tok.decode(new_toks, skip_special_tokens=True).strip())
    completion_tokens = len(new_toks)

    return {
        "id": f"chatcmpl-{uuid.uuid4().hex[:12]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": req.model or model_name_active,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": completion_text,
                },
                "finish_reason": "stop" if completion_tokens < req.max_tokens else "length",
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
        "performance": {
            "latency_sec": round(elapsed, 3),
            "tokens_per_sec": round(completion_tokens / max(0.001, elapsed), 2),
        }
    }

@app.post("/generate")
async def generate_simple(req: SimpleGenerateRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is still initializing.")

    tok = tokenizer if tokenizer is not None else getattr(processor, "tokenizer", None)
    device = next(model.parameters()).device
    inputs = (processor(text=[req.prompt], return_tensors="pt") if hasattr(processor, "tokenizer") else tok(req.prompt, return_tensors="pt")).to(device)

    prompt_tokens = inputs["input_ids"].shape[1]
    t0 = time.time()

    async with inference_lock:
        with torch.no_grad():
            pad_id = tok.pad_token_id if tok and tok.pad_token_id is not None else tok.eos_token_id
            gen_out = model.generate(
                **inputs,
                max_new_tokens=req.max_tokens,
                do_sample=(req.temperature > 0.0),
                temperature=max(0.01, req.temperature) if req.temperature > 0.0 else None,
                top_p=req.top_p if req.temperature > 0.0 else None,
                pad_token_id=pad_id,
            )

    elapsed = time.time() - t0
    new_toks = gen_out[0][prompt_tokens:]
    text = tok.decode(new_toks, skip_special_tokens=True).strip()
    completion_tokens = len(new_toks)

    return {
        "text": text,
        "tokens": completion_tokens,
        "latency_sec": round(elapsed, 3),
        "tokens_per_sec": round(completion_tokens / max(0.001, elapsed), 2),
    }


# ======================================================================
# CLI ENTRY POINT
# ======================================================================
def main():
    parser = argparse.ArgumentParser(description="Legal LLM HPC Inference Server")
    parser.add_argument("--host", default="0.0.0.0", help="Binding host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Binding port (default 8000)")
    parser.add_argument("--model-tag", default="4b", choices=["4b", "8b"], help="Model size tag")
    parser.add_argument("--adapter", default="4b-1ep", choices=["none", "base", "4b", "4b-1ep", "4b-4ep", "8b"], help="Adapter tag")
    parser.add_argument("--base-path", default=None, help="Explicit path to base model")
    parser.add_argument("--adapter-path", default=None, help="Explicit path to adapter")
    parser.add_argument("--quant", default="none", choices=["none", "4bit"], help="Quantization (default: none / bf16)")
    args = parser.parse_args()

    # Load model synchronously before starting uvicorn
    initialize_model(
        model_tag=args.model_tag,
        adapter_tag=args.adapter,
        explicit_base=args.base_path,
        explicit_adapter=args.adapter_path,
        quant=args.quant,
    )

    print(f"\n>>> Server starting on http://{args.host}:{args.port}")
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")

if __name__ == "__main__":
    main()
