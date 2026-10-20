"""
modeling.py - model loading shared by 04_train.py and 05_eval_and_merge.py.

Handles:
  * dense Qwen3, MoE Qwen3-30B-A3B, and Qwen3-VL (dense or MoE) checkpoints
  * bf16 (recommended on H200 / 96 GB cards) or 4-bit NF4 (smaller cards)
  * transformers 4.57 -> 5.x API differences (dtype / torch_dtype)
  * text-only forward on VL models (some versions want `mm_token_type_ids`)
"""

import importlib.util
import os

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

_NEEDS_MM_IDS = False


def _find_key(d, keys):
    """Recursively look for any of `keys` with a truthy value in a nested dict."""
    if isinstance(d, dict):
        for k, v in d.items():
            if k in keys and v:
                return v
            found = _find_key(v, keys)
            if found:
                return found
    return None


def detect_arch(model_path: str) -> dict:
    cfg = AutoConfig.from_pretrained(model_path, trust_remote_code=False)
    cfg_dict = cfg.to_dict()
    model_type = cfg_dict.get("model_type", "")
    return {
        "model_type": model_type,
        "is_vl": "vl" in model_type,
        "is_moe": bool(_find_key(cfg_dict, {"num_experts", "num_local_experts", "n_routed_experts"})),
    }


def pick_attn_impl(requested: str) -> str:
    if requested != "auto":
        return requested
    return "flash_attention_2" if importlib.util.find_spec("flash_attn") else "sdpa"


def load_tokenizer(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model_path: str, quant: str = "none", attn: str = "auto"):
    """Load the base model onto this process's GPU. Returns (model, arch_info)."""
    arch = detect_arch(model_path)
    if arch["is_vl"]:
        from transformers import AutoModelForImageTextToText as AutoCls
    else:
        AutoCls = AutoModelForCausalLM

    if torch.cuda.is_available():
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        target_device = {"": local_rank}
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        target_device = {"": "cpu"}
        dtype = torch.float32

    kwargs = dict(device_map=target_device, attn_implementation=pick_attn_impl(attn))

    if quant == "4bit":
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
            llm_int8_skip_modules=["visual", "lm_head"] if arch["is_vl"] else ["lm_head"],
        )

    print(f"Loading {model_path}  [{arch}]  quant={quant}  attn={kwargs['attn_implementation']}")
    try:
        model = AutoCls.from_pretrained(model_path, dtype=dtype, **kwargs)
    except TypeError:  # older transformers: `dtype` not accepted yet
        model = AutoCls.from_pretrained(model_path, torch_dtype=dtype, **kwargs)
    model.config.use_cache = False
    return model, arch


def probe_text_only(model, tokenizer) -> bool:
    """
    Qwen3-VL forward() on pure text works on some transformers versions and wants
    an all-zeros `mm_token_type_ids` on others. Find out once, remember the answer.
    """
    global _NEEDS_MM_IDS
    ids = tokenizer("hello world", return_tensors="pt").to(model.device)
    last = None
    for flag in (False, True):
        try:
            kw = dict(ids)
            if flag:
                kw["mm_token_type_ids"] = torch.zeros_like(ids["input_ids"])
            with torch.no_grad():
                model(**kw)
            _NEEDS_MM_IDS = flag
            print(f"[probe] text-only forward OK (mm_token_type_ids needed: {flag})")
            return flag
        except Exception as e:  # noqa: BLE001
            last = e
    raise RuntimeError(f"Text-only forward failed on this VL model: {last}")


def add_mm_ids(batch: dict) -> dict:
    if _NEEDS_MM_IDS and "mm_token_type_ids" not in batch:
        batch["mm_token_type_ids"] = torch.zeros_like(batch["input_ids"])
    return batch


def chat_prompt_text(tokenizer, user_prompt: str) -> str:
    """User turn + assistant generation prompt, thinking OFF (matches src/llm.py inference)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False,
    )
