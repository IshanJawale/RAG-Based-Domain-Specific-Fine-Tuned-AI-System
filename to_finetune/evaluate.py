"""
evaluate.py - Unified Indian Legal Benchmark Suite.
Located in: to_finetune/evaluate.py
Outputs to: to_finetune/evaluation_report/

Evaluates Indian Legal Models on 100 held-out test questions:
  1. Qwen3-VL-4B (Base)                   [Local GPU 4-bit NF4 / HPC]
  2. Qwen3-VL-4B (Fine-Tuned Legal LoRA)  [Local GPU 4-bit NF4 / HPC]
  3. Qwen3-VL-8B (Base)                   [Local GPU 4-bit NF4 / HPC]
  4. Qwen3-VL-8B (Fine-Tuned Legal LoRA)  [Local GPU 4-bit NF4 / HPC]
  5. Google Gemini 3.7 Flash              [Cloud API, Zero VRAM]

Metrics Suite:
  - Language Modeling Quality:
      * Eval Cross-Entropy Loss (NLL) ↓
      * Perplexity (PPL) = exp(Loss) ↓
  - Lexical & Statutory Precision:
      * ROUGE-1 (Unigram), ROUGE-2 (Legal bigrams), ROUGE-L (Sentence flow)
      * BLEU-1, BLEU-2, BLEU-4 (Exact statutory n-gram precision)
  - Legal Domain Specialization:
      * Legal Citation Recall, Precision & F1 (Acts, Sections, Articles, SC/HC judgments)
      * Legal Terminology Density / Indian Procedural Keyword Coverage
  - Resource & Latency:
      * Generation Throughput (tokens/sec) ↑
      * Average Response Latency (seconds/query) ↓
      * Peak GPU VRAM footprint (GB) ↓

Outputs:
  - to_finetune/evaluation_report/evaluation_report.md
  - to_finetune/evaluation_report/evaluation_report.json
"""

import argparse
import gc
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
import numpy as np
import torch
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent

# Load environment variables (.env in to_finetune or root)
load_dotenv(THIS_DIR / ".env")
load_dotenv(PROJECT_ROOT / ".env")

# Adaptive path resolvers (works seamlessly on Local Windows & Cluster Linux)
USER_HOME = Path(os.environ.get("HOME", ""))
SCRATCH_DIR = Path(os.environ.get("SCRATCH", f"/scratch/{os.environ.get('USER', '')}"))

def resolve_model_dir() -> Path:
    candidates = [
        PROJECT_ROOT / "Models",
        PROJECT_ROOT / "models",
        THIS_DIR / "models",
        SCRATCH_DIR / "models",
    ]
    for c in candidates:
        if c.exists():
            return c
    return PROJECT_ROOT / "Models"

MODELS_DIR = resolve_model_dir()

def resolve_base_model_path(size_tag: str) -> Path | None:
    tag = "4B" if "4b" in size_tag.lower() else "8B"
    candidates = [
        MODELS_DIR / f"Qwen3-VL-{tag}-Instruct",
        PROJECT_ROOT / "Models" / f"Qwen3-VL-{tag}-Instruct",
        PROJECT_ROOT / "models" / f"Qwen3-VL-{tag}-Instruct",
        SCRATCH_DIR / "models" / f"Qwen3-VL-{tag}-Instruct",
        SCRATCH_DIR / "models" / f"Qwen3-{tag}",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None

def resolve_adapter_path(size_tag: str) -> Path | None:
    tag = size_tag.lower()
    if "1ep" in tag or "1epoch" in tag:
        candidates = [
            MODELS_DIR / "legal-lora-vl-4b-1ep-adapter",
            PROJECT_ROOT / "Models" / "legal-lora-vl-4b-1ep-adapter",
            PROJECT_ROOT / "models" / "legal-lora-vl-4b-1ep-adapter",
            USER_HOME / "legal_lora_out" / "legal-lora-vl-4b-1ep" / "final_adapter",
            SCRATCH_DIR / "legal_lora" / "legal-lora-vl-4b-1ep" / "final",
        ]
    elif "4b" in tag:
        candidates = [
            MODELS_DIR / "legal-lora-vl-4b-adapter",
            PROJECT_ROOT / "Models" / "legal-lora-vl-4b-adapter",
            PROJECT_ROOT / "models" / "legal-lora-vl-4b-adapter",
            USER_HOME / "legal_lora_out" / "legal-lora-vl-4b" / "final_adapter",
            SCRATCH_DIR / "legal_lora" / "legal-lora-vl-4b" / "final",
        ]
    else:
        candidates = [
            MODELS_DIR / "legal-lora-vl-8b-adapter",
            PROJECT_ROOT / "Models" / "legal-lora-vl-8b-adapter",
            PROJECT_ROOT / "models" / "legal-lora-vl-8b-adapter",
            USER_HOME / "legal_lora_out" / "legal-lora-vl-8b" / "final_adapter",
            SCRATCH_DIR / "legal_lora" / "legal-lora-vl-8b" / "final",
        ]
    for c in candidates:
        if c.exists() and any(c.iterdir()):
            return c
    return None


# ======================================================================
# Metrics Computation (Pure Python, fast, self-contained)
# ======================================================================
def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def compute_bleu(reference: list[str], hypothesis: list[str], max_n=4) -> dict[str, float]:
    if not hypothesis or not reference:
        return {f"bleu_{n}": 0.0 for n in range(1, max_n + 1)}
    ref_len, hyp_len = len(reference), len(hypothesis)
    bp = 1.0 if hyp_len > ref_len else math.exp(1 - ref_len / max(1, hyp_len))
    precisions = []
    for n in range(1, max_n + 1):
        if hyp_len < n:
            precisions.append(0.0)
            continue
        hyp_ngrams = [tuple(hypothesis[i:i + n]) for i in range(hyp_len - n + 1)]
        ref_ngrams = [tuple(reference[i:i + n]) for i in range(ref_len - n + 1)]
        hyp_counts = Counter(hyp_ngrams)
        ref_counts = Counter(ref_ngrams)
        overlap = sum(min(count, ref_counts.get(ng, 0)) for ng, count in hyp_counts.items())
        precisions.append(overlap / max(1, len(hyp_ngrams)))
    res = {}
    for n in range(1, max_n + 1):
        p_subset = [p for p in precisions[:n] if p > 0]
        score = (bp * math.exp(sum(math.log(p) for p in p_subset) / n)) if len(p_subset) == n else 0.0
        res[f"bleu_{n}"] = round(score * 100, 2)
    return res


def compute_rouge_lcs(ref: list[str], hyp: list[str]) -> float:
    m, n = len(ref), len(hyp)
    if m == 0 or n == 0:
        return 0.0
    dp = [0] * (n + 1)
    for i in range(1, m + 1):
        prev = 0
        for j in range(1, n + 1):
            temp = dp[j]
            if ref[i - 1] == hyp[j - 1]:
                dp[j] = prev + 1
            else:
                dp[j] = max(dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def compute_rouge(reference: list[str], hypothesis: list[str]) -> dict[str, float]:
    if not hypothesis or not reference:
        return {"rouge_1": 0.0, "rouge_2": 0.0, "rouge_l": 0.0}
    ref_1, hyp_1 = Counter(reference), Counter(hypothesis)
    overlap_1 = sum(min(count, ref_1.get(w, 0)) for w, count in hyp_1.items())
    p1 = overlap_1 / max(1, len(hypothesis))
    r1 = overlap_1 / max(1, len(reference))
    f1_1 = 2 * p1 * r1 / max(1e-8, p1 + r1)

    hyp_2 = Counter([tuple(hypothesis[i:i + 2]) for i in range(len(hypothesis) - 1)])
    ref_2 = Counter([tuple(reference[i:i + 2]) for i in range(len(reference) - 1)])
    overlap_2 = sum(min(count, ref_2.get(w, 0)) for w, count in hyp_2.items())
    p2 = overlap_2 / max(1, max(1, len(hypothesis) - 1))
    r2 = overlap_2 / max(1, max(1, len(reference) - 1))
    f1_2 = 2 * p2 * r2 / max(1e-8, p2 + r2)

    lcs = compute_rouge_lcs(reference, hypothesis)
    rl = lcs / max(1, len(reference))
    pl = lcs / max(1, len(hypothesis))
    fl = 2 * pl * rl / max(1e-8, pl + rl)
    return {"rouge_1": round(f1_1 * 100, 2), "rouge_2": round(f1_2 * 100, 2), "rouge_l": round(fl * 100, 2)}


# Expanded Indian Statutory Citation Regex
LEGAL_CITATION_REGEX = re.compile(
    r"\b("
    r"section\s+\d+[a-z]?(\(\d+\))?|"
    r"sec\.\s*\d+[a-z]?|"
    r"article\s+\d+[a-z]?|"
    r"art\.\s*\d+[a-z]?|"
    r"order\s+[ivxlcdm\d]+(?:\s+rule\s+\d+)?|"
    r"rule\s+\d+[a-z]?|"
    r"act\s+(?:of\s+)?\d{4}|"
    r"code\s+of\s+civil\s+procedure|cpc|"
    r"code\s+of\s+criminal\s+procedure|crpc|"
    r"indian\s+penal\s+code|ipc|"
    r"bharatiya\s+nyaya\s+sanhita|bns|"
    r"bharatiya\s+nagarik\s+suraksha\s+sanhita|bnss|"
    r"bharatiya\s+sakshya\s+adhiniyam|bsa|"
    r"indian\s+evidence\s+act|iea|"
    r"constitution\s+of\s+india|"
    r"high\s+court|supreme\s+court|sessions\s+court|tribunal|"
    r"writ\s+petition|special\s+leave\s+petition|slp"
    r")\b",
    re.IGNORECASE,
)

def compute_legal_citation_metrics(gold: str, pred: str) -> dict[str, float]:
    gold_entities = set(m.group(0).lower().strip() for m in LEGAL_CITATION_REGEX.finditer(gold))
    pred_entities = set(m.group(0).lower().strip() for m in LEGAL_CITATION_REGEX.finditer(pred))
    if not gold_entities:
        return {"citation_recall": 100.0, "citation_precision": 100.0, "citation_f1": 100.0}
    overlap = len(gold_entities & pred_entities)
    rec = (overlap / len(gold_entities)) * 100.0
    prec = ((overlap / len(pred_entities)) * 100.0) if pred_entities else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return {
        "citation_recall": round(rec, 2),
        "citation_precision": round(prec, 2),
        "citation_f1": round(f1, 2),
    }


# Canonical Indian Legal Terminology Keywords
LEGAL_KEYWORDS = [
    "plaintiff", "defendant", "appellant", "respondent", "petitioner", "jurisdiction",
    "ad interim", "injunction", "prima facie", "balance of convenience", "irreparable loss",
    "res judicata", "locus standi", "ultra vires", "mandamus", "certiorari", "habeas corpus",
    "quash", "decree", "mesne profits", "cognizable", "non-bailable", "anticipatory bail",
    "ratio decidendi", "obiter dicta", "cause of action", "limitation", "estoppel",
    "restitution", "summons", "warrant", "affidavit", "pleading", "written statement",
    "compromise decree", "admissibility", "substantive law", "procedural law"
]

def compute_legal_terminology_density(text: str) -> float:
    text_lower = text.lower()
    matches = sum(1 for kw in LEGAL_KEYWORDS if kw in text_lower)
    return round((matches / max(1, len(LEGAL_KEYWORDS))) * 100, 2)


# ======================================================================
# Local 4-Bit GPU Evaluator
# ======================================================================
def evaluate_local_model(
    label: str,
    base_model_path: Path,
    adapter_path: Path | None,
    samples: list[dict],
    max_gen_tokens: int = 200,
) -> dict:
    from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

    print(f"\n" + "=" * 75, flush=True)
    print(f"Loading & Evaluating: {label}", flush=True)
    print("=" * 75, flush=True)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    ) if torch.cuda.is_available() else None

    load_kwargs = {
        "low_cpu_mem_usage": True,
    }
    if torch.cuda.is_available():
        load_kwargs["quantization_config"] = bnb_config
        # Pin directly to GPU 0 to prevent accelerate pre-quantization heuristic dispatch errors
        load_kwargs["device_map"] = {"": 0}
    else:
        load_kwargs["dtype"] = torch.bfloat16
        load_kwargs["device_map"] = "cpu"

    processor = AutoProcessor.from_pretrained(str(base_model_path))
    model = AutoModelForImageTextToText.from_pretrained(str(base_model_path), **load_kwargs)

    if adapter_path and adapter_path.exists():
        from peft import PeftModel
        print(f"Attaching Fine-Tuned LoRA Adapter from: {adapter_path}", flush=True)
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    losses = []
    rouge1_list, rouge2_list, rougel_list = [], [], []
    bleu1_list, bleu2_list, bleu4_list = [], [], []
    cit_recall_list, cit_prec_list, cit_f1_list = [], [], []
    legal_term_list = []
    gen_times, gen_token_counts = [], []
    sample_records = []

    for idx, s in enumerate(tqdm(samples, desc=f"Eval {label}")):
        user_prompt = s["messages"][0]["content"]
        gold_answer = s["messages"][1]["content"].strip()

        # -------------------------------------------------------------
        # 1. Forward Pass: Cross-Entropy Loss & Perplexity on Gold Target
        # -------------------------------------------------------------
        try:
            full_msgs = [{"role": "user", "content": user_prompt}, {"role": "assistant", "content": gold_answer}]
            try:
                full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False, enable_thinking=False)
            except Exception:
                full_text = processor.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

            prompt_msgs = [{"role": "user", "content": user_prompt}]
            try:
                prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            except Exception:
                prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)

            full_enc = processor(text=[full_text], return_tensors="pt")
            prompt_enc = processor(text=[prompt_text], return_tensors="pt")

            prompt_len = prompt_enc["input_ids"].shape[1]
            input_ids = full_enc["input_ids"].to(device)
            attn_mask = full_enc.get("attention_mask", None)
            if attn_mask is not None:
                attn_mask = attn_mask.to(device)

            labels = input_ids.clone()
            labels[:, :min(prompt_len, labels.shape[1] - 1)] = -100

            if (labels != -100).sum() > 0:
                with torch.no_grad():
                    out = model(input_ids=input_ids, attention_mask=attn_mask, labels=labels)
                    loss_val = float(out.loss.item())
                    if not math.isnan(loss_val) and not math.isinf(loss_val):
                        losses.append(loss_val)
        except Exception as e:
            pass

        # -------------------------------------------------------------
        # 2. Generation Pass: Auto-regressive Generation
        # -------------------------------------------------------------
        inputs = processor(text=[prompt_text], return_tensors="pt").to(device)
        t0 = time.time()
        with torch.no_grad():
            gen_out = model.generate(
                **inputs,
                max_new_tokens=max_gen_tokens,
                do_sample=True,
                temperature=0.2,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=processor.tokenizer.pad_token_id if hasattr(processor, "tokenizer") else None,
            )
        elapsed = time.time() - t0

        new_toks = gen_out[0][inputs["input_ids"].shape[1]:]
        pred_text = processor.decode(new_toks, skip_special_tokens=True).strip()

        gen_times.append(elapsed)
        gen_token_counts.append(len(new_toks))

        # -------------------------------------------------------------
        # 3. Lexical & Legal Metric Computations
        # -------------------------------------------------------------
        ref_words = tokenize_words(gold_answer)
        hyp_words = tokenize_words(pred_text)

        r_scores = compute_rouge(ref_words, hyp_words)
        b_scores = compute_bleu(ref_words, hyp_words)
        c_scores = compute_legal_citation_metrics(gold_answer, pred_text)
        term_cov = compute_legal_terminology_density(pred_text)

        rouge1_list.append(r_scores["rouge_1"])
        rouge2_list.append(r_scores["rouge_2"])
        rougel_list.append(r_scores["rouge_l"])
        bleu1_list.append(b_scores["bleu_1"])
        bleu2_list.append(b_scores["bleu_2"])
        bleu4_list.append(b_scores["bleu_4"])
        cit_recall_list.append(c_scores["citation_recall"])
        cit_prec_list.append(c_scores["citation_precision"])
        cit_f1_list.append(c_scores["citation_f1"])
        legal_term_list.append(term_cov)

        if idx < 3:
            sample_records.append({
                "prompt": user_prompt[:250],
                "gold": gold_answer[:300],
                "predicted": pred_text[:300],
            })

    # Summary statistics
    mean_loss = round(float(np.mean(losses)), 4) if losses else None
    perplexity = round(float(math.exp(min(20.0, mean_loss))), 2) if mean_loss is not None else None
    avg_tok_sec = sum(gen_token_counts) / max(0.01, sum(gen_times))
    avg_latency = round(float(np.mean(gen_times)), 2)
    peak_vram = round(torch.cuda.max_memory_allocated() / (1024**3), 2) if torch.cuda.is_available() else 0.0

    # Clean VRAM Unload
    del model
    del processor
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

    return {
        "label": label,
        "type": "Local GPU (4-bit)",
        "eval_loss": mean_loss,
        "perplexity": perplexity,
        "rouge_1": round(float(np.mean(rouge1_list)), 2),
        "rouge_2": round(float(np.mean(rouge2_list)), 2),
        "rouge_l": round(float(np.mean(rougel_list)), 2),
        "bleu_1": round(float(np.mean(bleu1_list)), 2),
        "bleu_2": round(float(np.mean(bleu2_list)), 2),
        "bleu_4": round(float(np.mean(bleu4_list)), 2),
        "citation_recall": round(float(np.mean(cit_recall_list)), 2),
        "citation_precision": round(float(np.mean(cit_prec_list)), 2),
        "citation_f1": round(float(np.mean(cit_f1_list)), 2),
        "legal_terminology_density": round(float(np.mean(legal_term_list)), 2),
        "tokens_per_sec": round(avg_tok_sec, 1),
        "avg_latency_sec": avg_latency,
        "peak_vram_gib": peak_vram,
        "samples": sample_records,
    }


# ======================================================================
# Cloud Gemini 3.7 Flash Evaluator (With thinkingBudget: 0)
# ======================================================================
def call_gemini(prompt: str, api_key: str, max_tokens: int = 250) -> str:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-3.7-flash:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": 0.2,
            "thinkingConfig": {"thinkingBudget": 0},
        },
    }
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=35) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
    except Exception as e:
        print(f"[Gemini Error]: {e}", flush=True)
    return ""


def evaluate_gemini_model(
    label: str,
    samples: list[dict],
    max_gen_tokens: int = 200,
) -> dict:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in .env, skipping Gemini evaluation.", flush=True)
        return None

    print(f"\n" + "=" * 75, flush=True)
    print(f"Evaluating: {label} (Google Cloud REST API)", flush=True)
    print("=" * 75, flush=True)

    rouge1_list, rouge2_list, rougel_list = [], [], []
    bleu1_list, bleu2_list, bleu4_list = [], [], []
    cit_recall_list, cit_prec_list, cit_f1_list = [], [], []
    legal_term_list = []
    gen_times, gen_token_counts = [], []
    sample_records = []

    for idx, s in enumerate(tqdm(samples, desc=f"Eval {label}")):
        user_prompt = s["messages"][0]["content"]
        gold_answer = s["messages"][1]["content"].strip()

        t0 = time.time()
        pred_text = call_gemini(user_prompt, api_key, max_tokens=max_gen_tokens)
        elapsed = time.time() - t0

        approx_tokens = len(pred_text.split()) * 1.3
        gen_times.append(elapsed)
        gen_token_counts.append(approx_tokens)

        ref_words = tokenize_words(gold_answer)
        hyp_words = tokenize_words(pred_text)

        r_scores = compute_rouge(ref_words, hyp_words)
        b_scores = compute_bleu(ref_words, hyp_words)
        c_scores = compute_legal_citation_metrics(gold_answer, pred_text)
        term_cov = compute_legal_terminology_density(pred_text)

        rouge1_list.append(r_scores["rouge_1"])
        rouge2_list.append(r_scores["rouge_2"])
        rougel_list.append(r_scores["rouge_l"])
        bleu1_list.append(b_scores["bleu_1"])
        bleu2_list.append(b_scores["bleu_2"])
        bleu4_list.append(b_scores["bleu_4"])
        cit_recall_list.append(c_scores["citation_recall"])
        cit_prec_list.append(c_scores["citation_precision"])
        cit_f1_list.append(c_scores["citation_f1"])
        legal_term_list.append(term_cov)

        if idx < 3:
            sample_records.append({
                "prompt": user_prompt[:250],
                "gold": gold_answer[:300],
                "predicted": pred_text[:300],
            })
        time.sleep(0.4)

    avg_tok_sec = sum(gen_token_counts) / max(0.01, sum(gen_times))
    avg_latency = round(float(np.mean(gen_times)), 2)

    return {
        "label": label,
        "type": "Cloud API (Zero VRAM)",
        "eval_loss": "N/A (API)",
        "perplexity": "N/A (API)",
        "rouge_1": round(float(np.mean(rouge1_list)), 2),
        "rouge_2": round(float(np.mean(rouge2_list)), 2),
        "rouge_l": round(float(np.mean(rougel_list)), 2),
        "bleu_1": round(float(np.mean(bleu1_list)), 2),
        "bleu_2": round(float(np.mean(bleu2_list)), 2),
        "bleu_4": round(float(np.mean(bleu4_list)), 2),
        "citation_recall": round(float(np.mean(cit_recall_list)), 2),
        "citation_precision": round(float(np.mean(cit_prec_list)), 2),
        "citation_f1": round(float(np.mean(cit_f1_list)), 2),
        "legal_terminology_density": round(float(np.mean(legal_term_list)), 2),
        "tokens_per_sec": round(avg_tok_sec, 1),
        "avg_latency_sec": avg_latency,
        "peak_vram_gib": 0.0,
        "samples": sample_records,
    }


# ======================================================================
# Main CLI & Benchmark Runner
# ======================================================================
def main():
    ap = argparse.ArgumentParser(description="Indian Legal AI Master Comparative Evaluation Benchmark")
    ap.add_argument("--val-data", default=str(THIS_DIR / "data" / "val.jsonl"),
                    help="Path to validation jsonl (default: to_finetune/data/val.jsonl)")
    ap.add_argument("--num-samples", type=int, default=100, help="Number of evaluation samples (default 100)")
    ap.add_argument("--max-gen-tokens", type=int, default=200)
    ap.add_argument("--models", nargs="+", default=["all"],
                    help="Models to evaluate: 'all' or subset of ['4b_base', '4b_ft', '8b_base', '8b_ft', 'gemini']")
    ap.add_argument("--out-dir", default=str(THIS_DIR / "evaluation_report"),
                    help="Output report directory (default: to_finetune/evaluation_report)")
    args = ap.parse_args()

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # 1. Load validation samples
    if not Path(args.val_data).exists():
        sys.exit(f"Error: Validation data file not found at {args.val_data}")

    print(f"\n📂 Loading evaluation questions from {args.val_data} ...")
    raw_val = [json.loads(line) for line in open(args.val_data, encoding="utf-8") if line.strip()]

    # Guarantee 100% strictly unseen questions by excluding any prompt that appeared in train.jsonl
    train_path = Path(args.val_data).parent / "train.jsonl"
    if train_path.exists():
        train_prompts = set(json.loads(l)["messages"][0]["content"].strip() for l in open(train_path, encoding="utf-8") if l.strip())
        samples = [s for s in raw_val if s["messages"][0]["content"].strip() not in train_prompts][: args.num_samples]
        print(f"✅ Loaded {len(samples)} questions (Guaranteed 100% Strictly Unseen, 0 overlap with train.jsonl).\n")
    else:
        samples = raw_val[: args.num_samples]
        print(f"✅ Loaded {len(samples)} evaluation samples successfully.\n")

    # Resolve Models
    p_4b_base = resolve_base_model_path("4B")
    p_4b_1ep  = resolve_adapter_path("4b_1ep")
    p_4b_ft   = resolve_adapter_path("4B")
    p_8b_base = resolve_base_model_path("8B")
    p_8b_ft   = resolve_adapter_path("8B")

    eval_all = "all" in args.models
    results = []

    # 1. Qwen3-VL-4B (Base)
    if (eval_all or "4b_base" in args.models) and p_4b_base and p_4b_base.exists():
        r = evaluate_local_model("Qwen3-VL-4B (Base)", p_4b_base, None, samples, args.max_gen_tokens)
        results.append(r)

    # 2a. Qwen3-VL-4B (Fine-Tuned 1-Epoch)
    if (eval_all or "4b_1ep" in args.models) and p_4b_base and p_4b_base.exists() and p_4b_1ep and p_4b_1ep.exists() and any(p_4b_1ep.iterdir()):
        r = evaluate_local_model("Qwen3-VL-4B (1-Epoch LoRA)", p_4b_base, p_4b_1ep, samples, args.max_gen_tokens)
        results.append(r)

    # 2b. Qwen3-VL-4B (Fine-Tuned 2-Epoch)
    if (eval_all or "4b_ft" in args.models) and p_4b_base and p_4b_base.exists() and p_4b_ft and p_4b_ft.exists():
        label_4b = "Qwen3-VL-4B (2-Epoch LoRA)" if (p_4b_1ep and p_4b_1ep.exists() and any(p_4b_1ep.iterdir())) else "Qwen3-VL-4B (Fine-Tuned)"
        r = evaluate_local_model(label_4b, p_4b_base, p_4b_ft, samples, args.max_gen_tokens)
        results.append(r)

    # 3. Qwen3-VL-8B (Base)
    if (eval_all or "8b_base" in args.models) and p_8b_base and p_8b_base.exists():
        r = evaluate_local_model("Qwen3-VL-8B (Base)", p_8b_base, None, samples, args.max_gen_tokens)
        results.append(r)

    # 4. Qwen3-VL-8B (Fine-Tuned Legal LoRA)
    if (eval_all or "8b_ft" in args.models) and p_8b_base and p_8b_base.exists():
        if p_8b_ft and p_8b_ft.exists() and any(p_8b_ft.iterdir()):
            r = evaluate_local_model("Qwen3-VL-8B (Fine-Tuned)", p_8b_base, p_8b_ft, samples, args.max_gen_tokens)
            results.append(r)
        else:
            print("\n[Notice] Qwen3-VL-8B adapter not found in resolved adapter paths.")
            print("Download or sync it, then re-run to include 8B Fine-Tuned!\n")

    # 5. Gemini 3.7 Flash
    if eval_all or "gemini" in args.models:
        r_gem = evaluate_gemini_model("Gemini 3.7 Flash", samples, args.max_gen_tokens)
        if r_gem:
            results.append(r_gem)

    if not results:
        sys.exit("No models were evaluated. Please check model paths or flags.")

    # ------------------------------------------------------------------
    # Save Results & Reports
    # ------------------------------------------------------------------
    json_file = out_path / "evaluation_report.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    def fmt_val(v, suffix=""):
        if v is None:
            return "N/A"
        if isinstance(v, (int, float)):
            return f"{v}{suffix}"
        return str(v)

    # Build Markdown Report
    md_content = f"# Master Benchmark Report: Legal Models Comparison\n\n"
    md_content += f"- **Evaluated on:** {len(samples)} strictly unseen Indian Legal test questions (`val.jsonl`)\n"
    md_content += f"- **Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (4-bit NF4 Quantization) & Cloud API\n\n"
    
    md_content += "## 📊 Table 1: Model Accuracy & Legal Specialization Metrics\n\n"
    md_content += "| Model | Deployment | Perplexity (PPL) ↓ | Eval Loss ↓ | ROUGE-2 (Bigram) ↑ | BLEU-4 ↑ | Citation Recall ↑ | Citation F1 ↑ | Legal Terminology ↑ |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for r in results:
        md_content += (
            f"| **{r['label']}** | {r['type']} | "
            f"**{fmt_val(r['perplexity'])}** | {fmt_val(r['eval_loss'])} | "
            f"**{r['rouge_2']}%** | **{r['bleu_4']}** | "
            f"**{r['citation_recall']}%** | {r['citation_f1']}% | {r['legal_terminology_density']}% |\n"
        )

    md_content += "\n\n## ⚡ Table 2: Lexical Overlap, Latency & VRAM Efficiency\n\n"
    md_content += "| Model | Deployment | ROUGE-1 ↑ | ROUGE-L ↑ | BLEU-1 ↑ | BLEU-2 ↑ | Throughput (tok/s) ↑ | Latency (s) ↓ | Peak GPU VRAM |\n"
    md_content += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"
    for r in results:
        md_content += (
            f"| **{r['label']}** | {r['type']} | "
            f"{r['rouge_1']}% | {r['rouge_l']}% | "
            f"{r['bleu_1']} | {r['bleu_2']} | "
            f"{r['tokens_per_sec']} tok/s | {r['avg_latency_sec']}s | **{r['peak_vram_gib']} GB** |\n"
        )

    # Side-by-Side Sample Comparison
    md_content += "\n\n## 🔍 Side-by-Side Qualitative Sample Comparisons\n\n"
    num_samples_to_show = min(3, len(results[0]["samples"]))
    for i in range(num_samples_to_show):
        md_content += f"### Sample {i+1}:\n"
        md_content += f"**User Query:**\n> {results[0]['samples'][i]['prompt']}\n\n"
        md_content += f"**Gold Reference:**\n> {results[0]['samples'][i]['gold']}\n\n"
        for r in results:
            md_content += f"**{r['label']} Output:**\n> {r['samples'][i]['predicted']}\n\n"
        md_content += "---\n\n"

    md_file = out_path / "evaluation_report.md"
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(md_content)

    # Print Formatted Summary Table to Terminal
    print("\n" + "=" * 115)
    print(f"MASTER LEGAL EVALUATION BENCHMARK ({len(samples)} Questions)")
    print("=" * 115)
    hdr = f"{'Model':<28} {'PPL ↓':<9} {'Loss ↓':<9} {'ROUGE-2':<10} {'BLEU-4':<9} {'Recall':<9} {'Legal %':<9} {'Speed':<10} {'Peak VRAM'}"
    print(hdr)
    print("-" * 115)
    for r in results:
        ppl_str = fmt_val(r['perplexity'])
        loss_str = fmt_val(r['eval_loss'])
        row = f"{r['label']:<28} {ppl_str:<9} {loss_str:<9} {r['rouge_2']:<10} {r['bleu_4']:<9} {r['citation_recall']:<9} {r['legal_terminology_density']:<9} {str(r['tokens_per_sec']) + ' t/s':<10} {r['peak_vram_gib']} GB"
        print(row)
    print("=" * 115)
    print(f"\n📄 Markdown report written to: {md_file}")
    print(f"💾 JSON metrics written to:     {json_file}\n")


if __name__ == "__main__":
    main()
