"""
common.py - shared helpers for the fine-tuning pipeline.

Imports the *exact* prompt builders used by the live app (src/generation.py,
src/routing.py) so the training prompts can never drift from inference prompts.
"""

import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from dotenv import load_dotenv
from tqdm import tqdm

FT_DIR = Path(__file__).resolve().parent
ROOT = FT_DIR.parent
DATA_DIR = FT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Make `src.*` importable and make relative paths in src/retrieval.py
# (data/lancedb) resolve against the project root.
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
load_dotenv(ROOT / ".env")

INSUFFICIENT_SENTENCE = "The available legal evidence is insufficient to answer this question."

TABLES = ["acts", "supreme_court", "high_court"]


# ----------------------------------------------------------------------
# Prompt builders (single source of truth = src/)
# ----------------------------------------------------------------------
def build_answer_prompt(question: str, evidence: list[dict]) -> str:
    from src.generation import build_prompt
    return build_prompt(question, evidence)


def build_router_prompt(question: str) -> str:
    from src.routing import ROUTER_SYSTEM_PROMPT
    return (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        f"USER QUESTION:\n{question}\n\n"
        f"Respond with JSON only."
    )


# ----------------------------------------------------------------------
# JSONL helpers (append-only => every stage is resumable)
# ----------------------------------------------------------------------
_write_lock = threading.Lock()


def read_jsonl(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # tolerate a half-written last line after a crash
    return out


def append_jsonl(path: Path, row: dict):
    with _write_lock:
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()


def write_jsonl(path: Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def done_ids(path: Path, key: str) -> set:
    return {r[key] for r in read_jsonl(path) if key in r}


# ----------------------------------------------------------------------
# Teacher LLM client - any OpenAI-compatible /chat/completions endpoint
#
#   Gemini : TEACHER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
#            TEACHER_MODEL=gemini-2.5-flash       TEACHER_API_KEY=<gemini key>
#   OpenAI : TEACHER_BASE_URL=https://api.openai.com/v1
#   vLLM   : TEACHER_BASE_URL=http://localhost:8000/v1   (e.g. on your H200)
#            TEACHER_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8 (or any model you serve)
#            TEACHER_API_KEY=EMPTY   TEACHER_DISABLE_THINKING=1
# ----------------------------------------------------------------------
class Teacher:
    def __init__(self):
        provider = os.getenv("TEACHER_PROVIDER", "").lower()
        if provider == "ollama" or "11434" in os.getenv("TEACHER_BASE_URL", ""):
            default_base = "http://localhost:11434/v1"
            default_key = "ollama"
            default_model = "gpt-oss:120b-cloud"
        else:
            default_base = "https://generativelanguage.googleapis.com/v1beta/openai"
            default_key = ""
            default_model = "gemini-3.8-flash"

        self.base_url = os.getenv("TEACHER_BASE_URL", default_base).rstrip("/")
        self.api_key = os.getenv("TEACHER_API_KEY") or os.getenv("GEMINI_API_KEY") or default_key
        self.model = os.getenv("TEACHER_MODEL", default_model)
        self.disable_thinking = os.getenv("TEACHER_DISABLE_THINKING", "0") == "1"
        if not self.api_key and "localhost" not in self.base_url and "127.0.0.1" not in self.base_url:
            raise RuntimeError(
                "TEACHER_API_KEY is not set (put it in .env). Use TEACHER_API_KEY=EMPTY or TEACHER_PROVIDER=ollama."
            )

    def chat(self, system: str, user: str, temperature: float = 0.7,
             max_tokens: int = 1500, retries: int = 6) -> str:
        payload = {
            "model": self.model,
            "messages": ([{"role": "system", "content": system}] if system else [])
                        + [{"role": "user", "content": user}],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        headers = {"Authorization": f"Bearer {self.api_key or 'EMPTY'}", "Content-Type": "application/json"}

        last_err = None
        for attempt in range(retries):
            try:
                r = requests.post(f"{self.base_url}/chat/completions",
                                  headers=headers, json=payload, timeout=60)
                if r.status_code == 429:
                    wait = min(60, 4 * (2 ** attempt) + random.uniform(1, 3))
                    time.sleep(wait)
                    continue
                if r.status_code >= 500:
                    time.sleep(min(30, 2 ** attempt + random.random()))
                    continue
                if r.status_code != 200:
                    # Non-retryable client error (e.g. 404 model not found, 401 bad key)
                    raise ValueError(f"Teacher API Error (HTTP {r.status_code}): {r.text[:300]}")
                msg = r.json()["choices"][0]["message"]
                text = msg.get("content") or msg.get("reasoning") or ""
                return strip_thinking(text)
            except ValueError:
                raise
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(30, 2 ** attempt + random.random()))
        raise RuntimeError(f"Teacher call failed after {retries} tries: {last_err}")


def strip_thinking(text: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    return text.strip()


def parse_json_loose(text: str):
    """Parse a JSON object from an LLM reply that may be fenced or padded."""
    text = text.strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        try:
            return json.loads(m.group(0))
        except Exception:
            cleaned = re.sub(r",\s*([}\]])", r"\1", m.group(0))
            return json.loads(cleaned)
    raise ValueError(f"Could not extract JSON from: {text[:120]}")


# ----------------------------------------------------------------------
# Parallel runner with progress bar
# ----------------------------------------------------------------------
def run_parallel(fn, items, workers: int, desc: str):
    """Run fn(item) over items in a thread pool; exceptions are logged, not raised."""
    ok = 0
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(fn, it): it for it in items}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc):
            try:
                fut.result()
                ok += 1
            except Exception as e:  # noqa: BLE001
                tqdm.write(f"[warn] {desc}: {type(e).__name__}: {str(e)[:160]}")
    return ok


def is_clean_text(text: str, min_words: int, alpha_ratio: float = 0.80) -> bool:
    """Reject OCR garbage / mojibake chunks as seeds."""
    if "\ufffd" in text:
        return False
    if len(text.split()) < min_words:
        return False
    letters = sum(ch.isalpha() or ch.isspace() for ch in text)
    return letters / max(1, len(text)) >= alpha_ratio
