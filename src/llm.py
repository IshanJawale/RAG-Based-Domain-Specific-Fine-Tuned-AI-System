"""
llm.py

Local inference for Indian Legal RAG Assistant.
Local models run on llama.cpp's llama-server (tools/llama.cpp) from Q4_K_M GGUFs in models/gguf,
fully on GPU. If those files are missing, or LOCAL_BACKEND=transformers, they load in-process
with transformers instead (about 20x slower).
"""

import atexit
import gc
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path
import torch
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# Resolve models dir (case-insensitive)
MODELS_DIR = BASE_DIR / "models"
if not MODELS_DIR.exists() and (BASE_DIR / "Models").exists():
    MODELS_DIR = BASE_DIR / "Models"

AVAILABLE_MODELS = {
    "Qwen3-VL-8B-Legal (Fine-Tuned)": MODELS_DIR / "Qwen3-VL-8B-Instruct",
    "Qwen3-VL-4B-Legal (Fine-Tuned)": MODELS_DIR / "Qwen3-VL-4B-Instruct",
    "Qwen3-VL-4B-Instruct": MODELS_DIR / "Qwen3-VL-4B-Instruct",
    "Qwen3-8B": MODELS_DIR / "Qwen3-8B",
    "Qwen3-VL-8B-Instruct": MODELS_DIR / "Qwen3-VL-8B-Instruct",
}

GEMINI_MODELS = {
    "Gemini 3.7 Flash (Cloud API)": "gemini-3.7-flash",
    "Gemini 3.8 Flash (Cloud API)": "gemini-3.8-flash",
}

HPC_MODEL_NAME = "Qwen3-VL-Legal (HPC Cluster - RTX PRO 6000)"
HPC_SERVER_URL = os.getenv("HPC_SERVER_URL", "http://localhost:8000")

ADAPTER_PATHS = {
    "Qwen3-VL-4B-Legal (Fine-Tuned)": MODELS_DIR / "legal-lora-vl-4b-adapter",
    "Qwen3-VL-8B-Legal (Fine-Tuned)": MODELS_DIR / "legal-lora-vl-8b-adapter",
}


# Local models are served by llama-server from the GGUFs built by scripts/build_gguf.py.
LOCAL_BACKEND = os.getenv("LOCAL_BACKEND", "llama-server")  # "transformers" forces the slow in-process path
LLAMA_CTX = int(os.getenv("LLAMA_CTX", "8192"))  # keeps the KV cache small enough for an 8 GB GPU
LLAMA_SERVER_EXE = BASE_DIR / "tools" / "llama.cpp" / ("llama-server.exe" if os.name == "nt" else "llama-server")
LLAMA_SERVER_PORT = int(os.getenv("LLAMA_SERVER_PORT", "8089"))
# 127.0.0.1, not localhost: on Windows "localhost" tries IPv6 first and adds ~2 s to every request
LLAMA_SERVER_URL = f"http://127.0.0.1:{LLAMA_SERVER_PORT}"
GGUF_DIR = MODELS_DIR / "gguf"
GGUF_NAMES = {
    "Qwen3-VL-8B-Legal (Fine-Tuned)": "legal-qwen3-vl-8b",
    "Qwen3-VL-4B-Legal (Fine-Tuned)": "legal-qwen3-vl-4b",
    "Qwen3-VL-4B-Instruct": "qwen3-vl-4b-instruct",
    "Qwen3-VL-8B-Instruct": "qwen3-vl-8b-instruct",
}


def _gguf_path(model_name: str) -> str | None:
    """The GGUF llama-server should load for this model, or None to use the transformers path."""
    name = GGUF_NAMES.get(model_name)
    if not name or LOCAL_BACKEND == "transformers" or not LLAMA_SERVER_EXE.exists():
        return None
    gguf = GGUF_DIR / f"{name}.Q4_K_M.gguf"
    return str(gguf) if gguf.exists() else None


# One llama-server process at a time (an 8 GB GPU holds one model). Its pid and model are kept in a
# state file so a Streamlit restart reuses the running server instead of starting a second one.
_LLAMA_STATE = Path(tempfile.gettempdir()) / f"legal-rag-llama-server-{LLAMA_SERVER_PORT}.json"
_LLAMA_LOG = Path(tempfile.gettempdir()) / f"legal-rag-llama-server-{LLAMA_SERVER_PORT}.log"
_llama_lock = threading.Lock()
_spawned_pids = set()


def _llama_server_healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{LLAMA_SERVER_URL}/health", timeout=1) as resp:
            return resp.status == 200
    except Exception:
        return False


def _stop_llama_server():
    try:
        pid = json.loads(_LLAMA_STATE.read_text())["pid"]
    except Exception:
        return
    if os.name == "nt":
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True)
    else:
        try:
            os.kill(pid, 15)
        except OSError:
            pass
    _LLAMA_STATE.unlink(missing_ok=True)


def _ensure_llama_server(gguf: str):
    """Start llama-server on `gguf` unless it is already serving it."""
    with _llama_lock:
        try:
            state = json.loads(_LLAMA_STATE.read_text())
        except Exception:
            state = {}
        if state.get("model") == gguf and _llama_server_healthy():
            return
        _stop_llama_server()
        print(f"Starting llama-server with {Path(gguf).name} ...", flush=True)
        cmd = [str(LLAMA_SERVER_EXE), "-m", gguf, "-ngl", "99", "-c", str(LLAMA_CTX), "-fa", "on", "-np", "1",
               "--host", "127.0.0.1", "--port", str(LLAMA_SERVER_PORT)]
        env = dict(os.environ)
        if os.name != "nt":  # the Linux build loads its shared libraries from its own folder
            env["LD_LIBRARY_PATH"] = f"{LLAMA_SERVER_EXE.parent}:{env.get('LD_LIBRARY_PATH', '')}"
        proc = subprocess.Popen(cmd, stdout=open(_LLAMA_LOG, "w"), stderr=subprocess.STDOUT, env=env,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        _LLAMA_STATE.write_text(json.dumps({"pid": proc.pid, "model": gguf}))
        _spawned_pids.add(proc.pid)
        deadline = time.time() + 180
        while not _llama_server_healthy():
            if proc.poll() is not None or time.time() > deadline:
                _stop_llama_server()
                tail = _LLAMA_LOG.read_text(errors="ignore")[-1500:]
                raise RuntimeError(f"llama-server failed to start:\n{tail}")
            time.sleep(0.25)


def _stop_own_llama_server():
    try:
        if json.loads(_LLAMA_STATE.read_text())["pid"] in _spawned_pids:
            _stop_llama_server()
    except Exception:
        pass


atexit.register(_stop_own_llama_server)


def check_hpc_server(timeout: float = 0.8) -> dict | None:
    """Check if the HPC inference server is online and reachable over SSH tunnel."""
    import urllib.request
    import json
    try:
        req = urllib.request.Request(f"{HPC_SERVER_URL}/health")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status == 200:
                return json.loads(resp.read().decode("utf-8"))
    except Exception:
        pass
    return None


def _clean_v3_meta_artifacts(text: str) -> str:
    """
    Strip teacher-prompt meta-commentary artifacts that occasionally leak from
    Indian-legal-data-v3 at the end of an answer (e.g. 'This improved answer is significantly more detailed...').
    """
    import re
    if not text:
        return ""
    # Cut off at known v3 teacher meta-commentary markers
    meta_patterns = [
        r"(?:#+\s*)?This improved answer is significantly more detailed[\s\S]*$",
        r"(?:#+\s*)?The answer remains strictly grounded in the original context[\s\S]*$",
        r"(?:#+\s*)?The answer does not remove correct information[\s\S]*$",
        r"(?:#+\s*)?This response is strictly grounded in[\s\S]*$",
    ]
    cleaned = text
    for pat in meta_patterns:
        cleaned = re.sub(pat, "", cleaned, flags=re.IGNORECASE).strip()

    # Remove trailing empty markdown headings or stray hashes like ' ##' at the very end
    cleaned = re.sub(r"\s*#+\s*$", "", cleaned).strip()
    return cleaned


def _is_truncated_output(text: str) -> bool:
    """
    Detect if output genuinely stopped mid-sentence or mid-heading.
    """
    import re
    s = _clean_v3_meta_artifacts(text).strip()
    if not s:
        return False
    # If this is a complete JSON tool call block, do not continue
    if s.endswith("```") or (s.startswith("{") and s.endswith("}")):
        return False
    # If it already has a Conclusion section and ends with sentence punctuation, it is complete
    if re.search(r"(?:^|\n)#*\s*(?:\*\*)?Conclusion", s, re.IGNORECASE) and s.endswith((".", "!", "?", '"', "'", ")")):
        return False
    # Ended mid-word, mid-sentence, or mid-heading (does not end with sentence-final punctuation)
    if not s.endswith((".", "!", "?", '"', "'", ")", "]", "}", "`")):
        return True
    return False


def _call_hpc_server(prompt: str, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    """Call the fine-tuned Legal model running on Plaksha HPC GPU over SSH tunnel."""
    import urllib.request
    import json
    import re

    url = f"{HPC_SERVER_URL}/v1/chat/completions"
    headers = {"Content-Type": "application/json"}

    def _post_chat(messages: list[dict], max_tok: int) -> tuple[str, int]:
        payload = {
            "model": "qwen3-vl-legal",
            "messages": messages,
            "max_tokens": max_tok,
            "temperature": temperature,
            "repetition_penalty": 1.15,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=90) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            choices = res_data.get("choices", [])
            content = choices[0].get("message", {}).get("content", "").strip() if choices else ""
            comp_toks = res_data.get("usage", {}).get("completion_tokens", 0)
            return content, comp_toks

    try:
        raw_text, comp_toks = _post_chat([{"role": "user", "content": prompt}], max_tokens)
        if not raw_text:
            return "[HPC Server returned empty response]"

        full_text = _clean_v3_meta_artifacts(raw_text)

        # Only run a single continuation if the output genuinely cut off mid-sentence
        if _is_truncated_output(full_text):
            m = re.search(r"^([\s\S]*[\.\!\?])(\s+[\s\S]+)$", full_text)
            if m and len(m.group(2).strip()) < 300:
                prefix = m.group(1).rstrip()
                unfinished_tail = m.group(2).strip()
            else:
                prefix = full_text
                unfinished_tail = ""

            cont_instruction = (
                "Your previous output stopped mid-sentence. "
                + (
                    f"Continue directly by completing this unfinished section:\n\"{unfinished_tail}\"\n"
                    if unfinished_tail
                    else "Continue directly from the exact point where you stopped "
                )
                + "and finish the remaining analysis and conclusion concisely. "
                "Do NOT repeat earlier sections and stop immediately after the Conclusion."
            )

            cont_messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": prefix},
                {"role": "user", "content": cont_instruction},
            ]
            next_chunk, _ = _post_chat(cont_messages, 450)
            next_chunk = _clean_v3_meta_artifacts(next_chunk)
            if next_chunk:
                if unfinished_tail:
                    clean_tail_start = re.sub(r"[^a-zA-Z0-9]+", "", unfinished_tail[:25]).lower()
                    clean_next_start = re.sub(r"[^a-zA-Z0-9]+", "", next_chunk[:25]).lower()
                    if clean_tail_start and clean_next_start.startswith(clean_tail_start[:8]):
                        full_text = f"{prefix}\n\n{next_chunk}"
                    else:
                        sep = "" if full_text.endswith(("*", "#", "-", " ")) else " "
                        full_text = f"{full_text}{sep}{next_chunk}"
                else:
                    full_text = f"{prefix}\n\n{next_chunk}"

        full_text = _clean_v3_meta_artifacts(full_text).strip()
        if not full_text.endswith((".", "!", "?", '"', "'", ")", "]", "}", "`")):
            m_clean = re.search(r"^([\s\S]*[\.\!\?])(\s+[^\.\!\?]*)$", full_text)
            if m_clean:
                full_text = m_clean.group(1).strip()

        return full_text
    except Exception as e:
        raise RuntimeError(
            f"Failed to query HPC inference server at {url}: {e}\n"
            f"Ensure your SSH tunnel is active on your laptop:\n"
            f"ssh -N -L 8000:cn02:8000 ishan.jawale.pg26@hpc.plaksha.edu.in"
        )


# Default to HPC server if online, else the 4B legal model (GGUF or adapter), else base
DEFAULT_MODEL = os.getenv("ACTIVE_MODEL")
if not DEFAULT_MODEL:
    if check_hpc_server() is not None:
        DEFAULT_MODEL = HPC_MODEL_NAME
    elif _gguf_path("Qwen3-VL-4B-Legal (Fine-Tuned)") or (MODELS_DIR / "legal-lora-vl-4b-adapter").exists():
        DEFAULT_MODEL = "Qwen3-VL-4B-Legal (Fine-Tuned)"
    else:
        DEFAULT_MODEL = "Qwen3-VL-4B-Instruct"

_active_model_name = None
_model = None
_processor = None
_lock = threading.Lock()


def get_available_models() -> list[str]:
    """Return list of valid model names available on disk, cloud API, and HPC server."""
    models = []
    
    # Check if HPC cluster server is reachable
    hpc_status = check_hpc_server()
    if hpc_status is not None:
        models.append(HPC_MODEL_NAME)

    # Local models
    for name, p in AVAILABLE_MODELS.items():
        if _gguf_path(name):
            models.append(name)
        elif name in ADAPTER_PATHS:
            if p.exists() and ADAPTER_PATHS[name].exists():
                models.append(name)
        elif p.exists():
            models.append(name)

    # Cloud Gemini models
    for gname in GEMINI_MODELS.keys():
        models.append(gname)

    # If HPC server wasn't online at check time, still add as selectable option
    if hpc_status is None:
        models.append(HPC_MODEL_NAME)

    return models


def _call_gemini_api(
    prompt: str,
    model_id: str = "gemini-3.8-flash",
    max_tokens: int = 1024,
    temperature: float = 0.0,
    thinking_budget: int | None = None,
) -> str:
    """Call Google Gemini REST API directly using standard library urllib."""
    import urllib.request
    import json

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) is not set! Please add GEMINI_API_KEY=your_key in your .env file."
        )

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "maxOutputTokens": max_tokens,
            "temperature": temperature,
        },
    }
    if thinking_budget is not None:
        payload["generationConfig"]["thinkingConfig"] = {"thinkingBudget": thinking_budget}

    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=45) as response:
            res_data = json.loads(response.read().decode("utf-8"))
            candidates = res_data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                if parts:
                    return parts[0].get("text", "").strip()
            return "[Gemini API returned empty response]"
    except urllib.error.HTTPError as e:
        err_msg = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Gemini API HTTP Error {e.code}: {err_msg}")
    except Exception as e:
        raise RuntimeError(f"Gemini API request failed: {e}")


def set_active_model(model_name: str):
    """Switch active model and unload previous weights to free memory."""
    global _active_model_name, _model, _processor
    gguf = _gguf_path(model_name)
    if not gguf:
        _stop_llama_server()  # free its VRAM for whatever runs next
    if model_name == HPC_MODEL_NAME or model_name in GEMINI_MODELS or gguf:
        # If switching to HPC or Cloud API model, unload any heavy GPU/CPU local model weights to free memory!
        with _lock:
            if _model is not None:
                print(f"Unloading local model {_active_model_name} to free memory for remote inference ({model_name})...")
                del _model
                del _processor
                _model = None
                _processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                gc.collect()
            _active_model_name = model_name
        return

    if model_name not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model: {model_name}. Choose from {get_available_models()}")

    if _active_model_name == model_name and _model is not None:
        return

    with _lock:
        if _model is not None:
            print(f"Unloading {_active_model_name} from memory...")
            del _model
            del _processor
            _model = None
            _processor = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        _active_model_name = model_name
        _load_model(model_name)


def _load_model(model_name: str):
    global _model, _processor, _active_model_name
    model_path = AVAILABLE_MODELS[model_name]
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    has_cuda = torch.cuda.is_available()
    use_gpu_4bit = has_cuda

    device_desc = "RTX 5060 GPU (4-bit NF4)" if use_gpu_4bit else "CPU (bfloat16)"
    print(f"Loading {model_name} on {device_desc} from {model_path} ...", flush=True)

    bnb_config = None
    if use_gpu_4bit:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if "VL" in model_name:
        from transformers import AutoModelForImageTextToText
        try:
            from transformers import AutoProcessor
            _processor = AutoProcessor.from_pretrained(str(model_path))
        except Exception:
            from transformers import AutoTokenizer
            _processor = AutoTokenizer.from_pretrained(str(model_path))
        load_kwargs = {
            "low_cpu_mem_usage": True,
        }
        if use_gpu_4bit:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "cpu"

        try:
            _model = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kwargs)
        except TypeError:
            if not use_gpu_4bit:
                load_kwargs.pop("dtype", None)
                load_kwargs["torch_dtype"] = torch.bfloat16
            _model = AutoModelForImageTextToText.from_pretrained(str(model_path), **load_kwargs)
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        _processor = AutoTokenizer.from_pretrained(str(model_path))
        load_kwargs = {
            "low_cpu_mem_usage": True,
        }
        if has_cuda:
            load_kwargs["quantization_config"] = bnb_config
            load_kwargs["device_map"] = {"": 0}
        else:
            load_kwargs["dtype"] = torch.bfloat16
            load_kwargs["device_map"] = "cpu"

        try:
            _model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)
        except TypeError:
            if not has_cuda:
                load_kwargs.pop("dtype", None)
                load_kwargs["torch_dtype"] = torch.bfloat16
            _model = AutoModelForCausalLM.from_pretrained(str(model_path), **load_kwargs)

    # Attach LoRA fine-tuned legal adapter if requested
    if model_name in ADAPTER_PATHS and ADAPTER_PATHS[model_name].exists():
        from peft import PeftModel
        adapter_path = ADAPTER_PATHS[model_name]
        print(f"Attaching Fine-Tuned Indian Legal LoRA adapter from {adapter_path} ...", flush=True)
        _model = PeftModel.from_pretrained(_model, str(adapter_path))

    _model.eval()
    _active_model_name = model_name
    print(f"Successfully loaded {model_name} on {device_desc}!", flush=True)


def _ensure_loaded(model_name: str = None):
    global _active_model_name, _model
    target = model_name or _active_model_name or DEFAULT_MODEL
    if target == HPC_MODEL_NAME or target in GEMINI_MODELS or _gguf_path(target):
        if _active_model_name != target:
            set_active_model(target)
        return None, None

    if _model is None or _active_model_name != target:
        set_active_model(target)
    return _model, _processor


def chat(
    prompt: str,
    model_name: str = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
) -> str:
    """Send a prompt to the active model and return generated text."""
    target_model = model_name or _active_model_name or DEFAULT_MODEL
    if target_model == HPC_MODEL_NAME:
        return _call_hpc_server(prompt, max_tokens=max_new_tokens, temperature=temperature)

    if target_model in GEMINI_MODELS:
        model_id = GEMINI_MODELS[target_model]
        return _call_gemini_api(prompt, model_id=model_id, max_tokens=max_new_tokens, temperature=temperature)

    if _gguf_path(target_model):
        return "".join(_stream_llama_server(prompt, target_model, max_new_tokens, temperature)).strip()

    model, processor, inputs, gen_kwargs = _local_generation_inputs(target_model, prompt, max_new_tokens, temperature)
    tokenizer = getattr(processor, "tokenizer", processor)
    stop = threading.Event()
    with torch.no_grad():
        output = model.generate(**inputs, **gen_kwargs,
                                stopping_criteria=_meta_stopper(tokenizer, inputs["input_ids"].shape[1], stop))

    new_tokens = output[0][inputs["input_ids"].shape[1]:]
    return _strip_meta(processor.decode(new_tokens, skip_special_tokens=True)).strip()


# Some training answers end with notes about how the answer was rewritten ("The length has been
# increased by at least 2-4x..."). The fine-tuned adapters learned to append them and then loop on
# them, so local generation stops as soon as one starts and the note is cut from the output.
_META_COMMENTARY = re.compile(
    r"(?:^|(?<=[.!?\n]))[ \t]*(?:"
    r"(?:This|The) (?:improved|original|revised|refined|enhanced|rewritten) (?:answer|response|version)\b"
    r"|The (?:answer|response) (?:is also|is structured|is now|is at least|is grounded|is concise|is formal"
    r"|is significantly|is substantially|is consistent|has been|avoids|ensures|also|remains|ends|maintains)\b"
    r"|This structure (?:ensures|allows)\b|This structured approach\b"
    r"|The length of the (?:answer|response)\b|The structure is maintained\b)"
    r"|at least 2[–-]4× the original length",
    re.MULTILINE,
)
_META_MIN_POS = 200  # never cut the opening of an answer


def _find_meta(text: str):
    return _META_COMMENTARY.search(text, _META_MIN_POS) if len(text) > _META_MIN_POS else None


def _strip_meta(text: str) -> str:
    m = _find_meta(text)
    return text[: m.start()].rstrip() if m else text


def _meta_stopper(tokenizer, prompt_len: int, stop: threading.Event):
    from transformers import StoppingCriteria, StoppingCriteriaList

    class _StopOnMeta(StoppingCriteria):
        def __call__(self, input_ids, scores, **kwargs):
            if stop.is_set():
                return True
            new = input_ids[0][prompt_len:]
            if len(new) > 60:  # roughly past the protected opening
                tail = tokenizer.decode(new[-40:], skip_special_tokens=True)
                if _META_COMMENTARY.search(tail):
                    stop.set()
            return stop.is_set()

    return StoppingCriteriaList([_StopOnMeta()])


def chat_stream(
    prompt: str,
    model_name: str = None,
    max_new_tokens: int = 1024,
    temperature: float = 0.0,
):
    """Yield the answer incrementally. Local models stream tokens, Gemini streams over SSE,
    and the HPC server (which has no streaming endpoint) yields its full answer at once."""
    target_model = model_name or _active_model_name or DEFAULT_MODEL
    if target_model == HPC_MODEL_NAME:
        yield _call_hpc_server(prompt, max_tokens=max_new_tokens, temperature=temperature)
        return
    if target_model in GEMINI_MODELS:
        yield from _stream_gemini_api(prompt, model_id=GEMINI_MODELS[target_model], max_tokens=max_new_tokens, temperature=temperature)
        return

    if _gguf_path(target_model):
        yield from _stream_llama_server(prompt, target_model, max_new_tokens, temperature)
        return

    from transformers import TextIteratorStreamer

    model, processor, inputs, gen_kwargs = _local_generation_inputs(target_model, prompt, max_new_tokens, temperature)
    tokenizer = getattr(processor, "tokenizer", processor)
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
    errors = []
    stop = threading.Event()
    stopper = _meta_stopper(tokenizer, inputs["input_ids"].shape[1], stop)

    def _run():
        try:
            with torch.no_grad():
                model.generate(**inputs, **gen_kwargs, streamer=streamer, stopping_criteria=stopper)
        except Exception as e:  # surface generation errors to the consumer instead of hanging
            errors.append(e)
            streamer.end()

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    # Hold back a short tail so a meta-commentary note can be cut before any of it is shown
    text, sent, hold = "", 0, 160
    for piece in streamer:
        text += piece
        if stop.is_set() or _find_meta(text):
            stop.set()  # drain what is left; the note is cut below
            continue
        if len(text) - hold > sent:
            yield text[sent:len(text) - hold]
            sent = len(text) - hold
    thread.join()
    clean = _strip_meta(text)
    if len(clean) > sent:
        yield clean[sent:]
    if errors:
        raise errors[0]


def _stream_llama_server(prompt: str, model_name: str, max_tokens: int, temperature: float):
    """Stream from llama-server, cutting the training-data meta-commentary like the transformers path does."""
    if _active_model_name != model_name:
        set_active_model(model_name)  # unloads any transformers model
    _ensure_llama_server(_gguf_path(model_name))
    pieces = _llama_server_pieces(prompt, max_tokens, temperature if temperature > 0 else 0.2)

    # Hold back a short tail so a meta-commentary note can be cut before any of it is shown
    text, sent, hold = "", 0, 160
    try:
        for piece in pieces:
            text += piece
            if _find_meta(text):
                break
            if len(text) - hold > sent:
                yield text[sent:len(text) - hold]
                sent = len(text) - hold
    finally:
        pieces.close()  # closes the HTTP connection, which stops generation server-side
    clean = _strip_meta(text)
    if len(clean) > sent:
        yield clean[sent:]


def _llama_server_pieces(prompt: str, max_tokens: int, temperature: float):
    req = urllib.request.Request(f"{LLAMA_SERVER_URL}/v1/chat/completions", data=json.dumps({
        "messages": [{"role": "user", "content": prompt}],
        "stream": True,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9,
        "repeat_penalty": 1.1,
    }).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                for choice in json.loads(data).get("choices", []):
                    yield choice.get("delta", {}).get("content") or ""
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"llama-server HTTP Error {e.code}: {e.read().decode('utf-8', errors='ignore')}")


def _stream_gemini_api(prompt: str, model_id: str, max_tokens: int = 1024, temperature: float = 0.0):
    """Stream text chunks from Gemini's streamGenerateContent endpoint (server-sent events)."""
    import json
    import urllib.request

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY (or GOOGLE_API_KEY) is not set! Please add GEMINI_API_KEY=your_key in your .env file.")
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:streamGenerateContent?alt=sse&key={api_key}"
    payload = {"contents": [{"parts": [{"text": prompt}]}], "generationConfig": {"maxOutputTokens": max_tokens, "temperature": temperature}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=90) as resp:
            for raw in resp:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = json.loads(line[5:].strip())
                for cand in data.get("candidates", []):
                    for part in cand.get("content", {}).get("parts", []):
                        if part.get("text"):
                            yield part["text"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Gemini API HTTP Error {e.code}: {e.read().decode('utf-8', errors='ignore')}")


def _local_generation_inputs(target_model: str, prompt: str, max_new_tokens: int, temperature: float):
    """Load the local model if needed and build the chat-templated inputs and generation kwargs."""
    model, processor = _ensure_loaded(target_model)

    messages = [{"role": "user", "content": prompt}]

    # Format using chat template, disable chain-of-thought to speed up generation.
    # The tokenizer's template gives the same text for text-only prompts; calling it on a VL processor
    # forwards enable_thinking to the processor and triggers the "processor_kwargs" warning.
    templater = getattr(processor, "tokenizer", processor)
    try:
        text = templater.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except Exception:
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    # Route input tensors to the model's primary device (e.g. cuda:0)
    device = getattr(model, "device", None)
    if device is None or str(device) == "meta":
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = "cuda" if torch.cuda.is_available() else "cpu"

    if "VL" in _active_model_name:
        # Text-only prompt: go through the tokenizer to avoid the processor kwargs warning
        inputs = getattr(processor, "tokenizer", processor)([text], return_tensors="pt").to(device)
    else:
        inputs = processor([text], return_tensors="pt").to(device)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "repetition_penalty": 1.1,
    }
    if temperature > 0:
        gen_kwargs.update(do_sample=True, temperature=temperature, top_p=0.9)
    else:
        # Subtle sampling (0.2) produces significantly cleaner formatting and lists than pure deterministic argmax
        gen_kwargs.update(do_sample=True, temperature=0.2, top_p=0.9)

    return model, processor, inputs, gen_kwargs


# Alias for compatibility across codebase
generate = chat
