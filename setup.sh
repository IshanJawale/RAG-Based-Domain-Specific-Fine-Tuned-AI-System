#!/usr/bin/env bash
# One-shot setup for Indian Legal AI: Python environment, Hugging Face downloads, llama-server, checks.
# Safe to re-run: existing files are skipped, .env is never overwritten.
#
#   bash setup.sh                         # everything (~32 GB: 15 GB index, 15 GB models, 0.7 GB llama.cpp)
#   bash setup.sh --models "legal-qwen3-vl-4b"   # only the GGUFs you list (default: all four)
#   bash setup.sh --skip-index            # skip the 15 GB RAG index (the app will not answer without it)
#   bash setup.sh --with-hf-weights       # also the LoRA adapters + base Qwen weights (transformers
#                                         # fallback and scripts/build_gguf.py; adds ~27 GB)
#
# Windows: run from Git Bash. Linux: needs python3.12 and curl.
set -euo pipefail
cd "$(dirname "$0")"

MODELS="legal-qwen3-vl-4b legal-qwen3-vl-8b qwen3-vl-4b-instruct qwen3-vl-8b-instruct"
SKIP_INDEX=0
WITH_HF_WEIGHTS=0
while [ $# -gt 0 ]; do
  case "$1" in
    --models) MODELS="$2"; shift 2 ;;
    --skip-index) SKIP_INDEX=1; shift ;;
    --with-hf-weights) WITH_HF_WEIGHTS=1; shift ;;
    -h|--help) sed -n '2,11p' "$0"; exit 0 ;;
    *) echo "Unknown option: $1 (see --help)"; exit 1 ;;
  esac
done

GGUF_REPO=ishan-jawale/indian-legal-qwen3-vl-gguf
TOOLS_REPO=ishan-jawale/indian-legal-llama-cpp-tools
LLAMA_BUILD=b11836
step() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\033[31mError: %s\033[0m\n' "$*" >&2; exit 1; }

case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) OS=windows; VENV_BIN=.venv/Scripts; EXE=.exe ;;
  Linux*) OS=linux; VENV_BIN=.venv/bin; EXE= ;;
  *) die "unsupported OS $(uname -s): use Windows (Git Bash) or Linux" ;;
esac

# ---------------------------------------------------------------- Python environment
step "Python environment (.venv)"
if [ ! -x "$VENV_BIN/python$EXE" ]; then
  PY=""
  for c in python3.12 python3 python "py -3.12"; do
    if $c -c 'import sys; sys.exit(sys.version_info[:2] != (3, 12))' 2>/dev/null; then PY="$c"; break; fi
  done
  [ -n "$PY" ] || die "Python 3.12 not found (https://www.python.org/downloads/)"
  $PY -m venv .venv
fi
PY="$VENV_BIN/python$EXE"
HF="$VENV_BIN/hf$EXE"
"$PY" -m pip install --upgrade pip -q
"$PY" -m pip install -r requirements.txt
"$PY" -c "import torch; print('PyTorch', torch.__version__, '| CUDA:', torch.cuda.is_available())"

# ---------------------------------------------------------------- .env and Hugging Face login
step "Keys (.env) and Hugging Face login"
if [ ! -f .env ]; then
  cat > .env <<'ENV'
TAVILY_API_KEY=""   # web search (optional but recommended)
GEMINI_API_KEY=""   # optional: Gemini backend and faster routing
HF_TOKEN=""         # read access to the private ishan-jawale/* repos
ENV
  echo "Created .env: fill in the keys you have."
fi
# Use HF_TOKEN from .env when it is set
TOKEN=$(sed -n 's/^HF_TOKEN=["'\'']\{0,1\}\([^"'\'' #]*\).*/\1/p' .env | head -1)
[ -n "$TOKEN" ] && export HF_TOKEN="$TOKEN"
if ! "$HF" auth whoami >/dev/null 2>&1; then
  echo "Not logged in to Hugging Face. Paste a token with read access (https://huggingface.co/settings/tokens):"
  "$HF" auth login
fi
"$HF" auth whoami | head -1

# ---------------------------------------------------------------- llama-server
step "llama.cpp $LLAMA_BUILD (llama-server) -> tools/llama.cpp"
if [ "$OS" = windows ]; then
  # one flag per pattern, no wildcards: on Windows patterns can get expanded against the current folder
  "$HF" download "$TOOLS_REPO" --local-dir tools/llama.cpp --exclude README.md --exclude .gitattributes
elif [ ! -x tools/llama.cpp/llama-server ]; then
  # The HF copy is the Windows build; on Linux take the official release and verify its checksum.
  base="https://github.com/ggml-org/llama.cpp/releases/download/$LLAMA_BUILD"
  zip="llama-$LLAMA_BUILD-bin-ubuntu-cuda-13.4-x64.tar.gz"
  sum=$(curl -fsSL "https://api.github.com/repos/ggml-org/llama.cpp/releases/tags/$LLAMA_BUILD" \
        | "$PY" -c "import sys, json; print(next(a['digest'] for a in json.load(sys.stdin)['assets'] if a['name'] == '$zip').split(':')[1])")
  tmp=$(mktemp -d)
  curl -fL -o "$tmp/$zip" "$base/$zip"
  echo "$sum  $tmp/$zip" | sha256sum -c - || die "checksum mismatch for $zip"
  mkdir -p tools/llama.cpp
  tar -xzf "$tmp/$zip" -C tools/llama.cpp --strip-components=1
  rm -rf "$tmp"
fi
"tools/llama.cpp/llama-server$EXE" --list-devices 2>&1 | tail -n +1

# ---------------------------------------------------------------- Models and data
step "Fine-tuned GGUF models -> Models/gguf ($MODELS)"
includes=()
for m in $MODELS; do includes+=(--include "$m.Q4_K_M.gguf"); done
"$HF" download "$GGUF_REPO" "${includes[@]}" --local-dir Models/gguf

if [ "$SKIP_INDEX" = 0 ]; then
  step "RAG index -> data/ (~15 GB)"
  "$HF" download ishan-jawale/indian-legal-rag-index --repo-type dataset --local-dir data
fi

step "Retrieval embedder and reranker (cached for offline loading)"
"$HF" download BAAI/bge-small-en-v1.5 >/dev/null
"$HF" download cross-encoder/ms-marco-MiniLM-L-6-v2 >/dev/null
echo "ok"

if [ "$WITH_HF_WEIGHTS" = 1 ]; then
  step "LoRA adapters and base Qwen3-VL weights (transformers fallback / build_gguf.py)"
  "$HF" download ishan-jawale/indian-legal-lora-adapters --local-dir Models
  "$HF" download Qwen/Qwen3-VL-4B-Instruct --local-dir Models/Qwen3-VL-4B-Instruct
  "$HF" download Qwen/Qwen3-VL-8B-Instruct --local-dir Models/Qwen3-VL-8B-Instruct
fi

# ---------------------------------------------------------------- Checks
step "Checks"
if [ -d data/lancedb ]; then
  "$PY" -c "import lancedb; db = lancedb.connect('data/lancedb'); print('Index:', {t: db.open_table(t).count_rows() for t in db.table_names()})"
fi
first=$(echo "$MODELS" | awk '{print $1}')
case "$first" in
  legal-qwen3-vl-4b) name="Qwen3-VL-4B-Legal (Fine-Tuned)" ;;
  legal-qwen3-vl-8b) name="Qwen3-VL-8B-Legal (Fine-Tuned)" ;;
  qwen3-vl-4b-instruct) name="Qwen3-VL-4B-Instruct" ;;
  qwen3-vl-8b-instruct) name="Qwen3-VL-8B-Instruct" ;;
  *) name="" ;;
esac
if [ -n "$name" ]; then
  "$PY" - "$name" <<'PY'
import sys, time
from src import llm
name = sys.argv[1]
assert llm._gguf_path(name), f"no GGUF/llama-server for {name}"
t = time.time()
answer = llm.chat("In one sentence, what is an FIR?", model_name=name, max_new_tokens=60)
print(f"{name} via llama-server ({time.time() - t:.1f}s incl. load): {answer}")
PY
fi

step "Done"
echo "Start the app:   $VENV_BIN/streamlit$EXE run app/app.py"
echo "Or the CLI:      $PY main.py \"What is anticipatory bail?\""
