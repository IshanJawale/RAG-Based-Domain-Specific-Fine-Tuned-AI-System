# SEVA AI: Trusted AI Help for Every Citizen

**One private, offline platform for everyday questions on law, medicine, finance, engineering and education. Each domain has its own fine-tuned model and verified sources. Legal is live; the other domains are coming soon.**

**[Watch the demo on YouTube](https://youtu.be/BE-PxThAp5o)**

[![Demo video](https://img.youtube.com/vi/BE-PxThAp5o/maxresdefault.jpg)](https://youtu.be/BE-PxThAp5o)

## Results at a glance

- **Perplexity down 44%.** 4B: 3.84 -> 2.14. 8B: 3.63 -> 2.06.
- **BLEU-4 up 67%, ROUGE-2 up 59%** on the 4B model, so answers use the exact wording of Indian statutes and judgments (*ad interim injunction*, *prima facie case*, *Section 138 NI Act*).
- **Beats Gemini 3.7 Flash on text overlap.** The fine-tuned 4B scores ROUGE-2 14.85% vs 10.91% and BLEU-4 4.57 vs 2.87, and slightly edges it on citation F1 (79.94% vs 79.18%).
- **Runs on a laptop.** As 4-bit GGUFs the fine-tuned models generate about 100 tokens/s (4B) and 63 tokens/s (8B) on an 8 GB RTX 5060 Laptop GPU, fully offline.
- **Cheap to train.** RS-LoRA updates about 0.7% of the weights (33.5M of 4.4B; 58.7M of 8.3B). One epoch on the 4B takes 3.5 hours on a single NVIDIA RTX PRO 6000 48 GB slice.

### Benchmarks

100 held-out Indian legal questions with no prompt overlap with the 47.5k training pairs. Evaluated in bf16 on the Plaksha HPC (NVIDIA RTX PRO 6000, 48 GB slice).

| Model | Perplexity ↓ | Eval loss ↓ | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU-4 ↑ | Citation recall ↑ | Citation F1 ↑ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Qwen3-VL-4B (base) | 3.84 | 1.3442 | 27.34% | 9.33% | 16.86% | 2.74 | 81.11% | 75.97% |
| Qwen3-VL-4B, LoRA 1 epoch | 2.16 | 0.7702 | **32.08%** | **14.85%** | **21.35%** | **4.57** | 78.34% | **79.94%** |
| Qwen3-VL-4B, LoRA 2 epochs | 2.14 | 0.7592 | 31.89% | 14.54% | 20.79% | 4.34 | 78.52% | 76.60% |
| Qwen3-VL-4B, LoRA 4 epochs (early-stopped at 1.28) | 2.23 | 0.8011 | 31.89% | 14.57% | 20.74% | 4.42 | 78.71% | 76.95% |
| Qwen3-VL-8B (base) | 3.63 | 1.2884 | 27.40% | 9.52% | 16.85% | 2.90 | 80.17% | 76.19% |
| Qwen3-VL-8B, LoRA 2 epochs | **2.06** | **0.7214** | 31.33% | 13.98% | 20.54% | 4.11 | 74.51% | 74.60% |
| Gemini 3.7 Flash (cloud API) | n/a | n/a | 28.18% | 10.91% | 17.39% | 2.87 | **85.06%** | 79.18% |

Citation recall dips slightly after fine-tuning and Gemini leads on it. In the full app, citations come from the retrieved Acts and judgments rather than from the model's memory, which covers this gap.

### Why this approach works

- **Fine-tuning plus retrieval.** Fine-tuning teaches the model the language and reasoning style of Indian law. Retrieval over 128K statutory sections and 5.5M judgment chunks supplies the actual provisions and precedents. Each covers what the other lacks.
- **Training choices that held up.** RS-LoRA (scale alpha/sqrt(r) = 11.3) on all seven attention and MLP projections, loss computed only on the answer tokens, cosine schedule with early stopping. The epoch ablation shows one epoch already reaches the best ROUGE-2 and BLEU-4. In the 4-epoch run, early stopping kicked in at epoch 1.28 and saved about 9.5 GPU-hours.
- **Rigorous evaluation.** The test set is fully unseen. Models are compared against both their own base and a commercial cloud model, using perplexity, ROUGE, BLEU and a legal-citation precision/recall metric.
- **Sovereign and private.** Training, evaluation and inference all run on hardware we control (an on-premise HPC and an ordinary laptop). Client facts never have to leave the machine, which matters for legal work.
- **Reproducible.** Data split with a fixed seed (42), Slurm jobs that checkpoint and resume themselves across time limits, and all adapters, GGUFs, data and the index stored on Hugging Face.

Full hyperparameters, formulas and training logs: [`to_finetune/FINETUNING_INFO.md`](to_finetune/FINETUNING_INFO.md).

---

Every domain follows the same recipe: fine-tune an open model on that domain's data, ground its answers in a library of verified documents, and run it on an ordinary laptop. The rest of this README covers the first live domain, **Legal**: fine-tuned Qwen3-VL, federated legal retrieval and a document-drafting agent, running on your own hardware.

The legal domain is a locally deployable assistant for Indian law. Ask about statutes, precedents or a practical dispute, and get an answer grounded in retrieved Acts and court judgments. Ask it to draft a document, and it exports a court-style Word file: a bail application, plaint, writ petition, Section 138 notice and more.

- **Domain fine-tuning.** Qwen3-VL-4B and 8B adapted with RS-LoRA on 47.5k Indian legal instruction pairs (about 0.7% of weights trained). Served by llama.cpp as 4-bit GGUFs: about 100 tokens/s (4B) and 63 tokens/s (8B) on an 8 GB laptop GPU.
- **Federated retrieval.** LanceDB search over 128K statutory sections (3,100+ Acts), 547K Supreme Court and 5M High Court judgment chunks, reranked by a cross-encoder. The live web is added only when local evidence is weak.
- **Drafting agent.** Recognises the requested document type, fills a format-specific template from the conversation and retrieved law, and exports a `.docx`. Also does interest and damages calculations.
- **Private by design.** Inference can run entirely on a laptop GPU; no client facts need to leave the machine.

Full design, training runs and benchmarks: [`Confai/COMPLETE_PROJECT_SPECIFICATION.md`](Confai/COMPLETE_PROJECT_SPECIFICATION.md).

---

## Contents

1. [How it works](#how-it-works)
2. [Requirements](#requirements)
3. [Setup](#setup)
4. [Running the project](#running-the-project)
5. [Local inference (llama-server)](#local-inference-llama-server)
6. [Document formats](#document-formats)
7. [Data and models on Hugging Face](#data-and-models-on-hugging-face)
8. [Repository layout](#repository-layout)
9. [Training and evaluation](#training-and-evaluation)
10. [Known limitations](#known-limitations)
11. [Troubleshooting](#troubleshooting)

---

## How it works

```
question
  -> intent router            STATUTE | PRECEDENT | GENERAL_ADVICE | CURRENT_INFORMATION | DRAFTING | ...
       -> document format     (drafting only) e.g. bail application, plaint, writ petition
  -> retrieval                LanceDB: Acts + Supreme Court + High Court (BGE-small embeddings)
  -> cross-encoder rerank     ms-marco-MiniLM-L-6-v2
  -> evidence decision        local only | hybrid (local + web) | web only
  -> answer                   grounded answer, or the agent for drafting / calculations
  -> .docx export             format-specific template
```

The evidence used depends on how well the legal corpus answers the question:

| Situation | Evidence used |
|---|---|
| Non-legal question (e.g. "capital of France") | Web only |
| Recent developments or broad scenarios | Local + web |
| Strong local match (top rerank score at least 3) | Local corpus only |
| Partial local match | Relevant local passages plus web (at least 2 web passages kept) |
| Nothing relevant locally (top score below -2) | Web only |
| Web needed but unavailable | Best local evidence instead of failing |

The thresholds were measured on this corpus: covered legal questions score 3 to 8, partial matches -1 to 3, off-topic questions below -2. The logic lives in [`src/pipeline.py`](src/pipeline.py).

---

## Requirements

| | Minimum | Notes |
|---|---|---|
| Python | 3.12 | Developed on 3.12.10 |
| GPU | NVIDIA, 8 GB VRAM | Holds one model at a time: 4B (~4 GB) or 8B (~6 GB) with an 8K context. Without a GPU, use the Gemini or HPC backend. |
| NVIDIA driver | CUDA 13.4 capable | For the bundled llama.cpp build; `requirements.txt` installs PyTorch built for CUDA 13.2 |
| Disk | ~19 GB minimum, ~32 GB for everything | Index ~15 GB, GGUF models 2.5 GB (4B) / 5 GB (8B) each, llama.cpp 0.7 GB |
| Accounts (optional) | Tavily, Google Gemini, Hugging Face | Tavily for web search, Gemini as a cloud backend and router, Hugging Face to pull the data |

Developed and tested on Windows 11. The code is cross-platform; the HPC training scripts target Linux with Slurm.

---

## Setup

### Quick setup (recommended)

```bash
git clone https://github.com/IshanJawale/RAG-Based-Domain-Specific-Fine-Tuned-AI-System.git
cd RAG-Based-Domain-Specific-Fine-Tuned-AI-System
bash setup.sh          # Windows: run in Git Bash
```

`setup.sh` creates `.venv` and installs `requirements.txt`, creates a `.env` template if there is none, logs in to Hugging Face (using `HF_TOKEN` from `.env` if set), downloads llama-server, the GGUF models, the RAG index, the embedder and the reranker, and finally checks the index and asks the 4B model a test question. It is safe to re-run: finished downloads are skipped.

| Option | Effect |
|---|---|
| `--models "legal-qwen3-vl-4b"` | Download only these GGUFs (default: all four, ~15 GB). Names: `legal-qwen3-vl-4b`, `legal-qwen3-vl-8b`, `qwen3-vl-4b-instruct`, `qwen3-vl-8b-instruct` |
| `--skip-index` | Skip the 15 GB RAG index |
| `--with-hf-weights` | Also download the LoRA adapters and base Qwen weights (~27 GB), only needed for the transformers fallback or to rebuild GGUFs |

Ollama is not needed. The manual steps below do the same thing.

### 1. Clone and create an environment

```bash
git clone https://github.com/IshanJawale/RAG-Based-Domain-Specific-Fine-Tuned-AI-System.git
cd RAG-Based-Domain-Specific-Fine-Tuned-AI-System

python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\activate
# Linux / macOS
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt` already points pip at the PyTorch CUDA 13.2 wheel index. If your driver is older, install a matching PyTorch build first from <https://pytorch.org/get-started/locally/> and remove the `torch`/`torchvision` lines before running the command above.

Check the GPU is visible:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

### 3. Configure keys

Create a `.env` file in the project root. It is git-ignored; never commit it.

```ini
TAVILY_API_KEY="tvly-..."     # web search and weak-evidence fallback (optional but recommended)
GEMINI_API_KEY="..."          # optional: Gemini backend and faster routing of ambiguous questions
HF_TOKEN="hf_..."             # needed to download the private Hugging Face repos

# Optional
# ACTIVE_MODEL="Qwen3-VL-8B-Legal (Fine-Tuned)"   # default model; otherwise HPC if reachable, else the 4B legal model
# HPC_SERVER_URL="http://localhost:8000"          # HPC inference server (through an SSH tunnel)
```

Without `TAVILY_API_KEY` the app still runs; it answers from the local corpus only. Without `GEMINI_API_KEY` the Gemini options still appear in the model list but return an error if selected, and ambiguous questions are routed with the local model instead.

### 4. Download the models, llama-server and the index

All large files are on Hugging Face (see [Data and models on Hugging Face](#data-and-models-on-hugging-face) for details). From the project root:

```bash
hf auth login                                     # or rely on HF_TOKEN in .env

# Fine-tuned models as GGUFs -> Models/gguf/ (2.5 GB for 4B, 5 GB for 8B; leave out --include for all four)
hf download ishan-jawale/indian-legal-qwen3-vl-gguf --include legal-qwen3-vl-4b.Q4_K_M.gguf --local-dir Models/gguf

# llama-server (Windows, CUDA 13.4) -> tools/llama.cpp/ (~0.7 GB)
hf download ishan-jawale/indian-legal-llama-cpp-tools --local-dir tools/llama.cpp
# Linux: extract llama-b11836-bin-ubuntu-cuda-13.4-x64.tar.gz from
# https://github.com/ggml-org/llama.cpp/releases/tag/b11836 into tools/llama.cpp (setup.sh does this)

# RAG index -> data/lancedb/ (~15 GB)
hf download ishan-jawale/indian-legal-rag-index --repo-type dataset --local-dir data

# Retrieval embedder and reranker (the embedder is loaded offline, so it must be cached first)
hf download BAAI/bge-small-en-v1.5
hf download cross-encoder/ms-marco-MiniLM-L-6-v2
```

On Windows, give `hf download` one `--include`/`--exclude` per pattern and avoid wildcards where you can: on Windows the patterns can get expanded against the current folder before `hf` sees them.

Optional, only for the transformers fallback or to rebuild the GGUFs: the LoRA adapters (`hf download ishan-jawale/indian-legal-lora-adapters --local-dir Models`) and the base weights (`hf download Qwen/Qwen3-VL-8B-Instruct --local-dir Models/Qwen3-VL-8B-Instruct`, same for 4B).

### 5. Verify

```bash
python -c "import lancedb; db = lancedb.connect('data/lancedb'); print({t: db.open_table(t).count_rows() for t in db.table_names()})"
# expected: {'acts': 127806, 'supreme_court': 546547, 'high_court': 5002082}
```

---

## Running the project

### Web app

```bash
streamlit run app/app.py
```

Open <http://localhost:8501>. The landing page lists domains; **Legal** is live and the others are placeholders. In the legal chat:

- **Sidebar, top:** new chat and the recent-chats list (saved to `app/chat_history/legal.json`). The menu next to a chat's title renames or deletes it, and the download icon exports it as Markdown.
- **LLM backend:** pick the model. Options appear depending on what is available:
  - *Qwen3-VL-8B-Legal / 4B-Legal (Fine-Tuned)*: the fine-tuned models, run locally by llama-server (see [Local inference](#local-inference-llama-server)).
  - *Qwen3-VL-4B / 8B-Instruct*: the same models without fine-tuning, for comparison.
  - *Qwen3-VL-Legal (HPC Cluster)*: the fine-tuned model served from the HPC (see below).
  - *Gemini (Cloud API)*: needs `GEMINI_API_KEY`.
- **Agent and search settings:** toggle agent tool calling (needed for `.docx` export), chat context (follow-ups), number of retrieval candidates and reranked passages, and whether to show citations and routing.
- Answers **stream** as they are generated (local and Gemini models; the HPC server returns its answer in one piece).
- **Follow-ups** such as *"and the limitation period for it?"* are resolved against the previous question, and the last few messages are passed to the model.
- Each answer has expandable **Query routing** (intent, document format, evidence used and why) and **Retrieved authorities** panels.
- Generated documents get a download button, an in-chat **preview** (blanks highlighted, plus the drafter's checklist) and a **Fill in missing details** form that re-renders the `.docx`. You can also ask for changes in chat, e.g. *"change the cheque amount to Rs 5,00,000"*, and the agent revises the same document.

The first answer with a local model, or after switching models, waits 3-5 seconds while llama-server loads it onto the GPU.

### Command line

```bash
python main.py "What is the procedure for anticipatory bail under Section 482 BNSS?"   # one question
python main.py                                                                         # interactive
python main.py --model "Qwen3-VL-8B-Legal (Fine-Tuned)" "Explain Article 21"
```

### Using the HPC-hosted model

Start the inference server on the cluster (`to_finetune/job_server.sh`), then open a tunnel from the laptop:

```bash
ssh -N -L 8000:cn02:8000 YOUR_HPC_USER@hpc.plaksha.edu.in
```

The app detects the server on `localhost:8000` and lists it as a backend; the sidebar shows the GPU and model it is running.

---

## Local inference (llama-server)

Local models run on [llama.cpp](https://github.com/ggml-org/llama.cpp)'s `llama-server` from 4-bit (Q4_K_M) GGUF files. `src/llm.py` starts it in the background on the first local request, loads one model at a time (an 8 GB GPU holds one), switches it when you pick another model, and stops it when the app exits or you switch to Gemini or the HPC. It listens only on `127.0.0.1`, so it is not reachable from the network.

| Model | Generation | Reading a 3K-token RAG prompt | 400-token answer |
|---|---|---|---|
| 4B (fine-tuned or base) | ~100 tokens/s | ~4,200 tokens/s | ~4 s |
| 8B (fine-tuned or base) | ~63 tokens/s | ~2,500 tokens/s | ~6.4 s |

Measured on an RTX 5060 Laptop GPU (8 GB), fully offloaded with an 8K context. The previous in-process transformers path (bitsandbytes 4-bit with the LoRA adapter applied at run time) managed about 3.6 tokens/s on the 4B. Ollama gave the same numbers as llama-server (it runs llama.cpp underneath), so it is not used.

The `legal-*` GGUFs have the LoRA adapter merged into the base weights. They are text-only (the chat never sends images). Sampling matches the earlier setup: temperature 0.2, top-p 0.9, repetition penalty 1.1.

Settings (in `.env`, all optional):

| Variable | Default | Effect |
|---|---|---|
| `LOCAL_BACKEND` | `llama-server` | `transformers` forces the old in-process path (needs `--with-hf-weights`) |
| `LLAMA_CTX` | `8192` | Context window. Larger needs more VRAM; the 8B model may no longer fit fully on an 8 GB GPU. |
| `LLAMA_SERVER_PORT` | `8089` | Port llama-server listens on |

If a GGUF or `tools/llama.cpp` is missing, that model falls back to the transformers path when its base weights are present, and is hidden from the model list otherwise.

**Rebuilding the GGUFs** after retraining an adapter (needs `--with-hf-weights`, ~2x the model size in free RAM):

```bash
python scripts/build_gguf.py 4b          # -> Models/gguf/legal-qwen3-vl-4b.Q4_K_M.gguf
python scripts/build_gguf.py 8b
python scripts/build_gguf.py 4b --base   # -> qwen3-vl-4b-instruct.Q4_K_M.gguf (no adapter)
```

---

## Document formats

Ask in plain language, e.g. *"Draft an anticipatory bail application for…"*. The router marks the request as **DRAFTING**, a sub-classifier picks the format below, and the agent fills that format's fields. Each document is A4, Times New Roman, 1.5 spacing, with a cause title, continuously numbered paragraphs, prayer and verification where the format needs them. Anything the user did not provide is left as **`[●]`** rather than invented. The last page carries a **drafter's checklist** (limitation periods, annexures, forum) to remove before filing.

| Format | Statutory basis | Example request |
|---|---|---|
| Cheque dishonour notice | Ss. 138, 142 Negotiable Instruments Act, 1881 | "Draft a cheque bounce notice for Rs 2 lakh" |
| Notice to Government | S. 80 Code of Civil Procedure, 1908 | "Draft a notice to the state government before filing a suit" |
| Legal notice (demand) | Provisions breached, e.g. S. 73 Contract Act | "Draft a legal notice for breach of contract" |
| Plaint | Order VII Rule 1, Order VI Rule 15 CPC | "Draft a plaint for a recovery suit of Rs 15 lakh" |
| Written statement | Order VIII CPC | "Prepare a written statement in reply to the plaint" |
| Regular bail application | Ss. 480, 483 BNSS, 2023 (old Ss. 437, 439 CrPC) | "Draft a bail application for an accused in custody" |
| Anticipatory bail application | S. 482 BNSS, 2023 (old S. 438 CrPC) | "Draft an anticipatory bail application, my client fears arrest" |
| Police complaint / FIR request | S. 173 BNSS, 2023 (old S. 154 CrPC) | "Draft a police complaint for theft of my bike" |
| Writ petition | Article 226, Constitution of India | "Write a writ petition challenging a transfer order" |
| Consumer complaint | Ss. 35, 39 Consumer Protection Act, 2019 | "Draft a consumer complaint for a defective phone" |
| RTI application | S. 6(1) Right to Information Act, 2005 | "Draft an RTI application for my ration card status" |
| Affidavit | Order XIX CPC (when filed in court) | "Draft an affidavit for name change" |
| Legal opinion / memorandum | Advisory | "Write a legal opinion on whether the arbitration clause is valid" |
| Case brief | Judgment analysis | "Prepare a case brief of Maneka Gandhi v. Union of India" |
| Legal assessment report | Research / advisory (default) | "Prepare a report on land dispute scenarios" |

Formats are defined declaratively in [`src/legal_formats.py`](src/legal_formats.py): each entry lists its fields (sent to the model as the JSON schema to fill), layout blocks, classifier patterns and checklist. To add a format, add an entry to `FORMATS` and its key to `FORMAT_ORDER` (most specific patterns first).

> **These are drafting aids, not legal advice.** Statutory references and checklists were compiled from bare acts and practitioner guides; verify them against the current law, the specific court's rules (High Court writ rules differ) and have a qualified advocate review any document before it is filed or sent.

---

## Data and models on Hugging Face

GitHub rejects files over 100 MB, so the index, training data and adapters live on Hugging Face and are downloaded into the folders the code expects.

| Repo | Type | Contents | Size | Local folder |
|---|---|---|---|---|
| [`ishan-jawale/indian-legal-qwen3-vl-gguf`](https://huggingface.co/ishan-jawale/indian-legal-qwen3-vl-gguf) | model | Q4_K_M GGUFs: fine-tuned 4B and 8B (adapters merged) and the base 4B and 8B | ~15 GB | `Models/gguf/` |
| [`ishan-jawale/indian-legal-llama-cpp-tools`](https://huggingface.co/ishan-jawale/indian-legal-llama-cpp-tools) | model | llama.cpp b11836 for Windows + CUDA 13.4 (`llama-server.exe`, CUDA DLLs, GGUF converter) | ~0.7 GB | `tools/llama.cpp/` |
| [`ishan-jawale/indian-legal-lora-adapters`](https://huggingface.co/ishan-jawale/indian-legal-lora-adapters) | model | four LoRA adapters (8B; 4B at 1, 2 and 4 epochs) | ~1.1 GB | `Models/` |
| [`ishan-jawale/indian-legal-rag-index`](https://huggingface.co/datasets/ishan-jawale/indian-legal-rag-index) | dataset | LanceDB tables, `acts_chunks.jsonl`, `curated_for_rag.jsonl` | ~16 GB | `data/` |
| [`ishan-jawale/indian-legal-sft-50k`](https://huggingface.co/datasets/ishan-jawale/indian-legal-sft-50k) | dataset | `train.jsonl` (47.5k), `val.jsonl` (2.5k) | ~150 MB | `to_finetune/data/` |

The repos are private; ask the owner for access, then authenticate with `hf auth login` or `HF_TOKEN`.

**Cost:** free. Free Hugging Face accounts include 100 GB of private storage ([storage limits](https://huggingface.co/docs/hub/storage-limits)); this project uses about 32 GB. Public storage is free on a best-effort basis.

### Download

```bash
hf download ishan-jawale/indian-legal-qwen3-vl-gguf --local-dir Models/gguf
hf download ishan-jawale/indian-legal-qwen3-vl-gguf --include legal-qwen3-vl-8b.Q4_K_M.gguf --local-dir Models/gguf   # one model only
hf download ishan-jawale/indian-legal-llama-cpp-tools --local-dir tools/llama.cpp
hf download ishan-jawale/indian-legal-lora-adapters --local-dir Models
hf download ishan-jawale/indian-legal-lora-adapters --include "legal-lora-vl-8b-adapter/*" --local-dir Models   # one adapter only
hf download ishan-jawale/indian-legal-rag-index --repo-type dataset --local-dir data
hf download ishan-jawale/indian-legal-sft-50k --repo-type dataset --local-dir to_finetune
```

### Upload (maintainers)

`hf upload` creates the repo if it does not exist (`--private` only applies at creation) and splits large folders into multiple commits. Re-run the same command to resume an interrupted upload.

```bash
hf auth login   # token with WRITE access: https://huggingface.co/settings/tokens

# GGUF models (after scripts/build_gguf.py)
hf upload ishan-jawale/indian-legal-qwen3-vl-gguf Models/gguf . --include "*.gguf" --repo-type model --private

# llama.cpp tools. Use the Python API: on Windows the ** in --exclude got expanded into local paths.
python -c "from huggingface_hub import HfApi; HfApi().upload_folder(repo_id='ishan-jawale/indian-legal-llama-cpp-tools', folder_path='tools/llama.cpp', ignore_patterns=['**/__pycache__/**', '*.pyc'])"

# Adapters
hf upload ishan-jawale/indian-legal-lora-adapters Models/legal-lora-vl-8b-adapter legal-lora-vl-8b-adapter --repo-type model --private
hf upload ishan-jawale/indian-legal-lora-adapters Models/legal-lora-vl-4b-adapter legal-lora-vl-4b-adapter --repo-type model
hf upload ishan-jawale/indian-legal-lora-adapters Models/legal-lora-vl-4b-1ep-adapter legal-lora-vl-4b-1ep-adapter --repo-type model
hf upload ishan-jawale/indian-legal-lora-adapters Models/legal-lora-vl-4b-4ep-adapter legal-lora-vl-4b-4ep-adapter --repo-type model

# Fine-tuning data
hf upload ishan-jawale/indian-legal-sft-50k to_finetune/data data --repo-type dataset --private

# RAG index (12 GB of it is the High Court table)
hf upload ishan-jawale/indian-legal-rag-index data/lancedb lancedb --repo-type dataset --private
hf upload ishan-jawale/indian-legal-rag-index data/acts_chunks.jsonl acts_chunks.jsonl --repo-type dataset
hf upload ishan-jawale/indian-legal-rag-index "law dataset/curated_for_rag.jsonl" curated_for_rag.jsonl --repo-type dataset
```

The Hub validates model-card metadata: an adapter `README.md` whose `base_model` is a local path (e.g. `/scratch/...`) is rejected. Use the Hub id, e.g. `base_model: Qwen/Qwen3-VL-8B-Instruct`.

Not uploaded (re-downloadable or scratch): `finetuning_data/` (the public Indian-legal-data-v3 dataset), base Qwen models, `hc_ingest_progress.db`, `_skipped_tmp.pkl`, `filtered_data.txt`. Before making any repo public, confirm the licence of the scraped Acts and judgments (Indian-legal-data-v3 itself is MIT).

---

## Repository layout

| Path | What it is |
|---|---|
| `app/app.py` | Streamlit entry point (navigation) |
| `app/app_pages/` | `home.py` (domain landing page), `legal_chat.py` (legal chat) |
| `app/chat_store.py` | Chat history persistence |
| `.streamlit/config.toml` | App theme |
| `src/pipeline.py` | Shared pipeline: routing, retrieval, evidence decision, answer (used by the app and CLI) |
| `src/routing.py` | Intent router (regex fast paths, LLM fallback) |
| `src/retrieval.py`, `src/reranking.py` | LanceDB retrieval and cross-encoder reranking |
| `src/generation.py`, `src/llm.py` | Grounded prompts; model backends (llama-server, transformers fallback, HPC server, Gemini) |
| `src/agent.py`, `src/tools.py` | Tool-using agent: drafting, calculator, search tools |
| `src/legal_formats.py` | Document format registry, sub-classifier and `.docx` renderer |
| `src/document_generator.py` | Document dispatcher; opinion, case brief and report templates |
| `src/ingest_*.py`, `src/build_acts_index.py` | Building the LanceDB index (`ingest_more_acts.py` adds missing central Acts; run with `--dry-run` first) |
| `main.py` | Command-line interface |
| `setup.sh` | One-shot setup: environment, Hugging Face downloads, checks |
| `scripts/build_gguf.py` | Merges a LoRA adapter and converts it to the Q4_K_M GGUF llama-server runs |
| `to_finetune/` | HPC training (`train.py`, Slurm `job*.sh`), inference server, evaluation and reports |
| `finetune/` | Earlier synthetic-data pipeline and dataset preparation |
| `law dataset/` | Scripts that download and extract Acts and judgments |
| `data/`, `Models/`, `to_finetune/data/`, `tools/llama.cpp/` | **Not in git**: downloaded from Hugging Face |
| `generated_documents/` | **Not in git**: exported `.docx` files |

---

## Training and evaluation

Fine-tuning runs on the Plaksha HPC with Slurm: RS-LoRA (r = 32, alpha = 64) on all attention and MLP projections, bf16, 4,096-token sequences, 2 epochs. See [`to_finetune/README.md`](to_finetune/README.md) and [`to_finetune/FINETUNING_INFO.md`](to_finetune/FINETUNING_INFO.md).

Headline results on 100 held-out questions (full tables in the specification):

| Model | Perplexity | ROUGE-2 | BLEU-4 |
|---|---|---|---|
| Qwen3-VL-4B base -> fine-tuned (1 epoch) | 3.84 -> 2.16 | 9.33% -> 14.85% | 2.74 -> 4.57 |
| Qwen3-VL-8B base -> fine-tuned | 3.63 -> 2.06 | 9.52% -> 13.98% | 2.90 -> 4.11 |

Reports: `to_finetune/evaluation_report/` and `eval_results_local/`.

---

## Known limitations

- **Acts index coverage.** 583 central Acts (Contract, IPC, Evidence, NI, Transfer of Property, Specific Relief, Limitation, Hindu Succession, PMLA, RTI, and others) were added with `src/ingest_more_acts.py` from [KanoonGPT/indian-legal-documents](https://huggingface.co/datasets/KanoonGPT/indian-legal-documents) and [civictech-India](https://github.com/civictech-India/Indian-Law-Penal-Code-Json). Still missing: the **CPC First Schedule (Orders and Rules)**, e.g. Order VII Rule 11, since only the CPC's sections are indexed. Some Acts were parsed partially (e.g. the Income-tax Act's sections are long and lettered), and a few Acts now appear under two titles (e.g. "POCSO Act, 2012" and its full title).
- **No keyword (BM25) retrieval yet.** Retrieval is dense-only, so exact citations like "Order VII Rule 11" can rank below semantically similar text.
- **Follow-up detection is heuristic.** Short questions that start with *and/what about/how about* or refer back (*it, this, the section*) are combined with the previous question; anything else is treated as a new question. Turn off *Use chat context* to disable it.
- **Drafts need review.** Document templates are a starting point; see the note under [Document formats](#document-formats).

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `OSError ... bge-small-en-v1.5 ... local_files_only` | Run `hf download BAAI/bge-small-en-v1.5` once on that machine. |
| `401` / `Repository not found` when downloading | The repos are private: run `hf auth login` or set `HF_TOKEN`, and make sure you have access. |
| `pip` cannot find `torch==...+cu132` | Your platform or driver needs a different build: install PyTorch from pytorch.org, then remove the torch lines from `requirements.txt`. |
| Local model missing from the list | Its GGUF is not in `Models/gguf/` or `tools/llama.cpp/` is missing: re-run `bash setup.sh --models "<name>"`. |
| `llama-server failed to start` | The error shows the end of its log (also in your temp folder as `legal-rag-llama-server-8089.log`). Usually the GPU is full (close other GPU apps such as Ollama, or use the 4B model), the driver is too old for CUDA 13.4, or port 8089 is taken (set `LLAMA_SERVER_PORT`). |
| Local answers very slow (a few tokens/s) | The app is on the transformers fallback: check that the GGUF and `tools/llama.cpp/llama-server(.exe)` exist and `LOCAL_BACKEND` is not `transformers`. |
| CUDA out of memory | Use the 4B model, lower `LLAMA_CTX`, close other GPU apps, or switch to the HPC or Gemini backend. |
| "HPC tunnel not connected" | Start the server on the cluster and open the SSH tunnel shown above. |
| No web results | Set `TAVILY_API_KEY` in `.env`; the sidebar shows whether web search is enabled. |
| No download button after a drafting request | Turn on **Agent tool calling** in the sidebar. |
| Push rejected for secrets | Never hardcode keys; keep them in `.env`. GitHub push protection blocks commits containing them. |
