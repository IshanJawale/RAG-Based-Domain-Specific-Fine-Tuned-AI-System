# Fine-tuning the legal RAG model

Goal: teach the model the *behaviour* the RAG app needs - answer **only** from retrieved evidence, cite it
(`[E1]`, `[E2]` ...), refuse cleanly when evidence is missing, and keep the query-router JSON intact.
Facts come from retrieval; fine-tuning shapes how the model uses them.

```
finetune/
  common.py              shared helpers (teacher client, JSONL utils; imports the app's real prompts)
  01_sample_seeds.py     sample clean passages from the LanceDB corpus
  02_generate.py         questions -> REAL retrieval -> teacher answers (+ routing labels)
  03_build_dataset.py    filter / judge / balance / split-by-document -> train.jsonl, val.jsonl
  modeling.py            model loading (dense, MoE, VL; bf16 or 4-bit)
  04_train.py            LoRA / QLoRA SFT
  05_eval_and_merge.py   base-vs-tuned metrics, merge adapter
  requirements-finetune.txt
```

## Where each step runs

| Steps | Machine | Why |
|---|---|---|
| 01, 02, 03 | **Your laptop / any box that has `data/lancedb`** | needs the vector DB; the teacher is an API |
| 04, 05 | **GPU box (H200 / RTX 6000)** | copy only `finetune/` and `data/train.jsonl`, `val.jsonl` |

## 1. Build the dataset (local)

Add a teacher to `.env`. Any OpenAI-compatible endpoint works:

```ini
# Gemini
TEACHER_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai
TEACHER_MODEL=gemini-2.5-flash
TEACHER_API_KEY=...

# or a model served by vLLM on your H200 (free, fast; strongest option = biggest model you can serve)
# TEACHER_BASE_URL=http://<h200-host>:8000/v1
# TEACHER_MODEL=Qwen/Qwen3-235B-A22B-Instruct-2507-FP8     (whatever you serve)
# TEACHER_API_KEY=EMPTY
# TEACHER_DISABLE_THINKING=1
```

Then (resumable at every step - re-run the same command after any interruption):

```powershell
python finetune/01_sample_seeds.py --n 12000
python finetune/02_generate.py --phase questions --limit 20     # smoke test first, read data/questions.jsonl
python finetune/02_generate.py --phase all --workers 32 --oos 1000
python finetune/03_build_dataset.py --judge                     # judge = extra faithfulness audit pass
```

* `--n 12000` seeds ≈ 12k answer + 12k router examples. More data is cheap with a hosted/vLLM teacher; 12-25k is a good
  range for behaviour tuning (quality matters more than count).
* Phase B (retrieval) is the slow local part (~1-2 s/question). Run it overnight; `--retrieval-workers` tunes it.
* **Read ~50 random lines of `data/train.jsonl` before training.** Garbage in = garbage out.

What the data looks like: the `user` turn is the **exact production prompt** (built by `src.generation.build_prompt`
over what the real retriever + cross-encoder returned, with 3-7 evidence chunks), the `assistant` turn is the teacher's
cited answer or the refusal sentence. About 15-22% are refusals (retrieval genuinely missed, or the question is
off-topic / foreign law / needs live data), capped by `--max-refusal`.

## 2. Train (GPU box)

```bash
pip install -r finetune/requirements-finetune.txt        # + a CUDA build of torch
python finetune/04_train.py --model /path/to/Qwen3-VL-30B-A3B-Instruct
```

| GPU | Recipe |
|---|---|
| **H200 (141 GB)** | default: bf16 weights (~61 GB) + LoRA. Fast, no quantisation issues. |
| **RTX 6000 Pro (96 GB)** | same as H200, lower batch size / `--max-len 6144` if needed. |
| **RTX 6000 Ada (48 GB)** | 30B bf16 does **not** fit. `--quant 4bit` (see caveat) or train Qwen3-8B/14B instead. |

Caveat on 4-bit + MoE: newer `transformers` fuse the MoE experts into 3-D tensors that bitsandbytes cannot quantise,
so 4-bit saves less than expected on 30B-A3B. If it OOMs, use the H200 or a dense 8B/14B model.

Useful flags: `--epochs 2` · `--lr 1e-4` · `--batch-size 2 --grad-accum 8` · `--lora-r 32` · `--max-len 8192`
· `--resume` (continue after a crash) · `--report-to tensorboard` · `--limit 200 --max-steps 20` (dry run).
Multi-GPU: `torchrun --nproc_per_node=2 finetune/04_train.py --model ...`.

What the script does for you: assistant-only loss masking, drops (never truncates) over-long examples, rsLoRA, cosine LR
with warmup, grad clipping, gradient checkpointing, length-grouped batches, eval-loss tracking + early stopping +
best-checkpoint restore, resumable checkpoints, and a text-only forward probe for VL models.

Which model? For a **text-only** RAG app, `Qwen3-30B-A3B-Instruct-2507` (text) is the simpler, more robust choice;
the VL variant works too (vision tower stays frozen) but its vision weights are dead weight here.

## 3. Evaluate and merge

```bash
python finetune/05_eval_and_merge.py --model /path/to/base --adapter finetune/outputs/legal-lora/final \
       --merge-to finetune/outputs/legal-lora/merged
```

Prints a base-vs-tuned table on held-out documents (never seen in training): refusal agreement/recall, false-refusal
rate, citation validity, format compliance, hedge-word rate, router JSON validity + accuracy. You want citation
validity and format ≈ 1.0, false-refusal low, router accuracy not worse than base.

## 4. Use it in the app

Copy `merged/` to your machine and add it to `AVAILABLE_MODELS` in `src/llm.py`, or serve it with vLLM
(`vllm serve <merged-dir>`) and point the app at the endpoint. The app can map `[E#]` markers to source cards in the UI.

## Notes

* Retrieval was changed to take a quota from each corpus (`LegalRetriever.search`) so Acts and Supreme Court passages
  are no longer drowned by the 5M High Court chunks; training data is generated with the same retriever the app uses.
* Keep the prompt template in `src/generation.py` stable once you train - the fine-tune is tied to it.
