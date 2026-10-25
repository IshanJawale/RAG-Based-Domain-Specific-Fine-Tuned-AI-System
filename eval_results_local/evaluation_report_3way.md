# 3-Way Benchmark: Qwen Base vs Fine-Tuned Legal vs Gemini 3.7 Flash

> **Superseded pilot run.** This is an early evaluation on a smaller question set. The reference benchmark is the 100-question evaluation in `to_finetune/FINETUNING_INFO.md` (Section 8) and `evaluation_report.md`. Do not quote these numbers alongside it.

**Evaluated on:** 25 held-out Indian Legal questions (`val.jsonl`)

## 📊 Benchmark Score Comparison Table

| Model | Deployment | ROUGE-1 ↑ | ROUGE-2 ↑ | ROUGE-L ↑ | BLEU-4 ↑ | Legal Citation Recall ↑ | Speed (tok/s) ↑ | VRAM Footprint |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-VL-4B (Base)** | Local GPU (4-bit) | 27.51% | 10.22% | 17.23% | 2.57 | **69.33%** | 20.7 | 2.79 GB |
| **Qwen3-VL-4B (Fine-Tuned)** | Local GPU (4-bit) | 30.59% | 15.15% | 20.83% | 4.33 | **72.33%** | 11.6 | 3.25 GB |
| **Gemini 3.7 Flash** | Cloud API (Zero VRAM) | 1.84% | 0.5% | 1.66% | 0.0 | **21.33%** | 1.4 | 0.0 GB |
