# Quick Benchmark Summary

| Model | Perplexity ↓ | ROUGE-2 ↑ | BLEU-4 ↑ | Citation Recall ↑ | Speed (tok/s) ↑ | Peak VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-VL-4B (Base)** | 3.84 | 9.33% | 2.74 | 81.11% | 63.2 | 10.7 GB |
| **Qwen3-VL-4B (1-Epoch LoRA)** | 2.16 | 14.85% | 4.57 | 78.34% | 33.0 | 10.94 GB |
| **Qwen3-VL-4B (2-Epoch LoRA)** | 2.14 | 14.54% | 4.34 | 78.52% | 32.5 | 10.94 GB |
| **Qwen3-VL-4B (4-Epoch LoRA)** | 2.23 | 14.57% | 4.42 | 78.71% | 32.1 | 10.94 GB |
| **Qwen3-VL-8B (Base)** | 3.63 | 9.52% | 2.9 | 80.17% | 41.1 | 18.77 GB |
| **Qwen3-VL-8B (Fine-Tuned)** | 2.06 | 13.98% | 4.11 | 74.51% | 32.5 | 19.09 GB |
| **Gemini 3.7 Flash** | N/A (API) | 10.91% | 2.87 | 85.06% | 41.4 | 0.0 GB |
