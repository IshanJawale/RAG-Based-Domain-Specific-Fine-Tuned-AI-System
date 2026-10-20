# Master Benchmark Report: Legal Models Comparison

- **Evaluated on:** 100 strictly unseen Indian Legal test questions (`val.jsonl`)
- **Execution Node:** Local/Cluster | Generated at: 2026-10-16 00:17:56

## 📊 Table 1: Model Accuracy & Legal Specialization Metrics

| Model | Deployment | Perplexity (PPL) ↓ | Eval Loss ↓ | ROUGE-2 (Bigram) ↑ | BLEU-4 ↑ | Citation Recall ↑ | Citation F1 ↑ | Legal Terminology ↑ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-VL-4B (Base)** | HPC GPU (bf16) | **3.84** | 1.3442 | **9.33%** | **2.74** | **81.11%** | 75.97% | 0.82% |
| **Qwen3-VL-4B (1-Epoch LoRA)** | HPC GPU (bf16) | **2.16** | 0.7702 | **14.85%** | **4.57** | **78.34%** | 79.94% | 0.45% |
| **Qwen3-VL-4B (2-Epoch LoRA)** | HPC GPU (bf16) | **2.14** | 0.7592 | **14.54%** | **4.34** | **78.52%** | 76.6% | 0.68% |
| **Qwen3-VL-4B (4-Epoch LoRA)** | HPC GPU (bf16) | **2.23** | 0.8011 | **14.57%** | **4.42** | **78.71%** | 76.95% | 0.76% |
| **Qwen3-VL-8B (Base)** | HPC GPU (bf16) | **3.63** | 1.2884 | **9.52%** | **2.9** | **80.17%** | 76.19% | 0.74% |
| **Qwen3-VL-8B (Fine-Tuned)** | HPC GPU (bf16) | **2.06** | 0.7214 | **13.98%** | **4.11** | **74.51%** | 74.6% | 0.6% |
| **Gemini 3.7 Flash** | Cloud API (Zero VRAM) | **N/A (API)** | N/A (API) | **10.91%** | **2.87** | **85.06%** | 79.18% | 0.87% |


## ⚡ Table 2: Lexical Overlap, Latency & VRAM Efficiency

| Model | Deployment | ROUGE-1 ↑ | ROUGE-L ↑ | BLEU-1 ↑ | BLEU-2 ↑ | Throughput (tok/s) ↑ | Latency (s) ↓ | Peak GPU VRAM |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen3-VL-4B (Base)** | HPC GPU (bf16) | 27.34% | 16.86% | 11.66 | 6.63 | 63.2 tok/s | 3.16s | **10.7 GB** |
| **Qwen3-VL-4B (1-Epoch LoRA)** | HPC GPU (bf16) | 32.08% | 21.35% | 12.26 | 8.14 | 33.0 tok/s | 5.48s | **10.94 GB** |
| **Qwen3-VL-4B (2-Epoch LoRA)** | HPC GPU (bf16) | 31.89% | 20.79% | 12.11 | 7.93 | 32.5 tok/s | 5.62s | **10.94 GB** |
| **Qwen3-VL-4B (4-Epoch LoRA)** | HPC GPU (bf16) | 31.89% | 20.74% | 12.26 | 8.1 | 32.1 tok/s | 5.64s | **10.94 GB** |
| **Qwen3-VL-8B (Base)** | HPC GPU (bf16) | 27.4% | 16.85% | 11.51 | 6.7 | 41.1 tok/s | 4.84s | **18.77 GB** |
| **Qwen3-VL-8B (Fine-Tuned)** | HPC GPU (bf16) | 31.33% | 20.54% | 12.04 | 7.68 | 32.5 tok/s | 5.55s | **19.09 GB** |
| **Gemini 3.7 Flash** | Cloud API (Zero VRAM) | 28.18% | 17.39% | 10.94 | 6.57 | 41.4 tok/s | 3.97s | **0.0 GB** |


## 🔍 Side-by-Side Qualitative Sample Comparisons

### Test Sample 1:
**User Query:**
> When was the Foreign Exchange Regulation Act passed and what is its act number?

**Gold Statutory Reference:**
> The Foreign Exchange Regulation Act was passed in 1973, and its act number is 12. This legislation, which aims to regulate the exchange of foreign currency, was enacted to ensure the stability and security of the country's financial system. The act number 12 signifies its importance and relevance in

**Qwen3-VL-4B (Base) Output:**
> The **Foreign Exchange Regulation Act (FERA)** was passed in the year **1973**.

Its **act number** is **Act No. 2 of 1973**.

This legislation was enacted by the Government of India to regulate foreign exchange transactions, control the inflow and outflow of foreign currency, and ensure that foreig

**Qwen3-VL-4B (1-Epoch LoRA) Output:**
> The Foreign Exchange Regulation Act was passed in 1973, with an act number of 42.

**Qwen3-VL-4B (2-Epoch LoRA) Output:**
> The Foreign Exchange Regulation Act was passed in 1973 as Act No. 46 of 1973.

**Qwen3-VL-4B (4-Epoch LoRA) Output:**
> The Foreign Exchange Regulation Act was passed in 1973, and its act number is 42.

**Qwen3-VL-8B (Base) Output:**
> The **Foreign Exchange Regulation Act (FERA)** was passed in **1973**. Its official act number is:

> **Act No. 2 of 1973**

This legislation was enacted by the Government of India to regulate foreign exchange transactions, control capital flows, and manage foreign investments during a period of eco

**Qwen3-VL-8B (Fine-Tuned) Output:**
> The Foreign Exchange Regulation Act was passed in 1973 and its act number is 14 of 1973.

**Gemini 3.7 Flash Output:**
> The **Foreign Exchange Regulation Act (FERA)** was passed by the Parliament of India on **September 19, 1973** (and came into force on January 1, 1974). 

Its official act number is **Act No. 46 of 1973**. 

*(Note: FERA was later repealed and replaced by the Foreign Exchange Management Act (FEMA) i

---

### Test Sample 2:
**User Query:**
> What is the evidentiary value of a 'compromise decree' in subsequent proceedings under the Indian Evidence Act?

**Gold Statutory Reference:**
> The evidentiary value of a compromise decree under the Indian Evidence Act (IEA) is significant, particularly in specific legal contexts. Here's a structured overview:

1. **Admissibility as Documentary Evidence**: 
   - Under section 28 of the IEA, compromise decrees are admissible as documentary e

**Qwen3-VL-4B (Base) Output:**
> Under the **Indian Evidence Act, 1872**, a **“compromise decree”** — which refers to a court order that settles a dispute between parties through mutual agreement (often with the court’s approval) — has **significant evidentiary value** in subsequent proceedings. However, its admissibility and weigh

**Qwen3-VL-4B (1-Epoch LoRA) Output:**
> **Answer:**

Under the Indian Evidence Act, 1872, a compromise decree obtained through mutual agreement between parties has significant evidentiary weight. Here's a structured summary:

1. **Definition and Nature**: A compromise decree is an agreement reached by both parties to settle their dispute 

**Qwen3-VL-4B (2-Epoch LoRA) Output:**
> The evidentiary value of a compromise decree in subsequent proceedings under the Indian Evidence Act can be summarized as follows:

1. **Admissibility**: A compromise decree, once legally binding and enforceable, can be admitted as evidence if it relates to matters that were previously litigated bet

**Qwen3-VL-4B (4-Epoch LoRA) Output:**
> **Answer:**

A compromise decree, as defined by Section 120 of the Indian Evidence Act (IEA), is an agreement between parties to settle their dispute without going to trial. This type of settlement can be used as evidence in subsequent legal proceedings, but its admissibility and impact are governed

**Qwen3-VL-8B (Base) Output:**
> Under the **Indian Evidence Act, 1872**, a **‘compromise decree’** — which is an order passed by a court when parties to a suit voluntarily agree to settle their dispute without going to trial — has specific evidentiary value in **subsequent proceedings**. Its admissibility and probative force are g

**Qwen3-VL-8B (Fine-Tuned) Output:**
> A compromise decree, once obtained through mutual agreement and court approval, holds significant evidentiary weight in subsequent legal proceedings. Here's a structured summary of its implications:

1. **Nature of Compromise Decree**: A compromise decree is an enforceable order resulting from a set

**Gemini 3.7 Flash Output:**
> Under the Indian legal framework, a **compromise decree** (passed under Order XXIII, Rule 3 of the Code of Civil Procedure, 1908) is a decree based on an agreement or contract between the parties, with the seal of the court affixed to it. 

Its evidentiary value in subsequent proceedings under the *

---

### Test Sample 3:
**User Query:**
> What does Section 293 of the Delhi Municipal Council Act, 1957 state?

**Gold Statutory Reference:**
> ## Section 293 of the Delhi Municipal Council Act, 1957 states that the Undertaking to be run on business principles shall be conducted in a manner that ensures transparency, accountability, and efficiency in the administration of the municipality. The section mandates that the Undertaking shall be 

**Qwen3-VL-4B (Base) Output:**
> **Section 293 of the Delhi Municipal Corporation Act, 1957** (not “Delhi Municipal Council Act”, which is outdated — the current body is called the *Delhi Municipal Corporation*, governed by the *Delhi Municipal Corporation Act, 1957* as amended) **provides for the power to remove or suspend a membe

**Qwen3-VL-4B (1-Epoch LoRA) Output:**
> ## Title: Section 293 of the Delhi Municipal Council Act, 1957

### Introduction
Section 293 of the Delhi Municipal Council Act, 1957, is a critical provision that outlines the powers and responsibilities of the Commissioner in relation to the enforcement of regulations. This section grants the Comm

**Qwen3-VL-4B (2-Epoch LoRA) Output:**
> ##Section 293 of the Delhi Municipal Council Act, 1957

### Introduction
Section 293 of the Delhi Municipal Council Act, 1957, is a critical provision that outlines the powers and responsibilities of the Commissioner in relation to public health measures. This section grants the Commissioner signifi

**Qwen3-VL-4B (4-Epoch LoRA) Output:**
> ## Introduction

Section 293 of the Delhi Municipal Council Act, 1957, is a critical provision that outlines the process for recovering arrears of taxes and fees from individuals who fail to pay them. This section is crucial for ensuring that the municipal council can effectively manage its financia

**Qwen3-VL-8B (Base) Output:**
> **Section 293 of the Delhi Municipal Corporation Act, 1957 (not “Delhi Municipal Council Act” — note: the correct name is *Delhi Municipal Corporation Act, 1957*)** deals with **the power to make bye-laws for regulating the use and occupation of premises in connection with public health, safety, con

**Qwen3-VL-8B (Fine-Tuned) Output:**
> ## Title: Section 293 of the Delhi Municipal Council Act, 1957

### Introduction
Section 293 of the Delhi Municipal Council Act, 1957, is a critical provision that outlines the powers and responsibilities of the Commissioner in relation to public health and sanitation. This section grants the Commis

**Gemini 3.7 Flash Output:**
> Under the **Delhi Municipal Corporation Act, 1957** (commonly referred to in relation to the MCD), **Section 293** deals with the **"Prohibition of projection upon streets, etc."**

Here is a summary of what the section states:

### Key Provisions of Section 293:

1. **Prohibition of Projections wit

---

