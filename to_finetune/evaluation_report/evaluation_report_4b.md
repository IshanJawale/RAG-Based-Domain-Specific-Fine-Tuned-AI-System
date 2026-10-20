# Local Benchmark: Qwen3-VL-4B (Base) vs Qwen3-VL-4B (Fine-Tuned Legal LoRA)

**Evaluated on:** 30 held-out Indian Legal test questions (`val.jsonl`)
**Hardware:** NVIDIA GeForce RTX 5060 Laptop GPU (4-bit NF4 Quantization)

## 📊 Benchmark Score Comparison

| Metric | Qwen3-VL-4B (Base) | Qwen3-VL-4B (Legal LoRA) | Better Model | Improvement / Difference |
| :--- | :---: | :---: | :---: | :---: |
| **Perplexity (PPL) ↓** | 3.47 | **2.04** | 🟢 **Legal LoRA** | **41.2% lower** (much sharper uncertainty reduction) |
| **Eval Cross-Entropy Loss ↓** | 1.2453 | **0.7132** | 🟢 **Legal LoRA** | **42.7% lower loss** |
| **ROUGE-1 (Unigram overlap) ↑** | 27.52% | **31.15%** | 🟢 **Legal LoRA** | **+13.2% relative gain** (+3.63 points) |
| **ROUGE-2 (Bigram legal phrasing) ↑** | 10.78% | **17.12%** | 🟢 **Legal LoRA** | **+58.8% relative gain** (+6.34 points) |
| **ROUGE-L (Sentence structure) ↑** | 18.18% | **22.33%** | 🟢 **Legal LoRA** | **+22.8% relative gain** (+4.15 points) |
| **BLEU-4 (Exact 4-gram match) ↑** | 2.68 | **5.34** | 🟢 **Legal LoRA** | **+99.3% relative gain (~2x better)** |
| **Legal Citation Recall ↑** | **66.92%** | 66.51% | ⚖️ **Parity** | -0.41 points (statistically identical) |
| **Peak GPU VRAM** | **5.14 GB** | 5.40 GB | 🟢 **Both fit GPU** | +0.26 GB (both well under 8 GB RTX 5060) |
| **Generation Speed (tok/s)** | **23.7 tok/s** | 10.1 tok/s | ⚪ Base Model | LoRA forward-path adds slight matrix overhead |


## 🔍 Side-by-Side Sample Comparison

### Sample 1:
**User Query:**
> Which state has a special provision with respect to Jammu and Kashmir?

**Gold Reference:**
> Jammu and Kashmir had a special provision with respect to itself under Article 370 of the Indian Constitution. This article granted Jammu and Kashmir a special autonomous status, allowing it to have its own constitution, a separate flag, and significant autonomy in governance, except in areas of def

**Base Model Output:**
> The state of **Jammu and Kashmir** has a **special provision** under **Article 370** of the Indian Constitution, which was in effect until **August 5, 2019**, when the Indian government revoked the article.

### What was Article 370?

Article 370 granted **Jammu and Kashmir** a **special status** un

**Fine-Tuned Legal Output:**
> The state that has a special provision with respect to Jammu and Kashmir is not specified in the provided context.

---

### Sample 2:
**User Query:**
> When was the Foreign Exchange Regulation Act passed and what is its act number?

**Gold Reference:**
> The Foreign Exchange Regulation Act was passed in 1973, and its act number is 12. This legislation, which aims to regulate the exchange of foreign currency, was enacted to ensure the stability and security of the country's financial system. The act number 12 signifies its importance and relevance in

**Base Model Output:**
> The **Foreign Exchange Regulation Act** was passed in **1973** and is **Act No. 10** of the Government of India.

It was enacted to regulate foreign exchange transactions and to ensure the stability of the Indian rupee and the orderly functioning of the foreign exchange market. The Act was passed by

**Fine-Tuned Legal Output:**
> The Foreign Exchange Regulation Act was passed in 1973 and its act number is 42.

---

### Sample 3:
**User Query:**
> What is the evidentiary value of a 'compromise decree' in subsequent proceedings under the Indian Evidence Act?

**Gold Reference:**
> The evidentiary value of a compromise decree under the Indian Evidence Act (IEA) is significant, particularly in specific legal contexts. Here's a structured overview:

1. **Admissibility as Documentary Evidence**: 
   - Under section 28 of the IEA, compromise decrees are admissible as documentary e

**Base Model Output:**
> Under the **Indian Evidence Act, 1872**, a **“compromise decree”** — which is a decree passed by a court in a civil suit, where the parties have agreed to settle the dispute for a sum or consideration, and the court has sanctioned that settlement — has **significant evidentiary value** in subsequent

**Fine-Tuned Legal Output:**
> A compromise decree, as a judicial instrument, holds significant evidentiary value in subsequent proceedings under the Indian Evidence Act. Here's a structured summary of its value and considerations:

1. **Definition and Nature**: A compromise decree is a formal agreement between parties to settle 

---

