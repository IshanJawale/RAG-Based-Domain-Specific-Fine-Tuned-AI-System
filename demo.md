# Adaptive Multimodal RAG — Research Paper Assistant
### Ishan Jawale · Plaksha University Capstone 2026

---

# Part 2 — What I Found

> Evaluation benchmark: **30 ground-truth questions** across 5 categories  
> Local Fact · Multi-Paper Comparison · Web/Temporal · Figure/Visual · Unanswerable

---

## Corpus Statistics

| Stat | Value |
|---|---|
| Papers ingested | 30 |
| Total chunks | 1,078 |
| Average words per chunk | 417 |
| Total figures extracted | 382 (97 bitmap + 285 vector-rendered) |
| Embedding dimensions | 384 (MiniLM-L6-v2) |

---

## Finding 1 — Chunking Strategy

Naive word-count splitting severs sentences mid-way, breaking formulae and citations across chunk boundaries. I switched to **NLTK sentence-aware chunking** with a 600-word target and a 100-word overlap.

The result is a clean, bell-shaped distribution of chunk lengths — no extreme outliers, no truncated equations, no split references.

![Chunk Distribution](notebooks/chunk_distribution.png)

---

## Finding 2 — Retrieval Evaluation

Four strategies were evaluated against 10 questions with a known ground-truth paper:

| Method | Recall@1 | Recall@5 | Recall@10 | MRR |
|---|---|---|---|---|
| BM25 | 60% | 80% | 90% | 0.698 |
| Dense (baseline) | 50% | 70% | 80% | 0.579 |
| Hybrid | 50% | 70% | 80% | 0.586 |
| **Hybrid + Reranker** | **50%** | **80%** | **90%** | **0.643** |

![Retrieval Comparison](notebooks/retrieval_comparison.png)

### The BM25 Paradox

BM25 scored almost identically to the full Hybrid + Reranker pipeline. This was an unexpected finding — and an important one.

Academic papers contain highly specific, rare acronyms (*LoRA*, *Self-RAG*, *RAPTOR*, *HyDE*). BM25's inverse document frequency weights rare tokens extremely heavily, so a query like *"What is RAPTOR?"* immediately surfaces the exact paper at Rank 1.

However, BM25 **completely fails** on conceptual queries like *"How do I reduce hallucinations without fine-tuning?"* — it scores near zero because none of the query words appear verbatim in the Self-RAG paper. The **Cross-Encoder Reranker** protects against this by understanding semantic meaning, not just word overlap. Both are needed.

---

## Finding 3 — Query Routing Accuracy

The LLM router was evaluated against all 30 benchmark questions:

- ✅ **100% routing accuracy** (30/30 correct)
- ✅ **0 unnecessary web searches** fired for local questions
- ✅ **0 missed web searches** for temporal or external queries

![Routing Accuracy](notebooks/routing_accuracy.png)

---

## Finding 4 — Vector Figure Extraction

Standard PDF libraries (`PyMuPDF.get_images()`) only capture **embedded raster bitmaps**. The majority of academic architecture diagrams are drawn as **vector graphics** — lines, boxes, and arrows — which are completely invisible to standard extractors.

I implemented a fallback renderer that:
1. Scans each page for `Figure X` caption patterns using regex
2. Renders the full page at 150 DPI using `page.get_pixmap()`
3. Applies a quality filter to reject blank pages and thin divider lines

**Result:** 285 vector diagrams extracted that would otherwise be lost.

**Example — CRAG Inference Architecture (Page 4):**

![CRAG Architecture Diagram](data/figures/Corrective%20Retrieval%20Augmented%20Generation_page_4_rendered_fig2.png)

---
---

# Part 3 — One Thing I'd Do Differently

---

## The Biggest Architectural Mistake

Figure retrieval in this system is **tightly coupled to text retrieval**.

The current flow is:
```
User Query → Hybrid Text Search → Top Chunks → Identify Paper → Find Figures from that Paper
```

This means the system can only find a figure *if the text retriever first finds a relevant passage from the same paper*. There is no independent visual search.

### The Bug This Caused

During testing, I asked: *"Tell me about the Mahabharata."*

The dense retriever was forced to return *some* chunks — it pulled a low-scoring passage from the GPT-3 paper. The figure retriever saw GPT-3 at Rank 1, and confidently served a GPT-3 few-shot learning diagram as a response to a question about an ancient epic.

I patched this with:
- **Cross-encoder negative score gating** — chunks with score < 0.0 do not contribute paper bonuses
- **Absolute figure score threshold** — figures scoring below 10 points are suppressed entirely

These are heuristics. They work, but they are not a principled solution.

---

## What I Would Do Differently

### ① Native Multimodal Figure Embedding (CLIP / ColPali)

During ingestion, embed each extracted figure image directly using a vision-language model like **CLIP** or **ColPali** into a dedicated image vector store. Query the image store separately in parallel with text retrieval.

```
User Query → [Text Vector Search]   → Top Text Chunks
           → [Image Vector Search]  → Top Figures (semantically matched)
                                         ↓
                               Fuse and send to LLM
```

This completely decouples figure retrieval from text retrieval. A question about *"the attention heatmap in Transformer papers"* would directly match the diagram visually — no text context needed.

---

### ② GraphRAG for Cross-Paper Synthesis

Right now, multi-paper comparison queries retrieve chunks independently and ask the LLM to synthesise. This works for simple comparisons but breaks down when relationships between papers are subtle.

A **GraphRAG** layer would build an entity-relationship graph at ingestion time — linking papers by shared concepts, citations, and contradicting claims — enabling true relational traversal across the corpus.

---

### ③ Fully Quantized Edge Deployment

The Ollama toggle I implemented already allows running `llama3.2-vision` locally. But on a laptop CPU, multimodal inference currently takes 15–25 seconds per response.

Using **llama.cpp** with 4-bit quantization, the same model could realistically run at under 5 seconds on a consumer laptop — making the entire multimodal pipeline fully air-gapped, privacy-preserving, and deployable in academic or enterprise settings where sending research documents to external APIs is not permitted.
