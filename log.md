# Project Log Report: Adaptive Multimodal RAG

## What I Did
- I developed a complete Multimodal RAG pipeline designed to answer complex questions over a local corpus of 30 AI/ML research papers.
- I implemented a robust **Hybrid Retrieval** system that combines dense vector search (using `sentence-transformers` and NumPy) with sparse lexical search (`rank-bm25`) to maximize recall. I then added an MS-MARCO Cross-Encoder to strictly rerank the top results for high precision.
- I built an **LLM Router** (using Gemini 3.5 Flash) that analyzes the user's question upfront to decide the optimal data source. It intelligently routes queries to the local corpus, a live web search (via Tavily), or both.
- I designed a custom **Multimodal Figure Retrieval** system. The system extracts diagrams from PDFs and matches them to the user's query using the context of the retrieved text chunks. If a relevant diagram is found, it is fed directly into Gemini alongside the text for true multimodal generation.
- I deployed the entire system into a modern, conversational **Streamlit chatbot interface** with persistent message history.
- I established a rigorous **Jupyter Notebook evaluation suite** to systematically test and plot retrieval metrics (Recall@K, MRR), chunking distributions, routing accuracy, and answer quality.

## Bottlenecks, Challenges & Solutions

### 1. Missing Vector-Based Diagrams
**The Problem & Why it was an issue:** 
My initial PDF extraction script relied solely on `PyMuPDF.get_images()`, which only captures embedded raster images (like JPEGs or PNGs). However, many crucial architecture diagrams in modern academic papers (like the core CRAG architecture diagram) are drawn natively within the PDF using vector commands (lines, boxes, text). Because they aren't traditional image files, my code was completely blind to them, preventing the multimodal LLM from accessing the most important visual evidence.

**How I Solved It:** 
I engineered a "hybrid extraction" method in `src/ingestion.py`. While still extracting standard rasters, I added a page-rendering mode that scans the PDF text for "Figure X" captions. When found, the script calculates the bounding box of that region and takes a high-resolution, cropped screenshot (`get_pixmap()`), effectively converting vector graphics into accessible images. To prevent extracting useless lines or borders, I applied Pillow and NumPy statistical filters to reject images that are >93% solid color or have extreme aspect ratios.

### 2. Unreliable LLM Modality Routing
**The Problem & Why it was an issue:** 
Initially, I designed the LLM router to explicitly classify every user query into strict modalities: "TEXT" or "IMAGE". This proved to be highly unreliable. Users don't always ask "show me a diagram"; they often ask conceptually (e.g., "explain this architecture"). Strict routing caused the system to frequently bypass the figure retrieval pipeline altogether, resulting in degraded, text-only answers when a helpful diagram was available.

**How I Solved It:** 
I completely stripped out the modality decision from the LLM router. I shifted the system to an "Always Fetch Both" pipeline. Now, the system searches for relevant text and corresponding figures in parallel for *every* local query. If a relevant figure is found, it automatically upgrades to a multimodal generation call. If no figure is found, it gracefully falls back to a standard text-only LLM call. Let the LLM decide if the visual context is useful, rather than trying to guess upfront.

### 3. "Hallucinated" Figure Retrieval on Off-Topic Queries
**The Problem & Why it was an issue:** 
If a user asked a completely off-topic question (like "tell me about the Mahabharat"), the dense/BM25 text retriever is still forced to return *some* chunks, even if the similarity score is practically zero. My figure retrieval logic was designed to prioritize figures from the same papers as the top text chunks. Consequently, the system would confidently fetch and display a random, completely irrelevant diagram from whatever paper happened to rank #1 for that nonsense query.

**How I Solved It:** 
I introduced score-gating in `src/figures.py`. The figure retriever now inspects the cross-encoder `rerank_score` of the retrieved text chunks before utilizing them. If the top chunks have deeply negative scores (indicating they are mathematical irrelevancies), the system ignores their paper affiliations. Furthermore, I added an absolute minimum relevance score threshold (`score >= 10`) to the figure matcher. If neither the text context nor the caption overlap is strong enough, the system successfully outputs "No relevant figure found."

### 4. Silent Dependency Failures & Windows Terminal Encoding
**The Problem & Why it was an issue:** 
During testing, the Streamlit app would crash with a `ModuleNotFoundError` for `torchvision`, even though I wasn't explicitly using it. This occurred because Streamlit's file watcher triggered `transformers`, which attempted to import `torchvision` for certain image processing models under the hood. Additionally, running the ingestion script in the standard Windows terminal caused fatal crashes (`UnicodeEncodeError`) simply because the terminal couldn't render the fancy box-drawing characters used in my print statements.

**How I Solved It:** 
I explicitly added `torchvision` to `requirements.txt` and the virtual environment to satisfy the internal dependencies of `sentence-transformers`. For the encoding issues, I replaced all extended Unicode characters in `ingest.py` with standard ASCII characters (`-` and `<-`). This guaranteed flawless execution across all environments without sacrificing the UI layout.

## What More Can Be Done
- **CLIP Image Embeddings**: Currently, figure retrieval relies on lexical matching between the user's text query, the figure's caption, and its parent paper's title. I could embed the actual images themselves using an open-source vision-language model (like OpenAI's CLIP) during ingestion, allowing for true semantic visual search.
- **GraphRAG Integration**: For questions requiring synthesis across multiple papers (e.g., "Compare the chunking strategies across all survey papers"), building a Knowledge Graph of citations, authors, and methodologies would vastly improve the system's reasoning capabilities over standard vector similarity.
- **Agentic Web Search**: Instead of a simple one-shot Tavily query for web fallback, I could implement an agentic loop where the LLM evaluates the initial web search results, decides if it has enough context, and issues follow-up, refined queries autonomously before finally answering the user.
