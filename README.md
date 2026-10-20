# Adaptive Multimodal RAG for Research Papers

## What is this project?

This project is an advanced, multimodal Retrieval-Augmented Generation (RAG) system designed to answer complex questions about AI/ML research papers. 

Unlike basic RAG systems that blindly fetch text from a single source, this system is **adaptive and multimodal**:
- **Adaptive Routing & Web Fallback**: It uses an LLM router to classify user questions. It intelligently decides whether to search the local research corpus, query the live web (e.g., for recent news or pricing), or both. If local retrieval yields low-confidence results, it automatically falls back to web search.
- **Multimodal Generation**: It extracts both text and figures from PDFs. When a user asks a question, it retrieves the most relevant text chunks, identifies the source paper, and then fetches the most relevant diagrams/figures from that specific paper to provide the figure caption as extra context to a local Qwen3-8B model for generating the final answer.

## Technology Stack (What is used, where, and why)

| Component | Tool / Library | Why it was chosen |
|---|---|---|
| **PDF Processing & Figure Extraction** | `PyMuPDF` (fitz) | Used in `src/ingestion.py` for text extraction and rendering vector/raster graphics from PDFs. It is extremely fast and allows precise page-region rendering. |
| **Chunking** | `NLTK` | Used in `src/chunking.py` for sentence-aware chunking to ensure text is split cleanly at sentence boundaries rather than cutting words in half. |
| **Dense Retrieval** | `sentence-transformers` & `numpy` | Used in `src/retrieval.py` to embed chunks and perform fast cosine-similarity vector search (NumPy is sufficient for a 30-paper corpus). |
| **Sparse Retrieval** | `rank-bm25` | Used in `src/retrieval.py` for exact keyword matching, running in parallel with the dense retriever (Hybrid Search). |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Used in `src/reranking.py` to re-score the top hybrid retrieval results, significantly improving the precision of the retrieved context. |
| **Web Search** | `Tavily` | Used in `src/web_search.py` as an external knowledge fallback for questions outside the corpus scope (like current events). |
| **Language Model** | `Qwen3-8B` (local, via `transformers`) | Loaded from `./Qwen3-8B` by `src/llm.py`; used in `src/routing.py` and `src/generation.py` for classifying user intent and generating the final answer. Runs fully offline. |
| **Web Interface** | `streamlit` | Used in `app/app.py` to provide a modern, conversational chatbot interface with persistent chat history. |

## How to Run It

### 1. Setup the Environment
Ensure you have Python 3.10+ installed. Activate your virtual environment and install the dependencies:
```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set API Keys
Create a `.env` file in the root directory and add your API keys:
```ini
TAVILY_API_KEY="your-tavily-api-key"
```

### 3. Ingest the Corpus
Before running the system, you must ingest the PDFs (located in the `papers/` folder) to build the text chunks, BM25 index, dense embeddings, and extract the figures.
```powershell
python ingest.py
```
*Note: This will take a few minutes. It will save the artifacts to the `data/` directory.*

### 4. Start the Chatbot UI
Once ingestion is complete, launch the Streamlit web interface:
```powershell
streamlit run app/app.py
```

## Evaluation
A comprehensive evaluation of the pipeline's performance (Retrieval Recall/MRR, Routing Accuracy, Chunking Distributions, and Multimodal capability) is available in the Jupyter Notebook:
```powershell
jupyter notebook notebooks/evaluation.ipynb
```
