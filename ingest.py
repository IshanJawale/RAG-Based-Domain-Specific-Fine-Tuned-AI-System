"""
ingest.py

One-shot corpus ingestion script.

Run this once (or whenever the papers/ directory changes):
    python ingest.py

What it does:
  1. Extract text + figures from all PDFs in papers/
  2. Chunk text with sentence-aware chunking
  3. Generate sentence-transformer embeddings → data/embeddings.npy
  4. Build BM25 index → data/bm25.pkl
  5. Save chunk metadata → data/chunks.json
  6. Save figure metadata → data/figures.json

After this runs, main.py loads the saved artifacts and is ready
without any re-processing.
"""

from src.ingestion import ingest_corpus
from src.chunking import create_chunks
from src.embeddings import create_embeddings
from src.retrieval import BM25Retriever


def main():

    print("\n")
    print("=" * 80)
    print("ADAPTIVE MULTIMODAL RAG — CORPUS INGESTION")
    print("=" * 80)

    # ------------------------------------------------------------------
    # STEP 1: Extract text + figures from PDFs
    # ------------------------------------------------------------------

    print("\n[1/4] INGESTING PAPERS")

    pages, figures = ingest_corpus()

    # ------------------------------------------------------------------
    # STEP 2: Sentence-aware chunking
    # ------------------------------------------------------------------

    print("\n[2/4] CREATING CHUNKS (sentence-aware)")

    chunks = create_chunks(
        pages,
        chunk_size=600,
        overlap=100,
    )

    # ------------------------------------------------------------------
    # STEP 3: Dense embeddings
    # ------------------------------------------------------------------

    print("\n[3/4] GENERATING EMBEDDINGS")

    embeddings = create_embeddings(chunks)

    # ------------------------------------------------------------------
    # STEP 4: BM25 index
    # ------------------------------------------------------------------

    print("\n[4/4] BUILDING BM25 INDEX")

    bm25_retriever = BM25Retriever(chunks)
    bm25_retriever.save()

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    print("\n")
    print("=" * 80)
    print("INGESTION COMPLETE")
    print("=" * 80)

    print(f"\nPapers processed  : {len(set(p['paper_id'] for p in pages))}")
    print(f"Pages extracted   : {len(pages)}")
    print(f"Figures extracted : {len(figures)}")
    print(f"Chunks created    : {len(chunks)}")
    print(f"Embedding shape   : {embeddings.shape}")

    print("\nSaved artifacts:")
    print("  data/chunks.json")
    print("  data/figures.json")
    print("  data/embeddings.npy")
    print("  data/bm25.pkl")
    print("  data/figures/")

    print("\nYou can now run:")
    print("  python main.py          <- interactive Q&A")
    print("  streamlit run app/app.py <- Streamlit UI")
    print("\n")


if __name__ == "__main__":
    main()
