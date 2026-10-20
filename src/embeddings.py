import numpy as np

from sentence_transformers import SentenceTransformer
from pathlib import Path


DATA_DIR = Path("data")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def create_embeddings(chunks):

    print("Loading embedding model...")

    model = SentenceTransformer(MODEL_NAME)

    texts = [chunk["text"] for chunk in chunks]

    print(f"Generating embeddings for {len(texts)} chunks...")

    embeddings = model.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    embeddings = np.asarray(embeddings, dtype=np.float32)

    output_path = DATA_DIR / "embeddings.npy"

    np.save(output_path, embeddings)

    print("--------------------------------")
    print("Embedding generation complete")
    print(f"Embedding shape: {embeddings.shape}")
    print(f"Saved to: {output_path}")
    print("--------------------------------\n")

    return embeddings