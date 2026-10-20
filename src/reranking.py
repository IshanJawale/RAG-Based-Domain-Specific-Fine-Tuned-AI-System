"""
reranking.py

Cross-Encoder reranker.

Takes a list of candidate chunks from hybrid retrieval (~20–40 items) and
re-scores each (question, chunk_text) pair using a lightweight cross-encoder
model.  Returns the top_k most relevant chunks in descending score order.

Model used: cross-encoder/ms-marco-MiniLM-L-6-v2
  - ~22 MB, runs on CPU in a few seconds for 40 candidates
  - Trained on MS MARCO passage ranking — good general relevance
"""

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

class Reranker:
    """
    Lightweight cross-encoder reranker.
    Usage:
        reranker = Reranker()
        top5 = reranker.rerank(query, candidates, top_k=5)
    """

    def __init__(self):
        print(f"Loading cross-encoder: {CROSS_ENCODER_MODEL} ...")
        # local_files_only=False first time to download it if missing, then it caches
        self.tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_MODEL)
        self.model = AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_MODEL)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        print("Cross-encoder loaded.")

    def rerank(
        self,
        query:      str,
        candidates: list[dict],
        top_k:      int = 5,
    ) -> list[dict]:
        """
        Score every (query, candidate_text) pair and return the
        top_k highest-scoring candidates.
        """

        if not candidates:
            return []

        pairs = [
            [query, chunk["text"]]
            for chunk in candidates
        ]

        scored = []
        batch_size = 16
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i+batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                # Cross-encoder typically returns 1 logit per pair
                logits = outputs.logits.squeeze(-1).cpu().numpy().tolist()

            # Handle case where batch size is 1 and squeeze removed sequence dim
            if isinstance(logits, float):
                logits = [logits]

            for j, score in enumerate(logits):
                chunk = candidates[i+j].copy()
                chunk["rerank_score"] = float(score)
                scored.append(chunk)

        # Sort by rerank_score descending
        scored.sort(key=lambda x: x["rerank_score"], reverse=True)

        return scored[:top_k]
