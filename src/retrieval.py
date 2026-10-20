"""
retrieval.py

Provides three retrieval strategies:
  1. DenseRetriever  — cosine similarity over sentence-transformer embeddings
  2. BM25Retriever   — sparse BM25 keyword retrieval (rank-bm25)
  3. HybridRetriever — union of Dense + BM25 results, deduplicated

The typical usage pattern is:
    dense   = DenseRetriever(embeddings, chunks)
    bm25    = BM25Retriever(chunks)           # or load from disk
    hybrid  = HybridRetriever(dense, bm25)
    results = hybrid.search(query, top_k=20)  # feed to reranker
"""

import lancedb
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModel

DB_PATH = Path("data/lancedb")

class LegalRetriever:
    """
    LanceDB-backed retriever for Legal Acts and Supreme Court cases.
    We merge all tables to query simultaneously.
    """

    def __init__(self):
        print("Loading LanceDB and Embedding model for retrieval...")
        self.db = lancedb.connect(DB_PATH)
        
        self.tables = []
        available_tables = self.db.table_names() if hasattr(self.db, "table_names") else self.db.list_tables()
        if "acts" in available_tables:
            self.tables.append(self.db.open_table("acts"))
        if "supreme_court" in available_tables:
            self.tables.append(self.db.open_table("supreme_court"))
        if "high_court" in available_tables:
            self.tables.append(self.db.open_table("high_court"))
            
        self.tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)
        self.model = AutoModel.from_pretrained('BAAI/bge-small-en-v1.5', local_files_only=True)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        self.model.eval()
        
    def _get_query_embedding(self, query: str):
        encoded_input = self.tokenizer(query, padding=True, truncation=True, return_tensors='pt').to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            emb = model_output[0][:, 0]
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.cpu().numpy().tolist()[0]

    def search(self, query: str, top_k: int = 20, per_table: int | None = None) -> list[dict]:
        """
        Query all available LanceDB tables and combine results.

        Each corpus contributes its own top `per_table` hits (default ceil(top_k / 2)),
        so the 5M High Court chunks cannot crowd out the Acts and Supreme Court
        passages. The merged pool (up to per_table x n_tables candidates, sorted by
        similarity) is returned for the cross-encoder to pick the best from.
        """
        query_vector = self._get_query_embedding(query)
        per_table = per_table or max(5, -(-top_k // 2))

        results = []

        # Search each table separately
        for table in self.tables:
            table_results = table.search(query_vector).limit(per_table).to_list()
            for r in table_results:
                r["score"] = r.get("_distance", 0.0)  # LanceDB returns _distance
                r["retrieval_type"] = "dense"
                # Standardise fields for pipeline
                r["paper_title"] = r.get("act_title", r.get("case_id", "Unknown Document"))
                r["page"] = r.get("section_number", "N/A")
                results.append(r)

        # Sort by distance (lower distance = higher similarity in vector space)
        # Note: L2 distance is smaller for similar vectors.
        results.sort(key=lambda x: x["score"])

        # Invert score so higher is better for the reranker if it expects it
        for r in results:
            r["score"] = 2.0 - r["score"]

        return results

    def search_acts(self, query: str, top_k: int = 5) -> list[dict]:
        """Query only statutory bare acts (IPC, CrPC, CPC, BNS, Companies Act, etc.)."""
        query_vector = self._get_query_embedding(query)
        try:
            table = self.db.open_table("acts")
            raw_hits = table.search(query_vector).limit(top_k).to_list()
            results = []
            for r in raw_hits:
                r["score"] = 2.0 - r.get("_distance", 0.0)
                r["retrieval_type"] = "acts"
                r["paper_title"] = r.get("act_title", "Central/State Act")
                r["page"] = r.get("section_number", "Section N/A")
                results.append(r)
            return results
        except Exception:
            return []

    def search_cases(self, query: str, top_k: int = 5, court_type: str = "all") -> list[dict]:
        """Query court judgments (Supreme Court and/or High Court)."""
        query_vector = self._get_query_embedding(query)
        tables_to_query = []
        if court_type in ("all", "supreme_court"):
            try: tables_to_query.append(self.db.open_table("supreme_court"))
            except Exception: pass
        if court_type in ("all", "high_court"):
            try: tables_to_query.append(self.db.open_table("high_court"))
            except Exception: pass

        results = []
        per_t = max(3, top_k)
        for t in tables_to_query:
            try:
                hits = t.search(query_vector).limit(per_t).to_list()
                for r in hits:
                    r["score"] = 2.0 - r.get("_distance", 0.0)
                    r["retrieval_type"] = "precedent"
                    r["paper_title"] = r.get("case_id", "Court Judgment")
                    r["page"] = r.get("court", "Court Precedent")
                    results.append(r)
            except Exception:
                continue

        results.sort(key=lambda x: x.get("score", 0.0), reverse=True)
        return results[:top_k]

# Monkey-patch dummy classes so pipeline.py doesn't crash before we update it
class DenseRetriever: pass
class BM25Retriever:
    @classmethod
    def load(cls, *args, **kwargs): return None
class HybridRetriever:
    def __init__(self, *args, **kwargs):
        self.retriever = LegalRetriever()
    def search(self, query, top_k=20):
        return self.retriever.search(query, top_k)
    def search_acts(self, query, top_k=5):
        return self.retriever.search_acts(query, top_k)
    def search_cases(self, query, top_k=5, court_type="all"):
        return self.retriever.search_cases(query, top_k, court_type=court_type)

def print_results(query: str, results: list[dict]):
    print("\n" + "=" * 80)
    print("QUERY:\n" + query)
    print("=" * 80 + "\nRETRIEVED CHUNKS:\n" + "=" * 80)
    for rank, result in enumerate(results, start=1):
        print(f"\n--- Result {rank} ---")
        print(f"Score          : {result.get('score', 0):.4f}")
        print(f"Source         : {result.get('source_type', 'unknown')}")
        print(f"Document       : {result.get('paper_title', '')}")
        print(f"Section/Page   : {result.get('page', '')}")
        print()
        preview = result.get("text", "")[:400]
        print(preview + ("..." if len(result.get("text", "")) > 400 else ""))