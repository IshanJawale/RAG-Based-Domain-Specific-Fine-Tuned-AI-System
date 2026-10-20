"""
credibility.py

Citation-based credibility scoring for retrieved chunks.

Credibility is computed from the number of times a paper has been
cited (via OpenAlex, stored in data/citations.json).

Per-source score (0–100):
    score = log10(citations + 1) / log10(MAX_CITATIONS + 1) * 100

Aggregate score for a response:
    Weighted average over the top-5 chunks, weighted by rerank_score
    (or uniform weight if scores are absent / all equal).

Star rating (for display):
    5 stars  >= 80
    4 stars  >= 60
    3 stars  >= 40
    2 stars  >= 20
    1 star   >= 0
"""

import json
import math
from pathlib import Path

# Practical upper-bound for log normalisation.
# Attention Is All You Need ~7 300 at time of writing.
_MAX_CITATIONS = 10_000

_CITATIONS_PATH = Path("data/citations.json")

# ── cache ─────────────────────────────────────────────────────────────────────

_CITATION_DB: dict | None = None


def load_citations(path: Path = _CITATIONS_PATH) -> dict:
    """Load (and cache) the citations.json lookup table."""
    global _CITATION_DB
    if _CITATION_DB is None:
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                _CITATION_DB = json.load(f)
        else:
            _CITATION_DB = {}
    return _CITATION_DB


# ── helpers ───────────────────────────────────────────────────────────────────

def _normalize(citation_count: int | None) -> float:
    """Map raw citation count → 0–100 using log scaling."""
    if citation_count is None or citation_count <= 0:
        return 0.0
    return (math.log10(citation_count + 1) / math.log10(_MAX_CITATIONS + 1)) * 100.0


def _stars(score: float) -> str:
    """Return a star string for display."""
    n = int(score // 20)           # 0–5
    n = min(max(n, 0), 5)
    return "★" * n + "☆" * (5 - n)


def _label(score: float) -> str:
    if score >= 80:
        return "Highly cited"
    if score >= 60:
        return "Well cited"
    if score >= 40:
        return "Moderately cited"
    if score >= 20:
        return "Low citations"
    return "Uncited / not found"


# ── public API ─────────────────────────────────────────────────────────────────

def get_paper_credibility(paper_title: str, citations: dict) -> dict:
    """
    Return credibility info for a single paper.

    Parameters
    ----------
    paper_title : chunk['paper_title']
    citations   : dict loaded by load_citations()

    Returns
    -------
    {
        "citation_count" : int | None,
        "year"           : int | None,
        "score"          : float,   # 0–100
        "stars"          : str,     # e.g. "★★★☆☆"
        "label"          : str,     # human-readable tier
        "found"          : bool,
    }
    """
    entry = citations.get(paper_title)

    if not entry or entry.get("match_status") not in ("matched",):
        return {
            "citation_count": None,
            "year": None,
            "score": 0.0,
            "stars": _stars(0),
            "label": "Not in citation database",
            "found": False,
        }

    count = entry.get("citation_count")
    year  = entry.get("year")
    score = _normalize(count)

    return {
        "citation_count": count,
        "year":           year,
        "score":          round(score, 1),
        "stars":          _stars(score),
        "label":          _label(score),
        "found":          True,
    }


def aggregate_credibility(chunks: list[dict], citations: dict) -> dict:
    """
    Compute a weighted aggregate credibility score for a list of chunks.

    Only local (non-web) chunks are scored; web sources are excluded.
    Weights are the rerank_scores (or equal weights if not present).

    Returns
    -------
    {
        "score"       : float,    # 0–100
        "stars"       : str,
        "label"       : str,
        "n_scored"    : int,      # how many local chunks were scored
        "n_web"       : int,      # how many were skipped (web sources)
    }
    """
    scored_pairs: list[tuple[float, float]] = []   # (weight, score)
    n_web = 0

    for chunk in chunks:
        if chunk.get("retrieval_type") == "web":
            n_web += 1
            continue

        title  = chunk.get("paper_title", "")
        cred   = get_paper_credibility(title, citations)
        weight = max(chunk.get("rerank_score", 1.0), 0.01)   # always positive
        scored_pairs.append((weight, cred["score"]))

    if not scored_pairs:
        return {
            "score":    0.0,
            "stars":    _stars(0),
            "label":    "No local sources",
            "n_scored": 0,
            "n_web":    n_web,
        }

    total_weight = sum(w for w, _ in scored_pairs)
    agg = sum(w * s for w, s in scored_pairs) / total_weight

    return {
        "score":    round(agg, 1),
        "stars":    _stars(agg),
        "label":    _label(agg),
        "n_scored": len(scored_pairs),
        "n_web":    n_web,
    }
