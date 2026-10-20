"""
figures.py

Figure retrieval — paper-aware, size-aware.

Core insight: the text retriever already found the right *paper*.
The figure retriever should look for figures from *that same paper* first,
then fall back to global search only if nothing is found.

Scoring:
  1. Paper match  — large bonus if figure is from the same paper as top text chunks
  2. Token overlap — caption / paper_id / paper_title vs. query tokens
  3. Size bonus   — larger images (width × height) are more likely to be diagrams
  4. Figure number — if query mentions "figure 2", boost exact figure-number matches
"""

import json
import re
from pathlib import Path


FIGURES_JSON = Path("data/figures.json")


# -----------------------------------------------------------------------
# Load
# -----------------------------------------------------------------------

def load_figures(path: Path = FIGURES_JSON) -> list[dict]:
    """Load figure metadata from disk. Returns [] if file missing."""
    if not path.exists():
        print(f"[Figures] figures.json not found at {path}.")
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# -----------------------------------------------------------------------
# Paper context from retrieved text chunks
# -----------------------------------------------------------------------

def _papers_from_chunks(retrieved_chunks: list[dict]) -> list[str]:
    """
    Return paper_ids ordered by how many top chunks they contributed.
    The top-ranked chunk counts most.
    """
    paper_counts: dict[str, float] = {}

    for rank, chunk in enumerate(retrieved_chunks):
        paper_id = chunk.get("paper_id", "")
        if not paper_id or paper_id == "web":
            continue
            
        # If reranker score is available and it's negative, the chunk is irrelevant.
        # Don't use it to pull in figures.
        if chunk.get("rerank_score", 1.0) < 0.0:
            continue

        # Weight by inverse rank (rank 0 → weight 1.0, rank 4 → weight 0.2)
        weight = 1.0 / (rank + 1)
        paper_counts[paper_id] = paper_counts.get(paper_id, 0.0) + weight

    # Sort by cumulative weight descending
    return sorted(paper_counts, key=lambda p: paper_counts[p], reverse=True)


# -----------------------------------------------------------------------
# Figure number extraction
# -----------------------------------------------------------------------

_FIGURE_NUM_RE = re.compile(r"\bfig(?:ure)?\.?\s*(\d+)", re.IGNORECASE)


def _query_figure_numbers(query: str) -> set[int]:
    """Extract explicit figure numbers from the query, e.g. 'figure 2' → {2}."""
    return {int(m.group(1)) for m in _FIGURE_NUM_RE.finditer(query)}


def _figure_index_from_id(figure_id: str) -> int | None:
    """Extract the figure index from a figure_id like 'paper_page_3_figure_2' → 2."""
    m = re.search(r"_figure_(\d+)$", figure_id)
    return int(m.group(1)) if m else None


# -----------------------------------------------------------------------
# Main retrieval function
# -----------------------------------------------------------------------

def find_figures_for_chunks(
    question:         str,
    retrieved_chunks: list[dict],
    figures:          list[dict],
    top_k:            int = 1,
) -> list[dict]:
    """
    Find the most relevant figures, strongly biased toward the papers
    that the text retriever already identified as relevant.

    Parameters
    ----------
    question         : user query string
    retrieved_chunks : top text chunks from hybrid + reranking (ordered best-first)
    figures          : all figure metadata dicts from figures.json
    top_k            : number of figures to return

    Returns
    -------
    list of figure metadata dicts, sorted by relevance score descending
    """

    if not figures:
        return []

    # --- Build scoring context ---
    q_lower        = question.lower()
    q_tokens       = set(re.findall(r"\w+", q_lower))
    q_fig_numbers  = _query_figure_numbers(question)
    top_papers     = _papers_from_chunks(retrieved_chunks)  # ordered by relevance
    top_paper_set  = set(top_papers)

    # Maximum pixel area among all figures (for normalised size bonus)
    max_area = max(
        (f.get("width", 0) * f.get("height", 0) for f in figures),
        default=1,
    ) or 1

    scored = []

    for fig in figures:

        score     = 0.0
        paper_id  = fig.get("paper_id",    "")
        caption   = fig.get("caption",     "").lower()
        p_title   = fig.get("paper_title", "").lower()
        width     = fig.get("width",  0)
        height    = fig.get("height", 0)
        fig_id    = fig.get("figure_id",   "")

        # 1. Paper match — the single most important signal
        if paper_id in top_paper_set:
            paper_rank = top_papers.index(paper_id)
            # Top paper → +30, second → +20, third → +15, etc.
            score += max(30 - paper_rank * 5, 10)

        # 2. Token overlap on caption + paper title
        blob     = f"{caption} {p_title}"
        b_tokens = set(re.findall(r"\w+", blob))
        overlap  = len(q_tokens & b_tokens)
        score   += overlap * 1.5

        # 3. Explicit figure number match
        fig_index = _figure_index_from_id(fig_id)
        if fig_index is not None and fig_index in q_fig_numbers:
            score += 15

        # 4. Size bonus — prefer bigger images (more likely to be real diagrams)
        area       = width * height
        size_bonus = (area / max_area) * 5   # 0–5 points
        score     += size_bonus

        # 5. Hard skip: if no paper match at all AND no token overlap,
        #    give a heavy penalty so globally wrong figures stay at the bottom
        if paper_id not in top_paper_set and overlap == 0:
            score -= 50

        scored.append((score, fig))

    scored.sort(key=lambda x: x[0], reverse=True)

    # Only return figures that passed a minimum relevance threshold
    return [fig for score, fig in scored[:top_k] if score >= 10]


# -----------------------------------------------------------------------
# Legacy helper (for cases where no chunks context is available)
# -----------------------------------------------------------------------

def find_figures_for_question(
    question:  str,
    figures:   list[dict],
    top_k:     int = 3,
) -> list[dict]:
    """
    Fallback: find figures by keyword matching only (no chunk context).
    Use find_figures_for_chunks() when retrieved chunks are available.
    """
    if not figures:
        return []

    q_lower   = question.lower()
    q_tokens  = set(re.findall(r"\w+", q_lower))
    q_fig_nums = _query_figure_numbers(question)
    max_area   = max(
        (f.get("width", 0) * f.get("height", 0) for f in figures), default=1
    ) or 1

    scored = []
    for fig in figures:
        caption  = fig.get("caption",     "").lower()
        p_title  = fig.get("paper_title", "").lower()
        fig_id   = fig.get("figure_id",   "")
        width    = fig.get("width",  0)
        height   = fig.get("height", 0)

        blob     = f"{caption} {p_title}"
        b_tokens = set(re.findall(r"\w+", blob))
        score    = len(q_tokens & b_tokens) * 1.5

        fig_index = _figure_index_from_id(fig_id)
        if fig_index is not None and fig_index in q_fig_nums:
            score += 15

        score += ((width * height) / max_area) * 5
        scored.append((score, fig))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [fig for score, fig in scored[:top_k] if score >= 10]


# -----------------------------------------------------------------------
# Path helper
# -----------------------------------------------------------------------

def get_figure_path(fig_meta: dict) -> Path | None:
    """Return the Path to a figure image, or None if file does not exist."""
    path_str = fig_meta.get("path", "")
    if not path_str:
        return None
    p = Path(path_str)
    return p if p.exists() else None
