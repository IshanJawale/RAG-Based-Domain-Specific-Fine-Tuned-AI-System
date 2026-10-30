"""
web_search.py

Tavily web search wrapper.

Converts Tavily search results into the same chunk-dict format used by
local retrieval, so the rest of the pipeline (reranker, generation)
can treat them uniformly.

Requires TAVILY_API_KEY in the .env file.
"""

import os

from dotenv import load_dotenv

load_dotenv()


TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

# Indian legal sources, boosted (not required) for legal questions so generic US pages stop winning
INDIAN_LEGAL_DOMAINS = [
    "indiankanoon.org", "indiacode.nic.in", "legislative.gov.in", "sci.gov.in", "livelaw.in",
    "barandbench.com", "scconline.com", "lawctopus.com", "ipleaders.in", "nalsa.gov.in",
]


# ------------------------------------------------------------------
# Lazy import — only fail at search time, not at import time
# ------------------------------------------------------------------

def _get_tavily_client():
    """
    Build and return a Tavily client.
    Raises a clear error if the key is missing or the package is absent.
    """

    if not TAVILY_API_KEY:
        raise EnvironmentError(
            "TAVILY_API_KEY not found in environment. "
            "Add it to your .env file to enable web search."
        )

    try:
        from tavily import TavilyClient
        return TavilyClient(api_key=TAVILY_API_KEY)
    except ImportError:
        raise ImportError(
            "tavily-python is not installed. "
            "Run: pip install tavily-python"
        )


# ------------------------------------------------------------------
# Main search function
# ------------------------------------------------------------------

class TavilySearcher:
    """
    Thin wrapper around the Tavily search API.

    Results are returned as chunk-like dicts so they can be passed
    directly to the reranker or generation module alongside local chunks.
    """

    def __init__(self):
        self._client = _get_tavily_client()

    def search(
        self,
        query:       str,
        max_results: int = 5,
        india:       bool = False,
    ) -> list[dict]:
        """
        Run a Tavily web search and return results as chunk dicts.

        india=True (legal questions) adds "India" to the query if missing, prefers results
        from India and boosts INDIAN_LEGAL_DOMAINS.

        Each result dict has:
          chunk_id       : "web_<i>"
          paper_id       : "web"
          paper_title    : page title (or URL)
          page           : "N/A"
          text           : snippet / content
          url            : source URL
          score          : Tavily relevance score (0–1)
          retrieval_type : "web"

        Returns an empty list on any error (don't crash the pipeline).
        """

        try:

            extra = {}
            if india:
                if "india" not in query.lower():
                    query = f"{query} India"
                extra = dict(country="india", include_domains=INDIAN_LEGAL_DOMAINS, include_domains_mode="boost")

            response = self._client.search(
                query=query,
                max_results=max_results,
                search_depth="basic",        # "advanced" costs more API credits
                include_answer=False,
                include_raw_content=False,
                **extra,
            )

            results = response.get("results", [])

            chunks = []

            for i, result in enumerate(results):

                content = result.get("content", "").strip()

                if not content:
                    continue

                chunks.append({
                    "chunk_id":      f"web_{i}",
                    "paper_id":      "web",
                    "paper_title":   result.get("title", result.get("url", "Web Result")),
                    "page":          "N/A",
                    "text":          content,
                    "url":           result.get("url", ""),
                    "score":         float(result.get("score", 0.0)),
                    "retrieval_type": "web",
                })

            return chunks

        except Exception as e:
            print(f"[WebSearch] Tavily search failed: {e}")
            return []
