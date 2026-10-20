"""
pipeline.py

Adaptive legal RAG pipeline shared by the Streamlit app and the CLI (main.py).

Per query:
  1. Route intent (src/routing.py) -> LOCAL | BOTH | WEB
  2. Retrieve + cross-encoder rerank, then pick the evidence mode from the scores:

       route  | local evidence (top rerank score)   | evidence used
       -------+-------------------------------------+---------------------------------------
       WEB    | not searched                        | web only (local corpus if web is down)
       BOTH   | any                                 | hybrid: relevant local + web
       LOCAL  | strong   (>= LOCAL_STRONG)          | local only
       LOCAL  | partial  (IRRELEVANT .. STRONG)     | hybrid: relevant local + web
       LOCAL  | none     (<  LOCAL_IRRELEVANT)      | web only

     If web search is unavailable or returns nothing, whatever local evidence
     exists is used instead of failing.
  3. Answer: drafting / calculation requests go to the tool-using agent;
     everything else gets a grounded answer from the retrieved evidence.

Cross-encoder (ms-marco-MiniLM-L-6-v2) logits measured on this corpus:
covered legal questions score 3-8, partially covered ones -1..3, off-topic < -2.
"""

import re
from typing import Callable, Optional

from src.routing import classify_query
from src.generation import generate_answer
from src.agent import run_legal_agent, is_drafting_request, is_calculation_request
from src import tools

LOCAL_STRONG     = 3.0   # top local score at/above this -> local evidence alone is enough
LOCAL_IRRELEVANT = -2.0  # top local score below this -> local corpus has nothing useful
CHUNK_MIN_SCORE  = 0.0   # individual local chunks below this are treated as noise
WEB_RESULTS      = 5
MIN_WEB_SLOTS    = 2     # hybrid answers always keep at least this many web passages
MAX_CONTEXT      = 7     # passages handed to the LLM

MODE_LABELS = {"local": "Local corpus only", "hybrid": "Hybrid (local + web)", "web": "Web only", "none": "No evidence"}


def _source_card(chunk: dict) -> dict:
    stype = chunk.get("source_type", "legal_corpus").upper()
    is_web = chunk.get("retrieval_type") == "web"
    kind = "web" if is_web else ("court" if "COURT" in stype else "act")
    text = chunk.get("text", "")
    return {
        "kind": kind,
        "type": "WEB" if is_web else stype,
        "title": chunk.get("act_title") or chunk.get("case_id") or chunk.get("paper_title", "Document"),
        "section": chunk.get("url") if is_web else str(chunk.get("section_number") or chunk.get("court") or chunk.get("page", "N/A")),
        "url": chunk.get("url"),
        "score": chunk.get("rerank_score", chunk.get("score", 0.0)),
        "scored_by": "rerank" if "rerank_score" in chunk else "distance",
        "text": text[:400] + ("..." if len(text) > 400 else ""),
    }


_DRAFT_PREFIX = re.compile(
    r"^\s*(?:please\s+)?(?:can\s+you\s+)?(?:draft|prepare|generate|create|write|compose|make)\s+"
    r"(?:me\s+)?(?:an?\s+|the\s+)?(?:formal\s+)?(?:legal\s+)?"
    r"(?:notice|opinion|case\s+brief|brief|report|assessment\s+report|document|memo)?\s*"
    r"(?:for|on|about|regarding|in\s+respect\s+of)?\s*",
    re.IGNORECASE,
)


def retrieval_query(question: str) -> str:
    """Search text for retrieval: drop the drafting instruction so the embedding captures the legal substance."""
    stripped = _DRAFT_PREFIX.sub("", question).strip()
    return stripped if len(stripped) >= 12 else question


def _merge_hybrid(local: list[dict], web: list[dict], k: int) -> list[dict]:
    """Best-scoring passages overall, but always leave room for some web evidence."""
    by_score = lambda c: c.get("rerank_score", -999.0)
    quota = min(MIN_WEB_SLOTS, len(web), k)
    rest = sorted(local + web[quota:], key=by_score, reverse=True)[: k - quota]
    return sorted(web[:quota] + rest, key=by_score, reverse=True)


def answer_query(
    question: str,
    *,
    hybrid,
    reranker,
    web_search,
    model_name: str,
    enable_agent: bool = True,
    top_k_retrieval: int = 20,
    top_k_rerank: int = 5,
    step: Optional[Callable[[str], None]] = None,
) -> dict:
    step = step or (lambda _label: None)
    tools.set_retriever(hybrid)  # agent tools reuse the already-loaded retriever

    step("Analysing query intent")
    routing = classify_query(question)
    route = routing["source"]
    k = top_k_rerank

    local_ranked: list[dict] = []
    if route in ("LOCAL", "BOTH") or (route == "WEB" and not web_search):
        step("Searching Acts, Supreme Court and High Court corpora")
        search_q = retrieval_query(question)
        candidates = hybrid.search(search_q, top_k=top_k_retrieval)
        step(f"Reranking {len(candidates)} candidates")
        local_ranked = reranker.rerank(search_q, candidates, top_k=max(k, 10)) if candidates else []

    top_local = local_ranked[0]["rerank_score"] if local_ranked else None
    relevant_local = [c for c in local_ranked if c["rerank_score"] >= CHUNK_MIN_SCORE][:k]

    # Decide whether web evidence is wanted
    if route == "WEB":
        want_web, reason = True, "Non-legal or general question: answered from the web."
    elif route == "BOTH":
        want_web, reason = True, "Router asked for both local jurisprudence and live web context."
    elif top_local is not None and top_local >= LOCAL_STRONG:
        want_web, reason = False, f"Strong local match (top score {top_local:.2f}): answered from the legal corpus."
    elif top_local is None or top_local < LOCAL_IRRELEVANT:
        want_web, reason = True, "Nothing relevant in the legal corpus: answered from the web."
    else:
        want_web, reason = True, f"Partial local match (top score {top_local:.2f}): supplemented with web sources."

    web_ranked: list[dict] = []
    web_unavailable = False
    if want_web:
        if web_search:
            step("Searching the web")
            web_hits = web_search.search(question, max_results=WEB_RESULTS)
            web_ranked = reranker.rerank(question, web_hits, top_k=WEB_RESULTS) if web_hits else []
        web_unavailable = not web_ranked

    # Assemble final evidence
    drop_local = route == "WEB" or (top_local is not None and top_local < LOCAL_IRRELEVANT)
    if web_ranked and (drop_local or not relevant_local):
        final = web_ranked[:k]
    elif web_ranked:
        final = _merge_hybrid(relevant_local, web_ranked, k)
    else:
        # Local only, or web was wanted but unavailable: use the best local evidence we have
        usable = top_local is not None and top_local >= LOCAL_IRRELEVANT
        final = relevant_local or (local_ranked[: min(3, k)] if usable else [])
        if want_web and web_unavailable:
            reason += " Web search unavailable, so falling back to the best local evidence."

    kinds = {c.get("retrieval_type") == "web" for c in final}
    mode = "none" if not final else ("hybrid" if kinds == {True, False} else ("web" if True in kinds else "local"))
    if mode == "web" and route != "WEB" and not drop_local:
        reason += " No local passage cleared the relevance bar, so only web sources were used."
    elif mode == "none":
        reason += " No usable evidence was found."

    # Answer
    drafting = is_drafting_request(question)
    use_agent = enable_agent and (drafting or is_calculation_request(question))
    doc_path = doc_type = tool_used = None
    if use_agent:
        step("Drafting document with the legal agent" if drafting else f"Agent reasoning with {model_name}")
        res = run_legal_agent(question, context_chunks=final[:MAX_CONTEXT], model_name=model_name)
        answer = res["response"]
        doc_path, doc_type, tool_used = res.get("document_path"), res.get("document_type"), res.get("tool_used")
        if res.get("error"):
            answer += f"\n\n> **Document generation failed:** {res['error']}"
    else:
        step(f"Generating answer with {model_name}")
        answer = generate_answer(question, final[:MAX_CONTEXT], model_name=model_name, intent=routing["intent"], source=route)
        if drafting:
            answer += "\n\n> Enable *Agent tool calling* in the sidebar to export this as a Word document."

    return {
        "content": answer,
        "routing": {
            "source": route,
            "intent": routing["intent"],
            "reasoning": routing["reasoning"],
            "mode": mode,
            "decision": reason,
            "top_local_score": top_local,
        },
        "sources": [_source_card(c) for c in final],
        "used_web": mode in ("web", "hybrid"),
        "tool_used": tool_used,
        "document_path": doc_path,
        "document_type": doc_type,
        "model": model_name,
    }


def run_query(question, hybrid, reranker, figures=None, web_search=None, verbose=True, model_name=None) -> dict:
    """CLI wrapper kept for main.py."""
    res = answer_query(
        question,
        hybrid=hybrid,
        reranker=reranker,
        web_search=web_search,
        model_name=model_name,
        step=(lambda s: print(f"[Pipeline] {s}")) if verbose else None,
    )
    if verbose:
        r = res["routing"]
        print(f"[Router]   source={r['source']} intent={r['intent']} | {r['reasoning']}")
        print(f"[Evidence] {MODE_LABELS[r['mode']]}: {r['decision']}")
    return {"question": question, "routing": res["routing"], "retrieved": res["sources"], "answer": res["content"], "used_web": res["used_web"]}
