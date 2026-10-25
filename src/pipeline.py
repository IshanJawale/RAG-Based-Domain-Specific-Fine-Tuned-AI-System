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
     everything else gets a grounded answer from the retrieved evidence, streamed.

Follow-ups ("and the punishment?") are resolved against the chat history, and requests to change a
document the assistant just drafted are routed back to the agent with that document's fields.
The UI calls prepare_query -> stream_answer / agent_answer -> finalize; answer_query wraps all of it.

Cross-encoder (ms-marco-MiniLM-L-6-v2) logits measured on this corpus:
covered legal questions score 3-8, partially covered ones -1..3, off-topic < -2.
"""

import re
from typing import Callable, Optional

from src.routing import classify_query
from src.generation import generate_answer_stream
from src.agent import run_legal_agent, is_drafting_request, is_calculation_request
from src import tools
from src.legal_formats import classify_document, describe_format

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


_FOLLOW_UP_START = re.compile(
    r"^\s*(?:and|but|also|so|then|what about|how about|what if|in that case|same|ok(?:ay)?|now|instead|similarly)\b", re.I
)
_REFERS_BACK = re.compile(r"\b(?:it|this|that|they|them|those|these|its|their|such|the same|above|said|the act|the section|the case)\b", re.I)
_REVISE_DOC = re.compile(
    r"\b(?:change|add|update|make it|replace|remove|delete|include|edit|revise|modify|correct|fix|set|put|mention|insert|rewrite|redo|instead)\b",
    re.I,
)


def contextualize(question: str, history: Optional[list[dict]]) -> dict:
    """Resolve follow-ups against the chat history.

    Returns {"search_text", "follow_up", "revise"}: search_text is what routing/retrieval use (the previous
    question is prepended for short follow-ups like "and the punishment?"); revise holds the earlier document's
    format and fields when the user asks to change a document the assistant just produced."""
    out = {"search_text": question, "follow_up": False, "revise": None}
    if not history:
        return out
    prev_user = next((m["content"] for m in reversed(history) if m.get("role") == "user"), None)
    last_bot = next((m for m in reversed(history) if m.get("role") == "assistant"), None)

    if last_bot and last_bot.get("document_args") and _REVISE_DOC.search(question) and not is_drafting_request(question):
        key = last_bot["document_args"].get("document_type")
        if key:
            out["revise"] = {"doc_key": key, "base_args": last_bot["document_args"]}

    short = len(question.split()) <= 14
    if prev_user and short and (_FOLLOW_UP_START.search(question) or _REFERS_BACK.search(question)):
        out["follow_up"] = True
        out["search_text"] = f"{prev_user} {question}"
    return out


def prepare_query(
    question: str,
    *,
    hybrid,
    reranker,
    web_search,
    top_k_retrieval: int = 20,
    top_k_rerank: int = 5,
    history: Optional[list[dict]] = None,
    step: Optional[Callable[[str], None]] = None,
) -> dict:
    """Route, retrieve and choose the evidence. Everything except generating the answer."""
    step = step or (lambda _label: None)
    tools.set_retriever(hybrid)  # agent tools reuse the already-loaded retriever

    ctx = contextualize(question, history)
    text = ctx["search_text"]

    step("Analysing query intent" + (" (follow-up)" if ctx["follow_up"] else ""))
    routing = classify_query(text)
    route = routing["source"]
    if ctx["follow_up"]:
        routing = {**routing, "reasoning": "Follow-up resolved against the previous question. " + routing["reasoning"]}

    # Drafting requests get a sub-classification into a concrete document format
    revise = ctx["revise"]
    drafting = bool(revise) or is_drafting_request(question)
    doc_key = revise["doc_key"] if revise else (classify_document(question) if drafting else None)
    doc_format = describe_format(doc_key) if drafting else None
    if drafting:
        what = "Revising the previous" if revise else "Drafting request:"
        routing = {**routing, "intent": "DRAFTING", "reasoning": f"{what} {doc_format['label']} ({doc_format['basis']}). " + routing["reasoning"]}
        if route == "WEB":
            route = "LOCAL"  # drafting is legal work: check the corpus first, web only as fallback
    k = top_k_rerank

    local_ranked: list[dict] = []
    if route in ("LOCAL", "BOTH") or (route == "WEB" and not web_search):
        step("Searching Acts, Supreme Court and High Court corpora")
        search_q = retrieval_query(text)
        if doc_format and re.search(r"\b(?:Section|Article|Order)\b", doc_format["basis"]):
            search_q += f" {doc_format['basis']}"  # pull in the governing provisions
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
            web_hits = web_search.search(text, max_results=WEB_RESULTS)
            web_ranked = reranker.rerank(text, web_hits, top_k=WEB_RESULTS) if web_hits else []
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

    return {
        "question": question,
        "history": history or [],
        "routing": {
            "source": route,
            "intent": routing["intent"],
            "reasoning": routing["reasoning"],
            "mode": mode,
            "decision": reason,
            "top_local_score": top_local,
            "document_format": doc_format,
            "follow_up": ctx["follow_up"],
        },
        "final": final,
        "drafting": drafting,
        "doc_key": doc_key,
        "revise": revise,
        "calculation": is_calculation_request(question),
    }


def needs_agent(prep: dict, enable_agent: bool) -> bool:
    return enable_agent and (prep["drafting"] or prep["calculation"])


def stream_answer(prep: dict, model_name: str):
    """Yield the grounded answer as it is generated (non-agent path)."""
    r = prep["routing"]
    yield from generate_answer_stream(
        prep["question"], prep["final"][:MAX_CONTEXT], model_name=model_name,
        intent=r["intent"], source=r["source"], history=prep["history"],
    )
    if prep["drafting"]:
        yield "\n\n> Enable *Agent tool calling* in the sidebar to export this as a Word document."


def agent_answer(prep: dict, model_name: str) -> dict:
    """Drafting / calculation through the tool-using agent. Returns the agent's result dict."""
    revise = prep["revise"] or {}
    res = run_legal_agent(
        prep["question"], context_chunks=prep["final"][:MAX_CONTEXT], model_name=model_name,
        history=prep["history"], doc_key=revise.get("doc_key") or (prep["doc_key"] if prep["drafting"] else None),
        base_args=revise.get("base_args"),
    )
    if res.get("error"):
        res["response"] += f"\n\n> **Document generation failed:** {res['error']}"
    return res


def finalize(prep: dict, content: str, model_name: str, agent_result: Optional[dict] = None) -> dict:
    """The stored assistant message: answer text plus routing, sources and any generated document."""
    agent_result = agent_result or {}
    mode = prep["routing"]["mode"]
    return {
        "content": content,
        "routing": prep["routing"],
        "sources": [_source_card(c) for c in prep["final"]],
        "used_web": mode in ("web", "hybrid"),
        "tool_used": agent_result.get("tool_used"),
        "document_path": agent_result.get("document_path"),
        "document_type": agent_result.get("document_type"),
        "document_args": agent_result.get("document_args") if agent_result.get("document_path") else None,
        "model": model_name,
    }


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
    history: Optional[list[dict]] = None,
    step: Optional[Callable[[str], None]] = None,
) -> dict:
    """Non-streaming convenience wrapper (CLI and tests)."""
    step = step or (lambda _label: None)
    prep = prepare_query(question, hybrid=hybrid, reranker=reranker, web_search=web_search,
                         top_k_retrieval=top_k_retrieval, top_k_rerank=top_k_rerank, history=history, step=step)
    if needs_agent(prep, enable_agent):
        step("Drafting document with the legal agent" if prep["drafting"] else f"Agent reasoning with {model_name}")
        res = agent_answer(prep, model_name)
        return finalize(prep, res["response"], model_name, res)
    step(f"Generating answer with {model_name}")
    return finalize(prep, "".join(stream_answer(prep, model_name)), model_name)


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
