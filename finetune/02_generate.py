"""
02_generate.py - Stage 2: build raw training examples.

Three resumable phases (each appends to its own JSONL, so Ctrl+C / crashes are safe):

  A. questions : teacher LLM writes a realistic user question per seed passage
                 (+ out-of-scope questions the corpus can NOT answer)
  B. retrieve  : the REAL retriever + cross-encoder produce the evidence the model
                 would actually see at inference time (so train == deploy)
  C. answers   : teacher LLM writes a grounded, cited answer (or a clean
                 "insufficient evidence" refusal) + a routing label

Because evidence comes from the real retriever, "unanswerable" examples occur
naturally when retrieval misses - exactly what the model must learn to handle.

Usage (from anywhere):
    python finetune/02_generate.py --phase all --workers 32
    python finetune/02_generate.py --phase questions --limit 20     # smoke test
"""

import argparse
import hashlib
import os
import random

from common import (
    DATA_DIR, INSUFFICIENT_SENTENCE, Teacher, append_jsonl, build_answer_prompt,
    build_router_prompt, done_ids, parse_json_loose, read_jsonl, run_parallel,
)

SEEDS_PATH = DATA_DIR / "seeds.jsonl"
QUESTIONS_PATH = DATA_DIR / "questions.jsonl"
CONTEXTS_PATH = DATA_DIR / "contexts.jsonl"
RAW_PATH = DATA_DIR / "raw.jsonl"

# ----------------------------------------------------------------------
# Phase A prompts
# ----------------------------------------------------------------------
STYLES = {
    "common_man": (6, "A layperson describes a real-life situation in 2-3 sentences (landlord, employer, police, "
                      "family property, cheque bounce, consumer issue...) and asks what the law says or what they can do."),
    "definition": (3, "A law student asks what a legal term, section or concept means, or what its ingredients/conditions are."),
    "precedent": (3, "A lawyer asks how Indian courts have interpreted or decided a specific legal issue. "
                     "Do not name any case."),
    "procedure": (3, "Someone asks for the step-by-step procedure, time limit, forum or remedy for a legal matter."),
    "scenario": (3, "A detailed fact pattern (3-4 sentences, with ages/amounts/dates) followed by a pointed legal "
                    "question such as 'is this an offence?' or 'does he have a right to ...?'."),
    "keyword": (2, "A very short search-style query of 4-9 words, like someone typing into a search box."),
}

SOURCE_DESC = {
    "acts": "a section of an Indian Central/State Act",
    "supreme_court": "a Supreme Court of India judgment",
    "high_court": "an Indian High Court judgment",
}

QUESTION_SYSTEM = ("You write realistic questions that real people ask an Indian legal assistant. "
                   "Output ONLY valid JSON.")

QUESTION_USER = """Below is a passage from {source_desc}.

PASSAGE:
\"\"\"
{text}
\"\"\"

Write ONE question in this style:
{style_desc}

Rules:
- The question must be fully self-contained: a stranger must understand it without the passage.
- It must concern the legal issue or principle in the passage, so that this passage (or closely related law) could help answer it.
- NEVER mention "passage", "text", "judgment above", file names, case numbers or party names.
- Do not copy long phrases from the passage; use natural wording that fits the style.
- Indian law only.

Return JSON: {{"question": "..."}}"""

OOS_CATEGORIES = {
    "current_events": "breaking news, ongoing high-profile trials, recent verdicts or recent political/legal events",
    "foreign_law": "laws or courts of other countries (US, UK, UAE, Canada, Australia...)",
    "non_legal": "everyday, medical, financial, technical or trivia questions that are not about law at all",
    "personalised": "questions needing private facts or live records (status of MY case number, what lawyer fees are in "
                    "MY city, whether MY specific contract is valid without seeing it)",
    "very_recent_law": "amendments, bills or notifications from 2025 or later, and 'latest' rule changes",
    "vague": "extremely vague or ambiguous legal questions with no usable detail (e.g. 'is it legal?', 'help me with law')",
}

OOS_USER = """Generate {n} diverse questions a user might ask an Indian legal assistant that CANNOT be answered from a
static database of Indian Acts and past court judgments.

Category: {cat_desc}

Vary tone, length and phrasing (some formal, some casual, some with typos). Return JSON:
{{"questions": ["...", "..."]}}"""

# ----------------------------------------------------------------------
# Phase C prompt
# ----------------------------------------------------------------------
ANSWER_SYSTEM = f"""You are the teacher for a legal assistant that must stay strictly grounded in retrieved evidence.
You will receive a prompt containing a USER QUESTION and numbered EVIDENCE items. Write the ideal reply.

HARD RULES
1. Use ONLY the EVIDENCE. No outside knowledge, no reciting statute text from memory, no guessing what a section "probably" says.
2. Cite inline using the evidence number in square brackets right after each claim, e.g. "... within 90 days [E2]."
   Also name the Act/section or the case in plain words the first time you use it.
3. If the evidence does NOT contain enough to answer, your reply MUST begin with this exact sentence:
   "{INSUFFICIENT_SENTENCE}"
   Then, in at most 2 short sentences, say what the evidence does cover (with [E#] citations) or what is missing.
   Do NOT then give a full answer anyway and do NOT contradict yourself.
4. If the evidence supports only PART of the question, answer that part and clearly state what is not covered.
5. Never use hedge-words such as "implied", "inferred", "presumably", "likely" for legal statements.
   Ignore garbled OCR text. Never invent case names, section numbers or amounts.
6. Questions that are off-topic, personal-record lookups, foreign law, or about events not in the evidence -> rule 3.
7. Plain, professional language a common person can follow. Maximum 320 words.

FORMAT (when answerable)
**Short answer:** 1-2 sentences.
**Explanation:** short paragraphs or bullets with [E#] citations.
**Sources relied upon:** one line per source used: "[E#] <Act/case title> - <section/page if available>".
(For the insufficient case, no headings are needed.)"""


def seeded_rng(key: str) -> random.Random:
    return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:12], 16))


# ----------------------------------------------------------------------
# Phase A
# ----------------------------------------------------------------------
def bad_question(q: str) -> bool:
    ql = q.lower()
    n = len(q.split())
    return (n < 4 or n > 150 or "passage" in ql or "the text" in ql or "above judgment" in ql
            or "this judgment" in ql or ".pdf" in ql)


def phase_questions(args, teacher):
    seeds = read_jsonl(SEEDS_PATH)
    done = done_ids(QUESTIONS_PATH, "qid")
    todo = [s for s in seeds if s["seed_id"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[A] {len(todo)} seeds need questions ({len(done)} done)")

    names = list(STYLES)
    weights = [STYLES[k][0] for k in names]

    def work(seed):
        rng = seeded_rng(seed["seed_id"])
        style = rng.choices(names, weights=weights, k=1)[0]
        user = QUESTION_USER.format(
            source_desc=SOURCE_DESC[seed["source"]],
            text=seed["text"][:3500],
            style_desc=STYLES[style][1],
        )
        for _ in range(3):
            out = parse_json_loose(teacher.chat(QUESTION_SYSTEM, user, temperature=0.9, max_tokens=2048))
            q = str(out.get("question", "")).strip()
            if q and not bad_question(q):
                append_jsonl(QUESTIONS_PATH, {
                    "qid": seed["seed_id"], "seed_id": seed["seed_id"], "kind": "seed",
                    "style": style, "question": q,
                })
                return
        raise RuntimeError("no valid question after 3 tries")

    run_parallel(work, todo, args.workers, "questions")

    # ---- out-of-scope questions -------------------------------------------------
    if args.oos > 0:
        existing_oos = [r for r in read_jsonl(QUESTIONS_PATH) if r["kind"] == "oos"]
        need = args.oos - len(existing_oos)
        if need > 0:
            per_call = 20
            jobs = []
            cats = list(OOS_CATEGORIES)
            for i in range((need + per_call - 1) // per_call):
                jobs.append((i, cats[i % len(cats)]))

            def oos_work(job):
                i, cat = job
                user = OOS_USER.format(n=per_call, cat_desc=OOS_CATEGORIES[cat])
                out = parse_json_loose(teacher.chat("Output ONLY valid JSON.", user, temperature=1.0, max_tokens=1500))
                for j, q in enumerate(out.get("questions", [])):
                    q = str(q).strip()
                    if q and not bad_question(q):
                        append_jsonl(QUESTIONS_PATH, {
                            "qid": f"oos:{cat}:{i}:{j}:{hashlib.md5(q.encode()).hexdigest()[:6]}",
                            "seed_id": None, "kind": "oos", "style": cat, "question": q,
                        })

            run_parallel(oos_work, jobs, args.workers, "oos-questions")


# ----------------------------------------------------------------------
# Phase B  (needs the local LanceDB + embedding/reranker models; GPU optional)
# ----------------------------------------------------------------------
def phase_retrieve(args):
    from tqdm import tqdm
    from src.reranking import Reranker
    from src.retrieval import LegalRetriever

    questions = read_jsonl(QUESTIONS_PATH)
    done = done_ids(CONTEXTS_PATH, "qid")
    todo = [q for q in questions if q["qid"] not in done]
    if args.limit:
        todo = todo[: args.limit]
    print(f"[B] {len(todo)} questions need retrieval ({len(done)} done)")
    if not todo:
        return

    seeds = {s["seed_id"]: s for s in read_jsonl(SEEDS_PATH)}
    retriever, reranker = LegalRetriever(), Reranker()

    # Inference uses 3-7 final chunks (CLI: 7, Streamlit slider: 3-10 -> [:7]).
    k_choices, k_weights = [3, 4, 5, 6, 7], [1, 2, 3, 3, 6]

    def work(q):
        cands = retriever.search(q["question"], top_k=args.top_k)
        ranked = reranker.rerank(q["question"], cands, top_k=args.rerank_k)
        if not ranked:
            return
        rng = seeded_rng(q["qid"] + "k")
        k = rng.choices(k_choices, weights=k_weights, k=1)[0]
        final = ranked[:k]

        evidence = [{
            "paper_title": e.get("paper_title", "Unknown Document"),
            "page": e.get("page", "N/A"),
            "text": e.get("text", ""),
            "source_type": e.get("source_type", ""),
            "chunk_id": e.get("chunk_id", ""),
            "rerank_score": round(float(e.get("rerank_score", 0.0)), 4),
        } for e in final]

        seed = seeds.get(q["seed_id"]) if q["seed_id"] else None
        append_jsonl(CONTEXTS_PATH, {
            **q,
            "doc_id": seed["doc_id"] if seed else "oos",
            "seed_hit": bool(seed and any(e["chunk_id"] == seed["chunk_id"] for e in evidence)),
            "evidence": evidence,
            "prompt": build_answer_prompt(q["question"], final),
        })

    run_parallel(work, todo, args.retrieval_workers, "retrieve")


# ----------------------------------------------------------------------
# Phase C
# ----------------------------------------------------------------------
VALID_INTENTS = {"STATUTE", "PRECEDENT", "GENERAL_ADVICE", "CURRENT_INFORMATION", "UNKNOWN"}
VALID_SOURCES = {"LOCAL", "WEB", "BOTH"}


def phase_answers(args, teacher):
    ctxs = read_jsonl(CONTEXTS_PATH)
    done = done_ids(RAW_PATH, "qid")
    todo = [c for c in ctxs if c["qid"] not in done]
    # Slow local teacher: do answerable examples (seed_hit) first, then out-of-scope, then the rest,
    # so stopping the run early still leaves a balanced, high-value dataset.
    todo.sort(key=lambda c: 0 if c["kind"] == "seed" and c["seed_hit"] else (1 if c["kind"] == "oos" else 2))
    if args.limit:
        todo = todo[: args.limit]
    print(f"[C] {len(todo)} examples need answers ({len(done)} done)")

    def work(c):
        answer = teacher.chat(ANSWER_SYSTEM, c["prompt"], temperature=0.3, max_tokens=900)

        route_prompt = build_router_prompt(c["question"])
        route = None
        try:
            r = parse_json_loose(teacher.chat("", route_prompt, temperature=0.0, max_tokens=1500))
            intent, source = str(r.get("intent", "")).upper(), str(r.get("source", "")).upper()
            if intent in VALID_INTENTS and source in VALID_SOURCES:
                route = {"intent": intent, "source": source, "reasoning": str(r.get("reasoning", ""))[:300]}
        except Exception:  # noqa: BLE001  (routing label is optional per example)
            pass

        append_jsonl(RAW_PATH, {
            "qid": c["qid"], "kind": c["kind"], "style": c["style"], "doc_id": c["doc_id"],
            "question": c["question"], "seed_hit": c["seed_hit"],
            "n_evidence": len(c["evidence"]), "prompt": c["prompt"], "answer": answer,
            "route_prompt": route_prompt, "route": route,
        })

    run_parallel(work, todo, args.workers, "answers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["questions", "retrieve", "answers", "all"], default="all")
    is_local = "11434" in os.getenv("TEACHER_BASE_URL", "") or "localhost" in os.getenv("TEACHER_BASE_URL", "")
    ap.add_argument("--workers", type=int, default=4 if is_local else 8, help="parallel teacher requests")
    ap.add_argument("--oos", type=int, default=1000, help="number of out-of-scope questions")
    ap.add_argument("--limit", type=int, default=0, help="process only the first N (smoke test)")
    ap.add_argument("--top-k", type=int, default=30)
    ap.add_argument("--rerank-k", type=int, default=10)
    ap.add_argument("--retrieval-workers", type=int, default=4,
                    help="parallel local retrievals in phase B (set 1 if you hit GPU/memory issues)")
    args = ap.parse_args()

    if args.phase in ("questions", "all"):
        phase_questions(args, Teacher())
    if args.phase in ("retrieve", "all"):
        phase_retrieve(args)
    if args.phase in ("answers", "all"):
        phase_answers(args, Teacher())
    print("\nDone. Next: python finetune/03_build_dataset.py")


if __name__ == "__main__":
    main()
