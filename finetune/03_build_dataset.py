"""
03_build_dataset.py - Stage 3: filter, (optionally) judge, balance, split.

Turns data/raw.jsonl into data/train.jsonl + data/val.jsonl in chat "messages"
format, mixing two tasks that the deployed system actually uses:

  task=answer : grounded + cited answer (or clean refusal)   <- the main skill
  task=route  : query-router JSON                            <- keeps routing intact

Quality gates (all automatic):
  * format: [E#] citations exist and point to evidence that was really provided
  * refusal: exact refusal sentence, short, no "answer anyway" contradiction
  * no hedge-words, no <think>, no CJK leakage, sane length
  * optional LLM judge for faithfulness (--judge)
  * exact/normalised question dedup
  * refusal share capped (--max-refusal) so the model doesn't learn to always refuse
  * split BY DOCUMENT so val never shares a source with train
"""

import argparse
import json
import random
import re
from collections import Counter

from common import (
    DATA_DIR, INSUFFICIENT_SENTENCE, Teacher, append_jsonl, parse_json_loose, read_jsonl,
    run_parallel, write_jsonl,
)

RAW_PATH = DATA_DIR / "raw.jsonl"
JUDGED_PATH = DATA_DIR / "judged.jsonl"
TRAIN_PATH = DATA_DIR / "train.jsonl"
VAL_PATH = DATA_DIR / "val.jsonl"
STATS_PATH = DATA_DIR / "stats.json"

CITE_RE = re.compile(r"\[E(\d+)\]")
HEDGE_RE = re.compile(r"\b(implied|inferred|presumably|i think|as an ai|it seems that|probably)\b", re.I)
CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def is_refusal(ans: str) -> bool:
    return ans.strip().startswith(INSUFFICIENT_SENTENCE)


def check_answer(row: dict):
    """Return a rejection reason, or None if the answer passes."""
    ans = row["answer"].strip()
    words = len(ans.split())
    if not ans or "<think>" in ans or CJK_RE.search(ans):
        return "empty_or_leak"
    if HEDGE_RE.search(ans):
        return "hedge_words"
    cites = [int(x) for x in CITE_RE.findall(ans)]
    if any(c < 1 or c > row["n_evidence"] for c in cites):
        return "bad_citation_index"
    if is_refusal(ans):
        if words > 90:
            return "refusal_too_long"
        # contradiction guard: a refusal must not be followed by a full structured answer
        if "**Short answer" in ans or "**Explanation" in ans:
            return "refusal_then_answer"
        return None
    # answerable
    if words < 25 or words > 420:
        return "length"
    if not cites:
        return "no_citations"
    if INSUFFICIENT_SENTENCE in ans:
        return "refusal_sentence_midanswer"
    if "**Short answer" not in ans:
        return "format"
    return None


def norm_q(q: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", q.lower()).strip()


JUDGE_SYSTEM = "You are a strict legal-RAG auditor. Output ONLY valid JSON."
JUDGE_USER = """Audit the ASSISTANT REPLY against the prompt it was given.

{prompt}

ASSISTANT REPLY:
{answer}

Score 1-5 for FAITHFULNESS: 5 = every claim is directly supported by the cited evidence and nothing is invented;
3 = minor unsupported detail; 1 = fabricated or contradicts the evidence. If the reply is a refusal, score 5 only if the
evidence truly lacks what is needed, else 1.
Return JSON: {{"faithfulness": <1-5>, "reason": "<short>"}}"""


def run_judge(rows: list, workers: int) -> dict:
    teacher = Teacher()
    done = {r["qid"] for r in read_jsonl(JUDGED_PATH)}
    todo = [r for r in rows if r["qid"] not in done]
    print(f"[judge] {len(todo)} to judge ({len(done)} cached)")

    def work(r):
        out = parse_json_loose(teacher.chat(
            JUDGE_SYSTEM, JUDGE_USER.format(prompt=r["prompt"], answer=r["answer"]),
            temperature=0.0, max_tokens=200))
        append_jsonl(JUDGED_PATH, {"qid": r["qid"], "faithfulness": int(out.get("faithfulness", 0)),
                                   "reason": str(out.get("reason", ""))[:200]})

    run_parallel(work, todo, workers, "judge")
    return {r["qid"]: r["faithfulness"] for r in read_jsonl(JUDGED_PATH)}


def to_example(prompt: str, completion: str, meta: dict) -> dict:
    return {
        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
        **meta,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true", help="LLM faithfulness audit (recommended for the final run)")
    ap.add_argument("--min-faith", type=int, default=4)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--max-refusal", type=float, default=0.22, help="max share of refusals among answer examples")
    ap.add_argument("--val-ratio", type=float, default=0.03)
    ap.add_argument("--max-val", type=int, default=600)
    ap.add_argument("--no-route", action="store_true", help="exclude router examples")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    raw = read_jsonl(RAW_PATH)
    print(f"raw examples: {len(raw)}")

    # ---- 1. rule-based filtering ---------------------------------------------
    reasons, kept, seen_q = Counter(), [], set()
    for r in raw:
        why = check_answer(r)
        key = norm_q(r["question"])
        if why is None and key in seen_q:
            why = "duplicate_question"
        if why:
            reasons[why] += 1
            continue
        seen_q.add(key)
        kept.append(r)
    print(f"after rule filters: {len(kept)}  | rejected: {dict(reasons)}")

    # ---- 2. optional LLM judge --------------------------------------------------
    if args.judge:
        scores = run_judge(kept, args.workers)
        before = len(kept)
        kept = [r for r in kept if scores.get(r["qid"], 0) >= args.min_faith]
        print(f"after judge (>= {args.min_faith}): {len(kept)} (dropped {before - len(kept)})")

    # ---- 3. cap the refusal share ----------------------------------------------
    refusals = [r for r in kept if is_refusal(r["answer"])]
    answerable = [r for r in kept if not is_refusal(r["answer"])]
    max_ref = int(args.max_refusal * len(answerable) / (1 - args.max_refusal)) if answerable else 0
    if len(refusals) > max_ref:
        rng.shuffle(refusals)
        refusals = refusals[:max_ref]
    kept = answerable + refusals
    print(f"answerable: {len(answerable)} | refusals kept: {len(refusals)}")

    # ---- 4. split by document ---------------------------------------------------
    docs = sorted({r["doc_id"] for r in kept if r["doc_id"] != "oos"})
    rng.shuffle(docs)
    val_docs = set(docs[: max(1, int(len(docs) * args.val_ratio))])
    train_rows, val_rows = [], []
    for r in kept:
        if r["doc_id"] == "oos":  # out-of-scope questions have no source doc -> split per question
            (val_rows if rng.random() < args.val_ratio else train_rows).append(r)
        else:
            (val_rows if r["doc_id"] in val_docs else train_rows).append(r)
    rng.shuffle(val_rows)
    val_rows = val_rows[: args.max_val]

    # ---- 5. materialise chat examples ------------------------------------------
    def build(rows):
        out = []
        for r in rows:
            meta = {"qid": r["qid"], "kind": r["kind"], "n_evidence": r["n_evidence"]}
            out.append(to_example(r["prompt"], r["answer"].strip(),
                                  {**meta, "task": "answer", "refusal": is_refusal(r["answer"])}))
            if not args.no_route and r.get("route"):
                out.append(to_example(r["route_prompt"], json.dumps(r["route"], ensure_ascii=False),
                                      {**meta, "task": "route", "refusal": False}))
        return out

    train, val = build(train_rows), build(val_rows)
    rng.shuffle(train)
    write_jsonl(TRAIN_PATH, train)
    write_jsonl(VAL_PATH, val)

    n_ans = max(1, sum(e["task"] == "answer" for e in train))
    stats = {
        "raw": len(raw), "kept_questions": len(kept), "train_examples": len(train), "val_examples": len(val),
        "train_by_task": dict(Counter(e["task"] for e in train)),
        "train_refusal_share_of_answers": round(sum(e["refusal"] for e in train if e["task"] == "answer") / n_ans, 3),
        "train_by_kind": dict(Counter(e["kind"] for e in train)),
        "rejections": dict(reasons),
    }
    STATS_PATH.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print("\n" + json.dumps(stats, indent=2))
    print(f"\nWrote {TRAIN_PATH.name} ({len(train)}) and {VAL_PATH.name} ({len(val)}) in {DATA_DIR}")
    print("Spot-check ~50 random lines of train.jsonl by eye before training!")


if __name__ == "__main__":
    main()
