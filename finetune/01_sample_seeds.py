"""
01_sample_seeds.py - Stage 1: sample "seed" passages from the LanceDB corpus.

A seed is a random, clean passage (an Act section or a slice of a judgment).
Stage 2 turns each seed into a realistic user question; the real retriever then
decides what evidence the model actually sees (so training matches deployment).

Run from anywhere:
    python finetune/01_sample_seeds.py --n 12000
"""

import argparse
import random
import re

import lancedb
from tqdm import tqdm

from common import DATA_DIR, ROOT, append_jsonl, is_clean_text, read_jsonl

SEEDS_PATH = DATA_DIR / "seeds.jsonl"

# Share of seeds per corpus (acts are small but dense, courts are huge).
QUOTAS = {"acts": 0.25, "supreme_court": 0.40, "high_court": 0.35}
MIN_WORDS = {"acts": 40, "supreme_court": 200, "high_court": 200}
MAX_PER_DOC = 2  # never over-represent one judgment / one Act


def strip_header(text: str) -> str:
    """Chunks start with '[High Court of India - 2019] case_id\\n\\n...' - drop it."""
    if text.startswith("[") and "\n\n" in text:
        return text.split("\n\n", 1)[1]
    return text


# Housekeeping sections make poor question seeds (no real legal substance).
_LOW_VALUE_ACT = re.compile(
    r"^\W*(amendment of|insertion of|substitution of|omission of|repeal|short title|"
    r"commencement|extent|savings?|power to remove difficulties|validation)",
    re.I,
)


def sample_table(db, name: str, target: int, rng: random.Random, seen_chunks: set) -> list[dict]:
    tbl = db.open_table(name)
    n_rows = tbl.count_rows()
    cols = (
        ["chunk_id", "act_id", "act_title", "section_number", "section_heading", "text"]
        if name == "acts"
        else ["chunk_id", "case_id", "year", "chunk_index", "text"]
    )

    out, per_doc = [], {}
    pbar = tqdm(total=target, desc=f"seeds:{name}")
    attempts = 0
    while len(out) < target and attempts < 60:
        attempts += 1
        offsets = sorted(rng.sample(range(n_rows), min(n_rows, max(500, target))))
        rows = tbl.take_offsets(offsets).select(cols).to_list()
        rng.shuffle(rows)
        for r in rows:
            if len(out) >= target:
                break
            if r["chunk_id"] in seen_chunks:
                continue
            body = strip_header(r["text"])
            if not is_clean_text(body, MIN_WORDS[name]):
                continue
            if name == "acts" and _LOW_VALUE_ACT.match(body):
                continue
            doc_id = r.get("act_id") or r.get("case_id")
            if per_doc.get(doc_id, 0) >= MAX_PER_DOC:
                continue
            per_doc[doc_id] = per_doc.get(doc_id, 0) + 1
            seen_chunks.add(r["chunk_id"])
            out.append({
                "seed_id": f"{name}:{r['chunk_id']}",
                "source": name,
                "doc_id": doc_id,
                "chunk_id": r["chunk_id"],
                "title": r.get("act_title") or doc_id,
                "section": r.get("section_number") or "",
                "year": r.get("year") or "",
                "text": body[:6000],
            })
            pbar.update(1)
    pbar.close()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", "--n-seeds", dest="n", type=int, default=3000, help="total seeds to sample")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    existing = read_jsonl(SEEDS_PATH)
    seen = {r["chunk_id"] for r in existing}
    have = {k: sum(1 for r in existing if r["source"] == k) for k in QUOTAS}

    db = lancedb.connect(ROOT / "data" / "lancedb")
    for name, share in QUOTAS.items():
        target = int(args.n * share) - have[name]
        if target <= 0:
            print(f"{name}: already has {have[name]} seeds, skipping")
            continue
        for row in sample_table(db, name, target, rng, seen):
            append_jsonl(SEEDS_PATH, row)

    final = read_jsonl(SEEDS_PATH)
    print(f"\nTotal seeds: {len(final)} -> {SEEDS_PATH}")
    for k in QUOTAS:
        print(f"  {k}: {sum(1 for r in final if r['source'] == k)}")


if __name__ == "__main__":
    main()
