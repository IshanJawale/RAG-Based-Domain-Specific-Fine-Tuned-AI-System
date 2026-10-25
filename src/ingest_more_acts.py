"""
ingest_more_acts.py - Add central Acts that are missing from the LanceDB `acts` table.

Sources (both public):
  1. KanoonGPT/indian-legal-documents (Hugging Face, Apache-2.0): cleaned full text of
     Indian Acts derived from India Code. Used for most Acts.
  2. civictech-India JSON (GitHub): section-wise Code of Civil Procedure, 1908, which the
     KanoonGPT release does not contain (only State amendments to it).

Pipeline: pick central principal Acts -> skip titles already in the table -> split full text into
sections -> strip footnotes -> sub-chunk long sections -> embed with BGE-small (CLS pooling,
L2-normalised, same as src/ingest_acts.py) -> append to the `acts` table and data/acts_chunks.jsonl.

Usage (from the project root):
    python -m src.ingest_more_acts --kanoon-dir <dir with data/*.parquet> --civictech-dir <dir with cpc.json> --dry-run
    python -m src.ingest_more_acts --kanoon-dir ... --civictech-dir ...            # writes to the DB

The table is versioned by LanceDB; the script prints the version before writing so it can be
rolled back with `table.restore(<version>)`.
"""

import argparse
import glob
import json
import os
import re
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "lancedb"
CHUNKS_FILE = ROOT / "data" / "acts_chunks.jsonl"

MAX_WORDS = 350  # BGE-small sees 512 tokens; longer sections are split so every part is embedded
FOOTNOTE_START = re.compile(
    r"^(?:Subs\.|Ins\.|Omitted|Rep\.|Added|The words|For the Statement|Now see|w\.e\.f|Vide|See |Cl\.|Section \d+ was|This Act has been|It has been|Came into force|The Act comes|Extended to|Amended by)",
    re.IGNORECASE,
)


# ----------------------------------------------------------------------------
# Titles
# ----------------------------------------------------------------------------
def norm_title(t: str) -> str:
    t = (t or "").lower().replace("&", "and")
    t = re.sub(r"\(.*?\)", " ", t)             # "(Act No. 45 of 1860)", "(As amended ...)"
    t = re.sub(r"\bno\.?\s*\d+\s*of\s*\d{4}", " ", t)
    t = re.sub(r"\bthe\b", " ", t)
    t = re.sub(r"[^a-z0-9]+", " ", t)
    return " ".join(t.split())


def clean_title(t: str) -> str:
    t = re.sub(r"\s*\((?:Act\s+)?No\.?.*?\)\s*$", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"\s+No\.\s*\d+\s+of\s+\d{4}\s*$", "", t, flags=re.IGNORECASE).strip(" .")
    if t.isupper():
        small = {"of", "and", "the", "for", "in", "on", "to", "or", "by", "with", "from"}
        words = t.lower().split()
        t = " ".join(w if (i and w in small) else w[:1].upper() + w[1:] for i, w in enumerate(words))
    return t


LEGACY_FONT = re.compile(r"[a-z][;/{\]][a-z]")  # Hindi typed in Kruti Dev etc. reads as "vf/kfu;e" when extracted


def is_english(text: str) -> bool:
    sample = text[:4000]
    if len(LEGACY_FONT.findall(sample)) > 15:
        return False
    letters = sum(c.isalpha() for c in sample)
    ascii_letters = sum(c.isascii() and c.isalpha() for c in sample)
    return letters > 0 and ascii_letters / letters > 0.9


# ----------------------------------------------------------------------------
# Section splitting for India Code style text
# ----------------------------------------------------------------------------
# "1. Short title.—", "19A. Power to set aside ...—", "[1][2A. Definitions.—", "13. “Consent” defined.—"
_MARKERS = r"(?:\[\d+\]\s*|\d{1,3}\[\s*)*\[?"
SECTION_HEAD = re.compile(
    rf"(?m)^[ \t]*{_MARKERS}(?P<num>\d{{1,3}}[A-Z]{{0,3}})\.\s*{_MARKERS}(?P<head>[^\n]{{2,220}}?)(?:\s*[—–―]+|\.-)\s*"
)  # ".-" form: "1.\nShort title, extent and commencement.-(1) This Act ..." (NDPS Act)
# PDF extraction sometimes moves the dash to the start of the line:
# "— 138. Dishonour of cheque for insufficiency, etc., of funds in the account. Where any cheque ..."
SECTION_HEAD_DASH_FIRST = re.compile(
    rf"(?m)^[ \t]*[—–―]\s*{_MARKERS}(?P<num>\d{{1,3}}[A-Z]{{0,3}})\.\s*{_MARKERS}(?P<head>[^\n]{{2,220}}?)\.\s+(?=[A-Z(“\"\[])"
)


def _body_start(text: str) -> int:
    """Skip the 'ARRANGEMENT OF SECTIONS' table of contents: the body starts at the enacting clause."""
    hits = [m.start() for pat in (r"(?i)be it enacted", r"(?i)it is hereby enacted", r"(?i)enacted by Parliament", r"(?im)^\s*ACT NO\.\s*\d+")
            if (m := re.search(pat, text))]
    return min(hits) if hits else 0


SCHEDULE_HEAD = re.compile(r"(?m)^[ \t]*(?:\d{1,3}\[)?\s*(?:THE\s+)?(?:FIRST\s+|SECOND\s+|THIRD\s+)?SCHEDULE\b")


def _strip_footnotes(body: str) -> str:
    out, skipping = [], False
    for line in body.splitlines():
        s = line.strip()
        if re.fullmatch(r"\d{1,3}\.", s):        # footnote number on its own line
            skipping = True
            continue
        if FOOTNOTE_START.match(s) or re.match(r"^\d{1,3}\.\s*(?:Subs|Ins|Omitted|Rep|Added|The words)\b", s):
            skipping = True
            continue
        if skipping and (not s or SECTION_HEAD.match(line) or re.match(r"^\([a-z0-9]+\)", s)):
            skipping = False
        if skipping:
            continue
        out.append(line)
    text = "\n".join(out)
    text = re.sub(r"\[\d{1,3}\]", "", text)      # inline footnote markers
    text = re.sub(r"[ \t]+", " ", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _pick(body: str) -> list:
    """Section heading matches forming a (mostly) increasing sequence; footnote numbers restart low and are rejected."""
    heads = sorted(list(SECTION_HEAD.finditer(body)) + list(SECTION_HEAD_DASH_FIRST.finditer(body)), key=lambda m: m.start())
    picked, last, last_pos = [], 0, -1
    for m in heads:
        if m.start() <= last_pos:
            continue
        n = int(re.match(r"\d+", m.group("num")).group())
        head = m.group("head").strip(" .[]")
        if not head or len(head.split()) > 30:
            continue
        # Accept the next number, a lettered insert (19A), or a forward jump over repealed blocks (Contract Act ss. 76-123)
        if last <= n <= last + 60 and not (n == last and not re.search(r"[A-Z]", m.group("num"))):
            picked.append(m)
            last, last_pos = n, m.end()
    return picked


def _toc_max(toc: str) -> int:
    nums = [int(x) for x in re.findall(r"(?m)^\s*(\d{1,3})[A-Z]{0,3}\.", toc)]
    return max(nums) if nums else 0


def split_sections(text: str) -> list[tuple[str, str, str]]:
    """Return [(number, heading, body)] in order.

    Text after the Schedule heading is returned as one ("Schedule", heading, body) entry, so schedule
    articles (e.g. the Limitation Act's periods) are not mislabelled as sections. A "SCHEDULE" heading only
    counts once the sections before it reach (most of) the last section listed in the table of contents;
    earlier ones are usually inside footnotes (e.g. extension notifications)."""
    start = _body_start(text)
    body, toc_max = text[start:], _toc_max(text[:start])
    picked = _pick(body)
    cut = None
    for sm in SCHEDULE_HEAD.finditer(body):
        before = [int(re.match(r"\d+", m.group("num")).group()) for m in picked if m.start() < sm.start()]
        reached = before[-1] if before else 0
        if (toc_max and reached >= 0.8 * toc_max) or (not toc_max and sm.start() > 0.5 * len(body)):
            cut = sm.start()
            break
    schedule = ""
    if cut is not None:
        body, schedule = body[:cut], body[cut:]
        picked = [m for m in picked if m.start() < cut]
    sections = []
    for i, m in enumerate(picked):
        end = picked[i + 1].start() if i + 1 < len(picked) else len(body)
        sec_body = _strip_footnotes(body[m.end():end])
        if sec_body:
            sections.append((m.group("num"), m.group("head").strip(" .[]"), sec_body))
    sched = _strip_footnotes(schedule)
    if sections and len(sched.split()) > 20:
        head = sched.splitlines()[0].strip(" .[]0123456789")[:80] or "Schedule"
        sections.append(("Schedule", head, sched))
    return sections


def sub_chunks(body: str) -> list[str]:
    words = body.split()
    if len(words) <= MAX_WORDS:
        return [body]
    paras, parts, cur = re.split(r"\n+", body), [], []
    for p in paras:
        if cur and len(" ".join(cur).split()) + len(p.split()) > MAX_WORDS:
            parts.append("\n".join(cur))
            cur = []
        cur.append(p)
    if cur:
        parts.append("\n".join(cur))
    # A single huge paragraph: hard split by words
    final = []
    for part in parts:
        w = part.split()
        final.extend(" ".join(w[i:i + MAX_WORDS]) for i in range(0, len(w), MAX_WORDS))
    return final


def make_records(act_id: str, title: str, sections: list[tuple[str, str, str]], unit: str = "Section") -> list[dict]:
    recs = []
    for num, head, body in sections:
        parts = sub_chunks(body)
        for k, part in enumerate(parts, 1):
            base = num if num in ("Schedule", "Preamble") else f"{unit} {num}"
            label = base + (f" (part {k}/{len(parts)})" if len(parts) > 1 else "")
            recs.append({
                "chunk_id": str(uuid.uuid4()),
                "source_type": "act",
                "act_id": act_id,
                "act_title": title,
                "jurisdiction": "CENTRAL",
                "section_number": str(num),
                "section_heading": head,
                "text": f"[{title}] {label}: {head}\n\n{part}",
            })
    return recs


# ----------------------------------------------------------------------------
# Sources
# ----------------------------------------------------------------------------
SKIP_TITLE = re.compile(r"amendment|\brepeal|validation|\bordinance\b|\bbill\b|\brules?\b|regulations?|scheme|notification", re.I)


def load_kanoon(kanoon_dir: str, have: set[str]) -> dict[str, dict]:
    import pandas as pd
    files = sorted(glob.glob(os.path.join(kanoon_dir, "data", "*.parquet")))
    df = pd.concat([pd.read_parquet(f, columns=["doc_id", "document_title", "document_type", "document_jurisdiction", "text"]) for f in files])
    df = df[df.document_type.str.contains(r"\bAct\b|Legislation|Code", case=False, na=False)]
    df = df[df.document_jurisdiction.str.contains("Central", na=False)]
    best: dict[str, dict] = {}
    for r in df.itertuples():
        title = clean_title(r.document_title or "")
        key = norm_title(title)
        if not key or SKIP_TITLE.search(title) or LEGACY_FONT.search(title) or not re.search(r"\b1[789]\d\d\b|\b20[0-2]\d\b", title):
            continue
        if key in have or any(h.startswith(key + " ") for h in have):
            continue
        if not is_english(r.text or ""):
            continue
        # Several copies of the same Act: keep the longest English text
        if key not in best or len(r.text) > len(best[key]["text"]):
            best[key] = {"act_id": f"kanoon:{r.doc_id}", "title": title, "text": r.text}
    return best


# Clean section-wise JSON preferred over parsing for these (file, title, unit)
CIVICTECH = [
    ("cpc.json", "The Code of Civil Procedure, 1908", "Section"),     # not in KanoonGPT (only State amendments)
    ("ipc.json", "The Indian Penal Code, 1860", "Section"),           # repealed by BNS w.e.f. 1 July 2024; still cited for older cases
    ("coi.json", "The Constitution of India", "Article"),
]


def load_civictech(civic_dir: str, filename: str) -> list[tuple[str, str, str]]:
    p = Path(civic_dir) / filename
    if not p.exists():
        return []
    secs = []
    for row in json.loads(p.read_text(encoding="utf-8")):
        num = row.get("section", row.get("Section", row.get("article")))
        num = "Preamble" if num == 0 and "article" in row else str(num if num is not None else "").strip()
        head = (row.get("title") or row.get("section_title") or "").strip()
        body = (row.get("description") or row.get("section_desc") or "").strip()
        if num and body:
            secs.append((num, head, re.sub(r"\n{3,}", "\n\n", body)))
    return secs


# ----------------------------------------------------------------------------
# Embedding + write
# ----------------------------------------------------------------------------
def embed(texts: list[str], batch: int = 256) -> list[list[float]]:
    import torch
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5", local_files_only=True)
    model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5", local_files_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    out = []
    from tqdm import tqdm
    for i in tqdm(range(0, len(texts), batch), desc="Embedding"):
        enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        with torch.no_grad():
            emb = model(**enc)[0][:, 0]
        out.extend(torch.nn.functional.normalize(emb, p=2, dim=1).cpu().numpy().tolist())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kanoon-dir", required=True)
    ap.add_argument("--civictech-dir", default=None)
    ap.add_argument("--dry-run", action="store_true", help="Parse and report only; do not write")
    ap.add_argument("--min-sections", type=int, default=3, help="Skip Acts where fewer sections were recovered")
    args = ap.parse_args()

    import lancedb
    table = lancedb.connect(str(DB_PATH)).open_table("acts")
    existing = table.search().select(["act_title"]).limit(10_000_000).to_pandas()
    have = {norm_title(t) for t in existing.act_title.dropna().unique()}
    print(f"acts table: {table.count_rows()} rows, {len(have)} distinct titles")

    civic = [c for c in CIVICTECH if args.civictech_dir and norm_title(c[1]) not in have]
    # Acts taken from civictech are excluded from KanoonGPT to avoid duplicates
    acts = load_kanoon(args.kanoon_dir, have | {norm_title(c[1]) for c in civic})
    print(f"KanoonGPT: {len(acts)} central Acts not yet in the table")

    records, report, skipped = [], [], []
    for key, a in sorted(acts.items(), key=lambda kv: kv[1]["title"]):
        secs = split_sections(a["text"])
        if len(secs) < args.min_sections:
            skipped.append((a["title"], len(secs)))
            continue
        recs = make_records(a["act_id"], a["title"], secs)
        records.extend(recs)
        report.append((a["title"], len(secs), len(recs)))

    for filename, title, unit in civic:
        secs = load_civictech(args.civictech_dir, filename)
        if secs:
            recs = make_records(f"civictech:{filename.removesuffix('.json')}", title, secs, unit=unit)
            records.extend(recs)
            report.append((title + " [civictech]", len(secs), len(recs)))

    print(f"\nParsed {len(report)} Acts into {len(records)} chunks; skipped {len(skipped)} with < {args.min_sections} sections")
    for title, n_sec, n_rec in sorted(report, key=lambda r: -r[1])[:25]:
        print(f"  {n_sec:4} sections {n_rec:5} chunks  {title}")
    if skipped:
        print("  skipped (sample):", skipped[:10])

    if args.dry_run:
        print("\nDry run: nothing written.")
        return

    version = table.version
    print(f"\nTable version before write: {version} (roll back with table.restore({version}))")
    vectors = embed([r["text"] for r in records])
    for r, v in zip(records, vectors):
        r["vector"] = v
    for i in range(0, len(records), 5000):
        table.add(records[i:i + 5000])
    with open(CHUNKS_FILE, "a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps({k: v for k, v in r.items() if k != "vector"}, ensure_ascii=False) + "\n")
    print(f"Added {len(records)} chunks. acts table now has {table.count_rows()} rows (version {table.version}).")


if __name__ == "__main__":
    main()
