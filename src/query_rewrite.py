"""
query_rewrite.py

Rewrites plain-language legal questions into the vocabulary of Indian statutes and judgments before retrieval.

"I have a land dispute with my brother over family land" shares almost no words with the Hindu Succession Act
or a partition judgment, so BGE-small and the cross-encoder rank them low (top rerank score -2.99, below the
evidence gate). Rewritten as "partition of joint family property between brothers, coparcenary share, Hindu
Succession Act", the same corpus scores +5 to +7. Questions that already cite a section, Article, Act or case
are searched as asked: retrieval handles those well and a rewrite could only drift from them.
"""

import re
from functools import lru_cache

# Already in legal vocabulary: a provision number, a named statute or a case citation
# ("tell me the acts and sections related to this" is not: it names no provision)
_ALREADY_LEGAL = re.compile(
    r"(?i:\b(?:sections?|sec\.?|ss?\.|articles?|art\.)\s*\d+|\border\s+[ivxl]+\b|\brule\s+\d+)"
    r"|\b(?:BNS|BNSS|BSA|IPC|CrPC|CPC|NI Act|RTI Act|POCSO|PMLA)\b"
    r"|\b[A-Z][A-Za-z]+(?:\s+(?:of|and|&|[A-Z][A-Za-z]+))*\s+(?:Act|Code|Sanhita|Adhiniyam)\b"
    r"|\bConstitution\b"
    r"|\s(?i:v\.|vs\.?|versus)\s"
)

# Conversational wrapping that dilutes the embedding and the cross-encoder's query
_FILLER = re.compile(
    r"\b(?:please|kindly|tell me|can you|could you|i want to know|i would like to know|explain to me"
    r"|what all can help me(?: with it)?|what (?:all )?can i do|what should i do|what to do"
    r"|help me(?: with it)?)\b",
    re.IGNORECASE,
)

_PROMPT = """\
You turn a citizen's question into a search query for a database of Indian Acts, Supreme Court judgments and High Court judgments.

Write the query in the words a lawyer or a judgment would use: name the legal concepts (for example partition, coparcenary property, co-sharer, injunction, maintenance, cruelty, dishonour of cheque, anticipatory bail) and the Indian Acts that govern the situation. Name a section only if you are sure of it. Keep the facts that matter (who the parties are, what kind of property or offence). Use the new criminal laws (BNS, BNSS, BSA) for crimes and criminal procedure.

Example
Question: my landlord is not returning my deposit after I vacated the flat
Query: recovery of security deposit from landlord after tenant vacated, lease, suit for recovery of money, Transfer of Property Act 1882 Section 108

Question: {question}
Query:"""


def strip_filler(question: str) -> str:
    """The question without conversational wrapping ("tell me what to do if...")."""
    text = _FILLER.sub(" ", question)
    text = re.sub(r"\s+", " ", text).strip(" .?,;")
    return text if len(text) >= 12 else question


def needs_rewrite(question: str) -> bool:
    return not _ALREADY_LEGAL.search(question)


@lru_cache(maxsize=256)
def _llm_rewrite(question: str) -> str:
    from src.routing import _call_llm  # Gemini when a key is set, otherwise the active local model
    raw = _call_llm(_PROMPT.format(question=question))
    line = next((ln for ln in raw.strip().splitlines() if ln.strip()), "")
    line = re.sub(r"^\s*(?:query|search query)\s*:\s*", "", line, flags=re.I).strip(" \"'`")
    return line[:300]


def rewrite_for_retrieval(question: str) -> str:
    """Search text for the legal corpus: an LLM rewrite in legal terms for plain-language questions,
    the question itself when it already cites the law. Falls back to the de-cluttered question."""
    if not needs_rewrite(question):
        return question
    plain = strip_filler(question)
    try:
        rewritten = _llm_rewrite(plain)
    except Exception as e:
        print(f"[Rewrite] failed ({e}), searching with the original question.")
        return plain
    return rewritten if len(rewritten) >= 12 else plain
