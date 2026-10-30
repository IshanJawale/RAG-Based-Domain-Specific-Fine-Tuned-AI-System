import os
import re
from pathlib import Path

# ======================================================================
# Shared prompt helpers
# ======================================================================

def _format_chunk(i: int, chunk: dict) -> str:
    source_label = chunk.get("paper_title", chunk.get("case_id", "Unknown Document"))
    page_label   = chunk.get("page", chunk.get("section_number", "N/A"))
    if chunk.get("retrieval_type") == "web":
        source_label = chunk.get("paper_title", "Web Source")
        page_label   = chunk.get("url", "N/A")
    return (
        f"EVIDENCE {i}\n"
        f"Source       : {source_label}\n"
        f"Section/Page : {page_label}\n"
        f"Content:\n{chunk.get('text', '')}\n"
    )

def _build_context(chunks: list[dict]) -> str:
    if not chunks:
        return "No external evidence provided."
    return "\n---\n".join(_format_chunk(i, c) for i, c in enumerate(chunks, start=1))

_MD_MARKUP = re.compile(r"^\s*#{1,6}\s*|^\s*(?:[-*+]|\d+\.)\s+|\*\*|__|`", re.MULTILINE)


def format_history(history: list[dict] | None, max_turns: int = 3, max_chars: int = 500) -> str:
    """Last few exchanges as plain text for the prompt (assistant replies truncated).

    Markdown is stripped before the reply is flattened to one line: otherwise "### Summary\\n\\nIf you..."
    becomes "### Summary If you...", and a model that echoes it renders a whole paragraph as a heading."""
    if not history:
        return ""
    lines = []
    for m in history[-2 * max_turns:]:
        text = " ".join(_MD_MARKUP.sub("", str(m.get("content", ""))).split())
        if m.get("role") == "assistant" and len(text) > max_chars:
            text = text[:max_chars].rsplit(" ", 1)[0] + " ..."
        lines.append(f"{'User' if m.get('role') == 'user' else 'Assistant'}: {text}")
    return (
        "CONVERSATION SO FAR (use it to resolve follow-up questions such as 'what about...' or 'and the punishment?'):\n"
        + "\n".join(lines)
        + "\n\nAnswer only the latest USER QUESTION below from the evidence. Do not repeat or summarise your earlier answers.\n\n"
    )


_LEGAL_PROMPT_TEMPLATE = """\
You are an expert Indian Legal Assistant. Your task is to answer the user's question accurately using ONLY the provided evidence from Indian Acts, Statutes, and Supreme Court or High Court precedents.

IMPORTANT RULES & FORMATTING:
1. Do not invent legal facts, laws, or use outside knowledge.
2. If the evidence does not contain enough information to answer, explicitly state: "The available legal evidence is insufficient to answer this question."
3. FORMAT YOUR ANSWER FOR MAXIMUM READABILITY:
   - Use clear markdown headings (e.g. `### Summary`, `### Key Provisions`, `### Legal Analysis`).
   - Break down dense explanations into clear **bullet points** or **numbered lists**.
   - NEVER output a giant wall of continuous unbroken text.
   - Separate every distinct point or paragraph with double line breaks.
4. You must cite your claims with the provided Source and Section/Page (e.g., "**According to Section 110 of Bharatiya Nyaya Sanhita (BNS)...**" or "**In [Case Name]...**").
5. If multiple laws or cases provide evidence, synthesize them clearly under appropriate headings.
6. At the end of your answer, include a `### 📜 Relied Statutes & Authorities` section listing all cited sections.

{history}USER QUESTION:
{question}

RETRIEVED LEGAL EVIDENCE:
{context}

Provide a well-structured, clean, bulleted, and grounded legal answer:
"""

_GENERAL_KNOWLEDGE_PROMPT_TEMPLATE = """\
You are an intelligent, authoritative AI Assistant. The user is asking a general knowledge, educational, historical, scientific, or cultural question.

IMPORTANT GUIDELINES:
1. Provide a comprehensive, clear, and well-structured answer with clean markdown headings and bullet points.
2. If web or factual evidence is provided below, incorporate the relevant facts smoothly.
3. If the question asks about ancient epics, literature, or mythology (e.g., the Mahabharata, Ramayana, Gita), provide accurate details regarding the narrative, main characters, themes, and all 18 Parvas (books) without repeating or inventing parva names.
4. If there is a meaningful connection or intersection with Indian law or the Supreme Court (e.g., the Supreme Court motto "Yato Dharmastato Jayah"), mention it as an insightful note.
5. NEVER say "The available legal evidence is insufficient" for general knowledge questions. Answer directly and factually.

{history}USER QUESTION:
{question}

RELEVANT FACTUAL EVIDENCE:
{context}

Provide a well-structured, informative, and engaging answer:
"""

_SECTION_NAMES = (r"Summary|Key Provisions|Legal Analysis|Relied Statutes (?:&|and) Authorities|Overview|Conclusion"
                  r"|Practical Steps|Next Steps|Remedies|Procedure|Legal Position|Answer")
_RUN_ON_HEADING = re.compile(rf"^(#{{1,6}}[ \t]*(?:\W[ \t]*)?(?:{_SECTION_NAMES}))[ \t:]+(?=\S[^\n]{{60,}})",
                             re.MULTILINE | re.IGNORECASE)  # only a long run-on line, not "### Summary of the law"
_MID_LINE_HEADING = re.compile(r"(?<=\S)[ \t]+(#{2,6}[ \t])")
_INLINE_LIST_ITEM = re.compile(r"(?<=[.:])[ \t]+(\d{1,2}\.[ \t]+\*\*)")


def tidy_markdown(text: str) -> str:
    """Repair answers whose line breaks were lost ("### Summary If you have ... ### Key Provisions 1. **...**"),
    which Streamlit would render as one huge heading. Well-formed markdown is left unchanged."""
    text = _MID_LINE_HEADING.sub(r"\n\n\1", text)
    text = _RUN_ON_HEADING.sub(r"\1\n\n", text)
    return _INLINE_LIST_ITEM.sub(r"\n\1", text)


def build_prompt(question: str, retrieved_chunks: list[dict], intent: str = "STATUTE", source: str = "LOCAL", history: list[dict] | None = None) -> str:
    template = _GENERAL_KNOWLEDGE_PROMPT_TEMPLATE if (intent == "GENERAL_KNOWLEDGE" or source == "WEB") else _LEGAL_PROMPT_TEMPLATE
    return template.format(question=question, context=_build_context(retrieved_chunks), history=format_history(history))

# ======================================================================
# Text-only generation
# ======================================================================

def _generate_text_qwen(prompt: str, model_name: str = None) -> str:
    from src.llm import chat
    return chat(prompt, model_name=model_name, max_new_tokens=1500)

def generate_answer(
    question: str,
    retrieved_chunks: list[dict],
    model_name: str = None,
    intent: str = "STATUTE",
    source: str = "LOCAL",
    history: list[dict] | None = None,
) -> str:
    """Generate a grounded legal or general knowledge answer from retrieved chunks using the selected model."""
    prompt = build_prompt(question, retrieved_chunks, intent=intent, source=source, history=history)
    try:
        return _generate_text_qwen(prompt, model_name=model_name)
    except Exception as e:
        return f"[Generation error] {e}"


def generate_answer_stream(
    question: str,
    retrieved_chunks: list[dict],
    model_name: str = None,
    intent: str = "STATUTE",
    source: str = "LOCAL",
    history: list[dict] | None = None,
):
    """Same as generate_answer, but yields text chunks as they are produced."""
    from src.llm import chat_stream
    prompt = build_prompt(question, retrieved_chunks, intent=intent, source=source, history=history)
    try:
        yield from chat_stream(prompt, model_name=model_name, max_new_tokens=1500)
    except Exception as e:
        yield f"\n\n[Generation error] {e}"