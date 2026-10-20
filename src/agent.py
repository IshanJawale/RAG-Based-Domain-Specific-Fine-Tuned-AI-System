"""
agent.py - Autonomous Legal Agent Runtime for Indian Law.

Orchestrates multi-turn reasoning and tool invocation:
  1. Inspects user intent & legal query
  2. Selects appropriate tool:
       - create_legal_document (Legal Notice, Legal Opinion, Case Brief -> .docx)
       - calculate (Sec 34 CPC interest, damages, limitation timeline)
       - search_acts (99K statutory sections)
       - search_cases (5.5M court precedents)
       - web_search (live legal gazettes / news)
  3. Executes tool outside model tokens (clean separation of reasoning vs rendering)
  4. Returns final structured response with file downloads & legal analysis.
"""

import json
import re
from pathlib import Path
from typing import Any, Optional

from src.llm import generate
from src.tools import TOOL_DEFINITIONS, execute_tool

TOOL_SYSTEM_PROMPT = """You are an Autonomous Indian Legal Agent.
You have access to specialized legal tools:

1. `create_legal_document`: Draft formal Legal Notices, Legal Opinions, Case Briefs, or Legal Assessment Reports exported directly as Word (.docx) files.
2. `calculate`: Calculate legal damages, interest under Section 34 CPC, limitation periods, or court fees.
3. `search_acts`: Look up specific bare acts and statutory provisions.
4. `search_cases`: Look up Supreme Court and High Court precedents.
5. `web_search`: Search live web for current legal updates.

INSTRUCTIONS FOR TOOL USAGE:
- You must ONLY call `create_legal_document` if the user EXPLICITLY asks to draft, write, prepare, generate, create, or export a formal document, notice, legal opinion, contract, or report into a downloadable file (.docx).
- For all conceptual, explanatory, statutory questions, legal advice, or general knowledge queries, DO NOT call `create_legal_document`. Answer directly in clear text.
- Supported document_type values: "legal_notice", "legal_opinion", "case_brief", "legal_report".
- If the user asks for a financial or mathematical calculation (e.g. interest rate under Sec 34 CPC, compensation, total claim), call `calculate`.
- When calling a tool, return ONLY a valid JSON object in a ```json markdown block:
```json
{
  "tool": "create_legal_document",
  "arguments": {
    "document_type": "legal_notice",
    "title": "Legal Notice for Breach of Contract",
    "sender_name": "ABC Enterprises",
    "sender_address": "Connaught Place, New Delhi",
    "recipient_name": "XYZ Corp Ltd",
    "recipient_address": "Bandra Kurla Complex, Mumbai",
    "subject": "Notice of demand for unpaid dues under Section 73 Indian Contract Act",
    "facts": [
      "On 15 January 2024, the parties entered into a Master Services Agreement.",
      "The client successfully delivered all contractual milestones on 30 March 2024.",
      "Invoice No. 2024-089 for INR 15,00,000 remains unpaid despite repeated reminders."
    ],
    "legal_provisions": [
      "Section 73 of the Indian Contract Act, 1872 (Compensation for loss or damage caused by breach)",
      "Section 34 of the Code of Civil Procedure, 1908 (Interest pendente lite and future interest at 18% p.a.)"
    ],
    "claims": [
      "Immediate remittance of principal sum of INR 15,00,000.",
      "Interest at the rate of 18% per annum from due date till payment.",
      "Cost of this legal notice quantified at INR 25,000."
    ],
    "demand_period_days": 15,
    "advocate_name": "Advocate Ramesh Sharma, High Court of Delhi"
  }
}
```

If no tool is required, answer the user's question directly with clear legal structure, statutory sections, and case citations.
"""


# Flexible intent regex to trigger document/report drafting when requested
DRAFT_INTENT_PATTERN = re.compile(
    r"\b("
    r"(?:draft|prepare|generate|create|write|export|compose|formulate|make|compile|build|provide)\s+(?:(?:\S+\s+){0,6})?"
    r"(?:notice|opinion|case\s+brief|brief|report|assessment|document|memo|memorandum|contract|agreement|petition|affidavit|plaint|deed|settlement|docx|word)"
    r"|export\s+(?:to\s+)?(?:docx|word|document|report)"
    r"|(?:legal\s+notice|legal\s+opinion|legal\s+report|assessment\s+report|case\s+brief|word\s+document|downloadable\s+report|\.docx)"
    r")\b",
    re.IGNORECASE,
)

CALC_INTENT_PATTERN = re.compile(
    r"\b(calculate|compute|work\s+out|how\s+much\s+(?:interest|compensation|damages|total))\b"
    r"|\binterest\b[^.?!]*\d|\d[^.?!]*\b(?:per\s+annum|p\.a\.|% interest)",
    re.IGNORECASE,
)

DOC_TYPES = ("legal_notice", "legal_opinion", "case_brief", "legal_report")


def is_drafting_request(query: str) -> bool:
    return bool(DRAFT_INTENT_PATTERN.search(query))


def is_calculation_request(query: str) -> bool:
    return bool(CALC_INTENT_PATTERN.search(query))


def infer_document_type(query: str) -> str:
    """Document type the user explicitly asked for (defaults to a legal report)."""
    q = query.lower()
    if "notice" in q:
        return "legal_notice"
    if "opinion" in q or "memo" in q:
        return "legal_opinion"
    if "case brief" in q or re.search(r"\bbrief\b", q):
        return "case_brief"
    return "legal_report"


def _try_repair_and_load(s: str) -> Optional[dict]:
    """Attempt to parse JSON, auto-repairing unclosed quotes or brackets if truncated."""
    s = s.strip()
    try:
        return json.loads(s)
    except Exception:
        pass

    # Basic repair
    cleaned = s
    # Remove dangling commas before closing brackets
    cleaned = re.sub(r",\s*([\]}])", r"\1", cleaned)

    # Balance quotes
    if cleaned.count('"') % 2 != 0:
        cleaned += '"'

    # Balance brackets and braces
    open_brackets = cleaned.count('[') - cleaned.count(']')
    if open_brackets > 0:
        cleaned += ']' * open_brackets

    open_braces = cleaned.count('{') - cleaned.count('}')
    if open_braces > 0:
        cleaned += '}' * open_braces

    try:
        return json.loads(cleaned)
    except Exception:
        return None


def extract_tool_call(response_text: str) -> Optional[dict]:
    """Parse tool call JSON from LLM response text, supporting multiple formats and repair."""
    # 1. Look for ```json ... ``` blocks
    fenced_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)(?:```|$)", response_text, re.IGNORECASE)
    for block in fenced_blocks:
        parsed = _try_repair_and_load(block.strip())
        if isinstance(parsed, dict):
            if "tool" in parsed:
                return parsed
            if "name" in parsed:
                return {"tool": parsed["name"], "arguments": parsed.get("arguments", {})}
            if "document_type" in parsed:
                return {"tool": "create_legal_document", "arguments": parsed}

    # 2. Look for bare JSON with "tool", "name", or "document_type"
    for candidate in re.finditer(r"(\{[\s\S]*)", response_text):
        parsed = _try_repair_and_load(candidate.group(1).strip())
        if isinstance(parsed, dict):
            if "tool" in parsed:
                return parsed
            if "name" in parsed:
                return {"tool": parsed["name"], "arguments": parsed.get("arguments", {})}
            if "document_type" in parsed:
                return {"tool": "create_legal_document", "arguments": parsed}

    return None


def _result(response, tool_used=None, tool_result=None, document_path=None, document_type=None, error=None) -> dict:
    return {
        "response": response,
        "tool_used": tool_used,
        "tool_result": tool_result,
        "document_path": document_path,
        "document_type": document_type,
        "error": error,
    }


def _export_document(targs: dict, analysis_text: str, user_query: str, tool_label: str) -> dict:
    """Fill missing fields from the analysis, write the .docx and report success or the failure reason."""
    targs = _enrich_doc_args_from_text(targs, analysis_text, user_query)
    exec_res = execute_tool("create_legal_document", targs)
    inner = exec_res.get("result") or {}
    file_path = inner.get("file_path")
    if exec_res.get("status") != "success" or inner.get("status") == "error" or not file_path:
        err = exec_res.get("message") or inner.get("message") or "the document generator returned no file"
        print(f"[Agent] Document export failed: {err}")
        return _result(analysis_text, tool_label, exec_res, error=err)
    return _result(analysis_text, tool_label, exec_res, file_path, inner.get("document_type"))


def run_legal_agent(
    user_query: str,
    context_chunks: Optional[list[dict]] = None,
    model_name: Optional[str] = None,
) -> dict:
    """
    Run the legal agent workflow on a user query.
    Returns:
      {
        "response": str (final text / summary),
        "tool_used": Optional[str],
        "tool_result": Optional[Any],
        "document_path": Optional[str],
        "document_type": Optional[str],
        "error": Optional[str] (set when a requested document could not be produced),
      }
    """
    is_drafting = is_drafting_request(user_query)
    requested_type = infer_document_type(user_query)

    prompt = f"{TOOL_SYSTEM_PROMPT}\n\n"
    if context_chunks:
        has_web = any(c.get("retrieval_type") == "web" for c in context_chunks)
        ev_title = "RELEVANT FACTUAL / WEB EVIDENCE" if has_web else "CONTEXT EVIDENCE FROM INDIAN LEGAL REPOSITORY"
        prompt += f"{ev_title} (Use this to formulate your response):\n"
        for i, c in enumerate(context_chunks[:7], 1):
            title = c.get("paper_title") or c.get("act_title") or c.get("case_id", "Evidence")
            sec = c.get("page") or c.get("section_number", "")
            text = c.get("text", "")[:700]
            prompt += f"[{i}] {title} ({sec}):\n{text}\n\n"

    prompt += f"USER REQUEST:\n{user_query}\n\n"
    if is_drafting:
        prompt += (
            "CRITICAL INSTRUCTION: The user explicitly requested a formal legal document/report. "
            "You MUST invoke the `create_legal_document` tool with a complete, structured JSON schema in a ```json code block. "
            f"Set 'document_type' to '{requested_type}'. Do not output only plain text.\n"
        )
    else:
        prompt += (
            "INSTRUCTION: Answer the user's question directly with clear structure and authoritative explanation. "
            "DO NOT invoke `create_legal_document` because the user did not ask to draft or export a document.\n"
        )

    raw_response = generate(prompt, model_name=model_name, max_new_tokens=1500, temperature=0.1)

    tool_call = extract_tool_call(raw_response)
    tname = tool_call.get("tool") if tool_call else None
    targs = (tool_call.get("arguments") or {}) if tool_call else {}
    is_doc_tool = tname in ("create_legal_document", "create_legal_notice", "create_legal_opinion", "create_case_brief", "create_legal_report")

    # --- Document requests -------------------------------------------------
    if is_drafting:
        if is_doc_tool and isinstance(targs, dict):
            targs = dict(targs)
            # The user's explicit wording wins over the model's choice of template
            if requested_type != "legal_report" or targs.get("document_type") not in DOC_TYPES:
                targs["document_type"] = requested_type
            doc_label = targs["document_type"].replace("_", " ").title()
            summary_prompt = (
                f"User Request: {user_query}\n\n"
                f"Structured Case Parameters:\n{json.dumps(targs, indent=2)}\n\n"
                "Provide a thorough, well-formatted legal analysis in Markdown covering:\n"
                "1. **Introduction & Factual Matrix**\n"
                "2. **Relevant Statutory Provisions** (Indian Contract Act, Arbitration & Conciliation Act, Evidence Laws, etc.)\n"
                "3. **Detailed Legal Analysis of Each Issue & Party Arguments**\n"
                "4. **Conclusion, Likely Outcome & Strategic Remedies**\n\n"
                f"Conclude with a brief note that the formal {doc_label} (.docx) has been generated and is ready for download."
            )
            final_text = generate(summary_prompt, model_name=model_name, max_new_tokens=1000, temperature=0.2)
            return _export_document(targs, final_text, user_query, tname)

        # Model answered in prose (or called an unrelated tool): compile the draft text into the document
        draft_text = raw_response
        if tool_call or not raw_response.strip():
            draft_text = generate(
                f"User Request: {user_query}\n\nWrite the complete {requested_type.replace('_', ' ')} in Markdown with "
                "headings for the facts, applicable statutory provisions, legal analysis, claims or recommendations, "
                "and conclusion. Output the document text only.",
                model_name=model_name, max_new_tokens=1200, temperature=0.2,
            )
        return _export_document({"document_type": requested_type}, draft_text, user_query, "create_legal_document (Auto-Compiled)")

    # --- Non-document requests ----------------------------------------------
    if tool_call and tname:
        if is_doc_tool:
            # Guardrail: User never asked for a document file. Suppress unrequested document creation!
            print(f"[Agent] Suppressed unrequested document generation for query: '{user_query}'")
            direct_prompt = (
                f"User Question: {user_query}\n\n"
                f"Context Evidence:\n{raw_response[:800]}\n\n"
                f"Provide a direct, thorough, and well-structured answer explaining the principles, "
                f"statutes, or facts directly in text. Do NOT create or refer to any external file or document."
            )
            return _result(generate(direct_prompt, model_name=model_name, max_new_tokens=900, temperature=0.2))

        exec_res = execute_tool(tname, targs)

        if tname == "calculate":
            calc_val = exec_res.get("result") or exec_res.get("message", "")
            final_text = (
                f"### 🧮 Legal Calculation Result\n\n"
                f"- **Expression Evaluated:** `{targs.get('expression', '')}`\n"
                f"- **{calc_val}**\n\n"
                f"{raw_response.replace(json.dumps(tool_call), '').strip()}"
            )
            return _result(final_text, "calculate", exec_res)

        # RAG / Search tool: feed observation back for synthesis
        obs = exec_res.get("result") or exec_res.get("message", "")
        synthesis_prompt = (
            f"Original User Question: {user_query}\n"
            f"Tool Invoked: {tname}\n"
            f"Tool Observation / Retrieved Legal Data:\n{obs}\n\n"
            "Synthesize a rigorous legal answer based on these findings, citing the specific statutes and precedents."
        )
        final_text = generate(synthesis_prompt, model_name=model_name, max_new_tokens=700, temperature=0.2)
        return _result(final_text, tname, exec_res)

    return _result(raw_response)


def _enrich_doc_args_from_text(targs: dict, analysis_text: str, user_query: str) -> dict:
    """Ensure all .docx document fields are richly populated from the generated legal analysis."""
    enriched = dict(targs)
    clean_blocks = [
        re.sub(r"^#+\s*", "", p).strip()
        for p in re.split(r"\n\s*\n", analysis_text)
        if len(p.strip()) > 30
    ]

    if not enriched.get("title"):
        m_title = re.search(r"(?:Title|Subject|Report on)[:\s]*([^\n*#]+)", analysis_text, re.IGNORECASE)
        if m_title:
            enriched["title"] = m_title.group(1).strip()
        else:
            first_line = user_query.strip().split("\n")[0][:85].strip(" .:")
            enriched["title"] = f"Legal Assessment: {first_line}" if len(first_line) > 10 else "Legal Assessment Report"

    if not enriched.get("subject"):
        enriched["subject"] = user_query.strip().split("\n")[0][:120]

    if not enriched.get("executive_summary") and clean_blocks:
        enriched["executive_summary"] = clean_blocks[0]

    if not enriched.get("scenarios") and not enriched.get("facts"):
        mid_blocks = clean_blocks[1:-1] if len(clean_blocks) > 2 else clean_blocks
        enriched["scenarios"] = mid_blocks[:8]
        enriched["facts"] = mid_blocks[:6]

    if not enriched.get("statutory_provisions") and not enriched.get("legal_provisions"):
        extracted_statutes = []
        for m_act in re.finditer(
            r"((?:Section\s+\d+[A-Za-z0-9\(\)]*\s+of\s+the\s+)?[A-Z][A-Za-z\s,&]+(?:Act|Sanhita|Adhiniyam|Code)(?:,\s*\d{4})?)",
            analysis_text,
        ):
            st_str = re.sub(r"\s+", " ", m_act.group(1)).strip(" ,.-*")
            if 10 <= len(st_str) <= 120 and st_str not in extracted_statutes:
                extracted_statutes.append(st_str)
        if not extracted_statutes:
            extracted_statutes = [
                "Indian Contract Act, 1872 (Breach of Contract & Contractual Damages under Sections 73–75)",
                "Arbitration and Conciliation Act, 1996 (Arbitral Jurisdiction & Dispute Resolution)",
                "Bharatiya Sakshya Adhiniyam, 2023 / Indian Evidence Act, 1872 (Admissibility of Electronic Records)",
                "Code of Civil Procedure, 1908 (Section 34 Interest & Civil Remedies)",
            ]
        enriched["statutory_provisions"] = extracted_statutes[:8]
        enriched["legal_provisions"] = extracted_statutes[:8]

    if not enriched.get("claims") and str(enriched.get("document_type", "")).endswith("notice"):
        claims = [
            re.sub(r"^[\s*\-\d.)]+", "", ln).strip()
            for ln in analysis_text.splitlines()
            if re.search(r"\b(demand|pay|remit|refund|compensat|relief|interest|cost)", ln, re.IGNORECASE)
            and len(ln.strip()) > 20
        ]
        if claims:
            enriched["claims"] = claims[:6]

    if not enriched.get("conclusion") and len(clean_blocks) > 1:
        enriched["conclusion"] = clean_blocks[-1]

    return enriched
