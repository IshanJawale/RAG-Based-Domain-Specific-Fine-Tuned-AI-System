"""
tools.py - Agentic Tool Registry for Indian Legal AI.

Equips the LLM with autonomous tools:
  1. search_acts(query)       - Precise retrieval over 99K Central & State Act sections
  2. search_cases(query)      - Precedent search over 5.5M Supreme Court & High Court judgments
  3. calculate(expression)    - Safe legal calculator (damages, interest under Sec 34 CPC, limitation periods)
  4. web_search(query)        - Live search for current legal developments & notifications
  5. create_legal_notice(...) - Generates formal Indian Legal Notice (.docx)
  6. create_legal_opinion(...)- Generates formal Legal Opinion / Memorandum (.docx)
  7. create_case_brief(...)   - Generates formal Judicial Case Brief (.docx)
"""

import json
import math
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from src.document_generator import (
    create_legal_notice,
    create_legal_opinion,
    create_case_brief,
    generate_legal_document,
)
from src.legal_formats import ALL_FORMATS

# Global lazy retriever reference
_retriever = None
_web_enabled = True

def set_retriever(retriever):
    """Share an already-loaded retriever so tools don't load a second embedding model."""
    global _retriever
    _retriever = retriever

def set_web_enabled(enabled: bool):
    """Follow the app's web search setting, so the agent cannot reach the web when it is off."""
    global _web_enabled
    _web_enabled = enabled

def _get_retriever():
    global _retriever
    if _retriever is None:
        try:
            from src.retrieval import HybridRetriever
            _retriever = HybridRetriever()
        except Exception as e:
            print(f"[Tools] Warning: Could not initialize retriever: {e}")
    return _retriever


# ======================================================================
# Tool Implementations
# ======================================================================

def search_acts(query: str, top_k: int = 4) -> str:
    """
    Search statutory bare acts (IPC, CrPC, CPC, BNS, Companies Act, NI Act, etc.)
    in LanceDB for specific sections, definitions, or procedural provisions.
    """
    retriever = _get_retriever()
    if not retriever:
        return "LanceDB Acts retriever is currently unavailable."
    
    try:
        results = retriever.search_acts(query, top_k=top_k)
        if not results:
            return f"No statutory sections found matching '{query}' in bare acts database."
        
        output = [f"Found {len(results)} statutory sections:"]
        for i, r in enumerate(results, 1):
            title = r.get("paper_title", "Act")
            sec = r.get("page", "Section")
            text = r.get("text", "").strip()[:500]
            output.append(f"\n[{i}] {title} ({sec}):\n{text}")
        return "\n".join(output)
    except Exception as e:
        return f"Error querying bare acts: {e}"


def search_cases(query: str, top_k: int = 4) -> str:
    """
    Search Indian judicial precedents (Supreme Court & High Courts) in LanceDB
    for case law, judge opinions, and legal ratio decidendi.
    """
    retriever = _get_retriever()
    if not retriever:
        return "LanceDB Court retriever is currently unavailable."

    try:
        results = retriever.search_cases(query, top_k=top_k)
        if not results:
            return f"No court precedents found matching '{query}'."

        output = [f"Found {len(results)} judicial precedents:"]
        for i, r in enumerate(results, 1):
            case_id = r.get("paper_title", "Precedent")
            court = r.get("page", "Court")
            text = r.get("text", "").strip()[:500]
            output.append(f"\n[{i}] {case_id} [{court}]:\n{text}")
        return "\n".join(output)
    except Exception as e:
        return f"Error querying case law: {e}"


def calculate(expression: str) -> str:
    """
    Safely evaluate mathematical expressions for legal calculations:
    damages, compound/simple interest (e.g. Sec 34 CPC, Sec 80 GST), limitation periods, court fees.
    Examples:
      - '2500000 * 0.18 * (3.5)' -> Interest calculation
      - '1500000 + (1500000 * 0.12 * 2)' -> Total claim with interest
    """
    # Whitelist allowed characters and math functions
    allowed_pattern = r"^[0-9+\-*/().,%\s_eE]|math\.(sqrt|pow|exp|log|floor|ceil)"
    cleaned = expression.strip()
    
    # Strip dangerous builtins
    if any(k in cleaned for k in ["import", "__", "eval", "exec", "open", "os", "sys", "lambda"]):
        return "Error: Unsupported or unsafe calculation expression."
    
    try:
        safe_env = {
            "math": math,
            "abs": abs,
            "round": round,
            "min": min,
            "max": max,
        }
        res = eval(cleaned, {"__builtins__": None}, safe_env)
        if isinstance(res, (int, float)):
            return f"Calculation Result: {res:,.2f}" if isinstance(res, float) else f"Calculation Result: {res:,}"
        return f"Calculation Result: {res}"
    except Exception as e:
        return f"Calculation Error: {e}"


def web_search(query: str) -> str:
    """Search live web for recent legal notifications, amendments, or current news."""
    if not _web_enabled:
        return "Web search is turned off; answer from the local legal corpus."
    try:
        from src.web_search import TavilySearcher
        searcher = TavilySearcher()
        results = searcher.search(query, max_results=4, india=True)
        if not results:
            return f"No web search results for '{query}'."
        output = [f"Web Search Results for '{query}':"]
        for i, r in enumerate(results, 1):
            title = r.get("paper_title", "Web Page")
            snippet = r.get("text", "")[:400]
            output.append(f"[{i}] {title}\n{snippet}\n")
        return "\n".join(output)
    except Exception as e:
        return f"Web search failed: {e}"


def tool_create_legal_document(document_type: str, **kwargs) -> dict:
    """
    Generate formal legal document (.docx) and return file path.
    Supported types: 'legal_notice', 'legal_opinion', 'case_brief'.
    """
    return generate_legal_document(document_type=document_type, **kwargs)


# ======================================================================
# Tool Specifications (JSON Schema for LLM function calling)
# ======================================================================

TOOL_DEFINITIONS = [
    {
        "name": "search_acts",
        "description": "Search statutory bare acts (IPC, CrPC, CPC, BNS, Companies Act, NI Act, etc.) for sections, provisions, and definitions.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The legal concept, act, or section to search (e.g. 'Section 138 NI Act notice period')"},
                "top_k": {"type": "integer", "description": "Number of sections to retrieve (default 4)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_cases",
        "description": "Search Indian Supreme Court and High Court precedents and judgments for case ratios, precedents, and legal principles.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The case topic or legal question (e.g. 'quashing Section 482 CrPC commercial dispute Supreme Court')"},
                "top_k": {"type": "integer", "description": "Number of precedents to retrieve (default 4)"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "calculate",
        "description": "Perform mathematical and financial calculations for legal damages, simple/compound interest under Sec 34 CPC, limitation days, or court fees.",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Mathematical expression to evaluate (e.g. '1500000 * 0.18 * 2.5')"}
            },
            "required": ["expression"]
        }
    },
    {
        "name": "web_search",
        "description": "Query the live web for the latest notifications, recent gazette amendments, or current legal news.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The search query (e.g. 'latest Supreme Court ruling on arbitration stamp duty 2024')"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "create_legal_document",
        "description": "Generate a formal Indian legal document in Word (.docx) format. Use this when the user asks to draft a notice (general, Sec 138 NI Act, Sec 80 CPC), plaint, written statement, bail or anticipatory bail application, police complaint, writ petition, consumer complaint, RTI application, affidavit, legal opinion, case brief or report. Format-specific fields are listed in src/legal_formats.py.",
        "parameters": {
            "type": "object",
            "properties": {
                "document_type": {
                    "type": "string",
                    "enum": list(ALL_FORMATS),
                    "description": "The type of document to generate"
                },
                "title": {"type": "string", "description": "Title or subject of the document"},
                "sender_name": {"type": "string", "description": "Name of the aggrieved party / client sending notice"},
                "sender_address": {"type": "string", "description": "Address of sender"},
                "recipient_name": {"type": "string", "description": "Name of the opposite party / recipient"},
                "recipient_address": {"type": "string", "description": "Address of recipient"},
                "subject": {"type": "string", "description": "Subject line of notice, opinion, or report"},
                "facts": {"type": "array", "items": {"type": "string"}, "description": "Numbered chronological factual statements"},
                "scenarios": {"type": "array", "description": "List of factual scenarios or case studies for legal reports"},
                "executive_summary": {"type": "string", "description": "Executive summary for legal report"},
                "legal_provisions": {"type": "array", "items": {"type": "string"}, "description": "Applicable statutory sections violated or analyzed"},
                "statutory_provisions": {"type": "array", "items": {"type": "string"}, "description": "Statutes and acts governing the scenario"},
                "claims": {"type": "array", "items": {"type": "string"}, "description": "Specific remedies, monetary demands, or requisitions demanded"},
                "recommendations": {"type": "array", "items": {"type": "string"}, "description": "Preventive remedies and legal solutions"},
                "demand_period_days": {"type": "integer", "description": "Notice period to comply (default 15 days)"},
                "advocate_name": {"type": "string", "description": "Name and credentials of advocate issuing notice"},
                "client_name": {"type": "string", "description": "Client name for legal opinion"},
                "query_issue": {"type": "string", "description": "The central legal issue to address in legal opinion"},
                "applicable_statutes": {"type": "array", "items": {"type": "object"}, "description": "Statutes and sections applicable for legal opinion"},
                "judicial_precedents": {"type": "array", "description": "Relevant case laws and citations for opinion/brief/report"},
                "legal_analysis": {"type": "string", "description": "Detailed legal reasoning, doctrine application, and argument synthesis"},
                "conclusion_advice": {"type": "string", "description": "Final opinion, strategic advice, and recommendations for the client"},
                "case_title": {"type": "string", "description": "Case title for case brief (e.g. 'Maneka Gandhi v. Union of India')"},
                "citation": {"type": "string", "description": "Citation for case brief (e.g. 'AIR 1978 SC 597')"},
                "court_bench": {"type": "string", "description": "Bench and court name for case brief"},
                "issues": {"type": "array", "items": {"type": "string"}, "description": "Legal questions framed in case brief"},
                "ratio_decidendi": {"type": "string", "description": "Core legal principle / holding of court"}
            },
            "required": ["document_type", "title"]
        }
    }
]


# ======================================================================
# Tool Dispatcher
# ======================================================================

def execute_tool(tool_name: str, arguments: dict) -> dict:
    """Execute a tool by name with arguments and return result."""
    tname = tool_name.strip().lower()
    try:
        if tname == "search_acts":
            res = search_acts(query=arguments.get("query", ""), top_k=arguments.get("top_k", 4))
            return {"status": "success", "tool": "search_acts", "result": res}
        
        elif tname == "search_cases":
            res = search_cases(query=arguments.get("query", ""), top_k=arguments.get("top_k", 4))
            return {"status": "success", "tool": "search_cases", "result": res}
        
        elif tname == "calculate":
            res = calculate(expression=arguments.get("expression", ""))
            return {"status": "success", "tool": "calculate", "result": res}
        
        elif tname == "web_search":
            res = web_search(query=arguments.get("query", ""))
            return {"status": "success", "tool": "web_search", "result": res}
        
        elif tname in ("create_legal_document", "create_legal_notice", "create_legal_opinion", "create_case_brief"):
            if "document_type" not in arguments:
                arguments["document_type"] = tname.replace("create_", "")
            res = tool_create_legal_document(**arguments)
            return {"status": "success", "tool": "create_legal_document", "result": res}
        
        else:
            return {"status": "error", "message": f"Unknown tool: '{tool_name}'"}
    except Exception as e:
        return {"status": "error", "message": f"Execution of tool '{tool_name}' failed: {e}"}
