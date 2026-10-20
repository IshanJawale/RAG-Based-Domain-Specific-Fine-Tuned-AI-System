"""
document_generator.py - Professional Legal Document Generator for Indian Law.

Converts structured JSON from LLM tool-calls into clean, formal legal documents:
  - Legal Notice (Section 138 NI Act, Breach of Contract, Defamation, Property Dispute)
  - Legal Opinion / Memorandum
  - Case Brief / Judicial Summary
  - Written Submissions / Court Affidavits

Exports to formal .docx (and text fallback) with legal formatting, headers, citations, and signature blocks.
"""

import json
from datetime import datetime
from pathlib import Path
import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

DOCS_DIR = Path("generated_documents")
DOCS_DIR.mkdir(parents=True, exist_ok=True)


def create_legal_notice(
    title: str,
    sender_name: str,
    sender_address: str,
    recipient_name: str,
    recipient_address: str,
    subject: str,
    facts: list[str],
    legal_provisions: list[str],
    claims: list[str],
    demand_period_days: int = 15,
    advocate_name: str = "Advocate, High Court / Supreme Court of India",
) -> str:
    """Generate a formal Indian Legal Notice (.docx)."""
    doc = docx.Document()

    # Page Margins (Legal standard 1 inch)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Header / Notice Banner
    p_header = doc.add_paragraph()
    p_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_h = p_header.add_run("REGISTERED A.D. / SPEED POST\nLEGAL NOTICE")
    run_h.bold = True
    run_h.font.size = Pt(16)
    run_h.font.color.rgb = RGBColor(15, 23, 42)

    doc.add_paragraph(f"Date: {datetime.now().strftime('%d %B %Y')}")

    # To / Recipient
    doc.add_paragraph(f"To,\n{recipient_name}\n{recipient_address}\n")

    # From / Sender
    doc.add_paragraph(f"Under instructions from my client: {sender_name}, residing at {sender_address}.\n")

    # Subject
    p_sub = doc.add_paragraph()
    r_sub = p_sub.add_run(f"SUBJECT: {subject.upper()}")
    r_sub.bold = True
    r_sub.font.size = Pt(12)

    doc.add_paragraph("Sir/Madam,\nUnder instructions and authority from my said client, I do hereby serve upon you this Legal Notice:")

    # Facts
    doc.add_heading("I. FACTS OF THE CASE", level=2)
    for i, fact in enumerate(facts, start=1):
        doc.add_paragraph(f"{i}. {fact}")

    # Legal Grounds
    doc.add_heading("II. LEGAL PROVISIONS & VIOLATIONS", level=2)
    for prov in legal_provisions:
        doc.add_paragraph(f"• {prov}")

    # Demands / Requisition
    doc.add_heading("III. REQUISITION & DEMAND", level=2)
    for claim in claims:
        doc.add_paragraph(f"• {claim}")

    doc.add_paragraph(
        f"\nYou are hereby called upon to comply with the requisitions mentioned hereinabove within {demand_period_days} days "
        "from the receipt of this notice, failing which my client shall be constrained to initiate appropriate civil and/or "
        "criminal legal proceedings against you in a competent court of law, entirely at your risk, cost, and consequences."
    )

    # Sign-off
    p_sign = doc.add_paragraph("\nYours faithfully,\n\n\n___________________________\n")
    p_sign.add_run(f"{advocate_name}\nAdvocate for the Client")

    filename = f"Legal_Notice_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    filepath = DOCS_DIR / filename
    doc.save(str(filepath))
    return str(filepath)


def create_legal_opinion(
    title: str,
    client_name: str,
    query_issue: str,
    facts_summary: list[str],
    applicable_statutes: list[dict],
    judicial_precedents: list[dict],
    legal_analysis: str,
    conclusion_advice: str,
) -> str:
    """Generate a formal Legal Opinion / Memorandum (.docx)."""
    doc = docx.Document()

    # Header
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t = p_title.add_run(f"LEGAL OPINION / MEMORANDUM\n{title.upper()}")
    run_t.bold = True
    run_t.font.size = Pt(16)

    doc.add_paragraph(f"Client: {client_name}\nDate: {datetime.now().strftime('%d %B %Y')}\n")

    doc.add_heading("1. QUERY / QUESTION PRESENTED", level=2)
    doc.add_paragraph(query_issue)

    doc.add_heading("2. RELEVANT FACTUAL MATRIX", level=2)
    for f in facts_summary:
        doc.add_paragraph(f"• {f}")

    doc.add_heading("3. APPLICABLE STATUTES & SECTIONS", level=2)
    for stat in applicable_statutes:
        act = stat.get("act", "")
        sec = stat.get("section", "")
        desc = stat.get("relevance", "")
        doc.add_paragraph(f"• {act} (Section {sec}): {desc}")

    doc.add_heading("4. JUDICIAL PRECEDENTS & CASE LAW", level=2)
    for prec in judicial_precedents:
        case = prec.get("case", "")
        cit = prec.get("citation", "")
        hold = prec.get("holding", "")
        doc.add_paragraph(f"• {case} ({cit}): {hold}")

    doc.add_heading("5. LEGAL ANALYSIS & SYNTHESIS", level=2)
    doc.add_paragraph(legal_analysis)

    doc.add_heading("6. CONCLUSION & STRATEGIC RECOMMENDATION", level=2)
    doc.add_paragraph(conclusion_advice)

    doc.add_paragraph("\n\n___________________________\nSenior Legal Consultant / Advocate")

    filename = f"Legal_Opinion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    filepath = DOCS_DIR / filename
    doc.save(str(filepath))
    return str(filepath)


def create_case_brief(
    case_title: str,
    citation: str,
    court_bench: str,
    parties: str,
    facts: list[str],
    issues: list[str],
    appellant_arguments: list[str],
    respondent_arguments: list[str],
    ratio_decidendi: str,
    holding_decision: str,
) -> str:
    """Generate a formal Judicial Case Brief / Precedent Analysis (.docx)."""
    doc = docx.Document()

    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Title
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_t = p_title.add_run(f"JUDICIAL CASE BRIEF\n{case_title.upper()}")
    run_t.bold = True
    run_t.font.size = Pt(16)

    # Metadata table
    table = doc.add_table(rows=3, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    
    table.rows[0].cells[0].paragraphs[0].add_run("Citation:").bold = True
    table.rows[0].cells[1].paragraphs[0].text = citation
    table.rows[1].cells[0].paragraphs[0].add_run("Court & Bench:").bold = True
    table.rows[1].cells[1].paragraphs[0].text = court_bench
    table.rows[2].cells[0].paragraphs[0].add_run("Parties:").bold = True
    table.rows[2].cells[1].paragraphs[0].text = parties

    doc.add_paragraph()

    # Facts
    doc.add_heading("1. STATEMENT OF FACTS", level=2)
    for f in facts:
        doc.add_paragraph(f"• {f}")

    # Issues
    doc.add_heading("2. QUESTIONS OF LAW / ISSUES FRAMED", level=2)
    for issue in issues:
        doc.add_paragraph(f"• {issue}")

    # Submissions
    doc.add_heading("3. ARGUMENTS OF THE APPELLANT / PETITIONER", level=2)
    for arg in appellant_arguments:
        doc.add_paragraph(f"• {arg}")

    doc.add_heading("4. ARGUMENTS OF THE RESPONDENT", level=2)
    for arg in respondent_arguments:
        doc.add_paragraph(f"• {arg}")

    # Ratio
    doc.add_heading("5. RATIO DECIDENDI (LEGAL PRINCIPLE / REASONING)", level=2)
    doc.add_paragraph(ratio_decidendi)

    # Decision
    doc.add_heading("6. FINAL HOLDING & DISPOSITION", level=2)
    doc.add_paragraph(holding_decision)

    doc.add_paragraph(f"\nBrief Prepared: {datetime.now().strftime('%d %B %Y')}\nIndian Legal AI")

    filename = f"Case_Brief_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    filepath = DOCS_DIR / filename
    doc.save(str(filepath))
    return str(filepath)


def create_legal_report(
    title: str,
    subject: str = "Legal Assessment & Advisory Report",
    executive_summary: str = "",
    scenarios: list = None,
    statutory_provisions: list = None,
    judicial_precedents: list = None,
    recommendations: list = None,
    conclusion: str = "",
    author: str = "Indian Legal AI Intelligence",
) -> str:
    """Generate a formal comprehensive Legal Research & Advisory Report (.docx)."""
    doc = docx.Document()

    # Margins
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Document Header
    p_header = doc.add_paragraph()
    p_header.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run_h = p_header.add_run(f"LEGAL ASSESSMENT REPORT\n{title.upper()}")
    run_h.bold = True
    run_h.font.size = Pt(16)
    run_h.font.color.rgb = RGBColor(15, 23, 42)

    doc.add_paragraph(f"Date: {datetime.now().strftime('%d %B %Y')}\nSubject: {subject}\nPrepared By: {author}\n")

    # 1. Executive Summary
    if executive_summary:
        doc.add_heading("1. EXECUTIVE SUMMARY", level=2)
        doc.add_paragraph(executive_summary)

    # 2. Scenarios / Case Studies
    if scenarios:
        doc.add_heading("2. KEY SCENARIOS & FACTUAL PATTERNS", level=2)
        for i, sc in enumerate(scenarios, 1):
            if isinstance(sc, dict):
                s_name = sc.get("name") or sc.get("title") or f"Scenario {i}"
                s_desc = sc.get("description") or sc.get("facts") or str(sc)
                doc.add_heading(f"Scenario {i}: {s_name}", level=3)
                doc.add_paragraph(s_desc)
            else:
                doc.add_paragraph(f"{i}. {sc}")

    # 3. Applicable Statutory Framework
    if statutory_provisions:
        doc.add_heading("3. APPLICABLE STATUTORY & REGULATORY FRAMEWORK", level=2)
        for prov in statutory_provisions:
            doc.add_paragraph(f"• {prov}")

    # 4. Judicial Precedents
    if judicial_precedents:
        doc.add_heading("4. JUDICIAL PRECEDENTS & HIGH COURT / SUPREME COURT RATIO", level=2)
        for prec in judicial_precedents:
            doc.add_paragraph(f"• {prec}")

    # 5. Strategic Recommendations
    if recommendations:
        doc.add_heading("5. STRATEGIC LEGAL RECOMMENDATIONS & PREVENTIVE REMEDIES", level=2)
        for rec in recommendations:
            doc.add_paragraph(f"• {rec}")

    # 6. Conclusion
    if conclusion:
        doc.add_heading("6. CONCLUSION & NEXT STEPS", level=2)
        doc.add_paragraph(conclusion)

    doc.add_paragraph(f"\n\nReport Compiled on {datetime.now().strftime('%d %B %Y')}\n{author}")

    filename = f"Legal_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    filepath = DOCS_DIR / filename
    doc.save(str(filepath))
    return str(filepath)


def generate_legal_document(document_type: str, **kwargs) -> dict:
    """Unified dispatcher for document generation."""
    dtype = document_type.lower().strip().replace(" ", "_").replace("-", "_")
    try:
        if "notice" in dtype:
            path = create_legal_notice(
                title=kwargs.get("title", "Legal Notice"),
                sender_name=kwargs.get("sender_name", "Aggrieved Party"),
                sender_address=kwargs.get("sender_address", "New Delhi, India"),
                recipient_name=kwargs.get("recipient_name", "Opposite Party"),
                recipient_address=kwargs.get("recipient_address", "India"),
                subject=kwargs.get("subject", kwargs.get("title", "Breach of Obligations")),
                facts=kwargs.get("facts", []),
                legal_provisions=kwargs.get("legal_provisions", []),
                claims=kwargs.get("claims", kwargs.get("relief_sought", [])),
                demand_period_days=int(kwargs.get("demand_period_days", 15)),
                advocate_name=kwargs.get("advocate_name", "Advocate, High Court of Delhi"),
            )
            return {"status": "success", "file_path": path, "document_type": "Legal Notice"}

        elif "opinion" in dtype or "memo" in dtype:
            path = create_legal_opinion(
                title=kwargs.get("title", "Legal Opinion"),
                client_name=kwargs.get("client_name", "Client"),
                query_issue=kwargs.get("query_issue", kwargs.get("issue", "Query on Applicable Law")),
                facts_summary=kwargs.get("facts_summary", kwargs.get("facts", [])),
                applicable_statutes=kwargs.get("applicable_statutes", kwargs.get("applicable_law", [])),
                judicial_precedents=kwargs.get("judicial_precedents", kwargs.get("precedents", [])),
                legal_analysis=kwargs.get("legal_analysis", kwargs.get("analysis", "")),
                conclusion_advice=kwargs.get("conclusion_advice", kwargs.get("conclusion", "")),
            )
            return {"status": "success", "file_path": path, "document_type": "Legal Opinion"}

        elif "brief" in dtype or "case" in dtype:
            path = create_case_brief(
                case_title=kwargs.get("case_title", kwargs.get("title", "Case Brief")),
                citation=kwargs.get("citation", "Citation N/A"),
                court_bench=kwargs.get("court_bench", "Supreme Court / High Court of India"),
                parties=kwargs.get("parties", "Appellant vs Respondent"),
                facts=kwargs.get("facts", []),
                issues=kwargs.get("issues", []),
                appellant_arguments=kwargs.get("appellant_arguments", []),
                respondent_arguments=kwargs.get("respondent_arguments", []),
                ratio_decidendi=kwargs.get("ratio_decidendi", kwargs.get("ratio", "")),
                holding_decision=kwargs.get("holding_decision", kwargs.get("holding", "")),
            )
            return {"status": "success", "file_path": path, "document_type": "Case Brief"}

        elif "report" in dtype or "assessment" in dtype or "study" in dtype or "analysis" in dtype:
            path = create_legal_report(
                title=kwargs.get("title", "Legal Assessment Report"),
                subject=kwargs.get("subject", kwargs.get("title", "Legal Analysis Report")),
                executive_summary=kwargs.get("executive_summary", kwargs.get("summary", kwargs.get("introduction", ""))),
                scenarios=kwargs.get("scenarios", kwargs.get("scenarios_analyzed", kwargs.get("facts", []))),
                statutory_provisions=kwargs.get("statutory_provisions", kwargs.get("legal_provisions", kwargs.get("statutes", []))),
                judicial_precedents=kwargs.get("judicial_precedents", kwargs.get("precedents", [])),
                recommendations=kwargs.get("recommendations", kwargs.get("strategic_advice", kwargs.get("solutions", []))),
                conclusion=kwargs.get("conclusion", kwargs.get("conclusion_advice", "")),
            )
            return {"status": "success", "file_path": path, "document_type": "Legal Report"}

        else:
            # Fallback to report for any unspecified document request
            path = create_legal_report(
                title=kwargs.get("title", "Legal Document"),
                subject=kwargs.get("subject", document_type),
                executive_summary=kwargs.get("summary", ""),
                scenarios=kwargs.get("facts", []),
            )
            return {"status": "success", "file_path": path, "document_type": "Legal Document"}
    except Exception as e:
        return {"status": "error", "message": str(e)}
