"""
routing.py

Intelligent query router for the Indian Legal AI Assistant.

For each user question, the router determines:
  - intent   : STATUTE | PRECEDENT | GENERAL_ADVICE | CURRENT_INFORMATION | GENERAL_KNOWLEDGE | UNKNOWN
  - source   : LOCAL | WEB | BOTH

Features:
  - Fast-path deterministic classification (instant 0ms response)
  - Broad property, succession, family dispute, and civil litigation taxonomy
  - Scenario & exploratory query awareness (routing practical dispute overviews to BOTH for hybrid coverage)
  - Non-legal / mythology / science / coding separation from Indian legal corpus
  - Semantic LLM fallback for ambiguous queries
  - Dynamic cross-encoder confidence checking (needs_web_fallback)
"""

import json
import os
import re

from dotenv import load_dotenv

load_dotenv()


# ------------------------------------------------------------------
# Valid categories
# ------------------------------------------------------------------

VALID_INTENTS = {
    "STATUTE",
    "PRECEDENT",
    "GENERAL_ADVICE",
    "CURRENT_INFORMATION",
    "GENERAL_KNOWLEDGE",
    "UNKNOWN",
}

VALID_SOURCES = {"LOCAL", "WEB", "BOTH"}


# ------------------------------------------------------------------
# Domain Regex Markers
# ------------------------------------------------------------------

LEGAL_STATUTORY_MARKERS = re.compile(
    r"\b("
    r"(?:section|sec\.?|sections|article|art\.?|articles|order|orders|schedule|schedules|clause|clauses|rule|rules)\s+[a-z0-9\(\)\-]+|"
    r"ipc|crpc|cpc|bns|bnss|bsa|iea|nia|pocso|ndps|rera|gst|ibc|it\s*act|"
    r"bare\s*acts?|statutes?|statutory|legislative|provisions?|amendments?|ordinances?|"
    r"penal\s*code|contract\s*acts?|arbitration\s*acts?|evidence\s*acts?|negotiable\s*instruments|"
    r"specific\s*relief|limitation\s*acts?|motor\s*vehicles|hindu\s*marriage|special\s*marriage|"
    r"hindu\s*succession|indian\s*succession|partition\s*acts?|land\s*acquisition|registration\s*acts?|"
    r"stamp\s*acts?|easements?\s*acts?|transfer\s*of\s*property|tpa\b|"
    r"family\s*courts?\s*acts?|guardians?\s*and\s*wards?|domestic\s*violence\s*acts?|"
    r"sarfaesi|benami|court\s*fees?|prevention\s*of\s*corruption|pmla\b|"
    r"juvenile\s*justice|consumer\s*protection|right\s*to\s*information|rti\s*acts?|"
    r"companies\s*acts?|competition\s*acts?"
    r")\b",
    re.IGNORECASE,
)

LEGAL_PRECEDENT_MARKERS = re.compile(
    r"\b("
    r"case\s*laws?|precedents?|landmark\s*judgments?|landmark\s*rulings?|landmark\s*cases?|"
    r"judgments?|judgements?|rulings?|verdicts?|decrees?|orders?\s*of\s*court|"
    r"supreme\s*court|high\s*court|district\s*court|magistrate|sessions\s*court|"
    r"bench(?:es)?|division\s*bench|full\s*bench|constitution\s*bench|tribunals?|nclt|nclat|ngt|cat\b|drt\b|rera\s*authority|"
    r"curative\s*petitions?|review\s*petitions?|slp\b|special\s*leave\s*petitions?|writ\s*petitions?|"
    r"habeas\s*corpus|mandamus|certiorari|prohibition|quo\s*warranto|"
    r"quash(?:ing)?\s*(?:of\s*)?fir|bail\s*orders?|stay\s*orders?|interim\s*relief|"
    r"ratio\s*decidendi|obiter\s*dicta|locus\s*standi|res\s*judicata|lis\s*pendens|"
    r"kesavananda|maneka\s*gandhi|puttaswamy|navtej\s*singh|vishaka|shah\s*bano|minerva\s*mills|"
    r"vineeta\s*sharma|danamma|kale\s*(?:v\.|vs\.?|and)?\s*deputy|rangappa|basalingappa"
    r")\b|"
    r"\b[a-z0-9\.\s]+(?:\bvs\.?|\bv\.)\s+[a-z0-9\.\s]+",
    re.IGNORECASE,
)

LEGAL_PROPERTY_FAMILY_MARKERS = re.compile(
    r"\b("
    # Land & Property
    r"land\s*disputes?|property\s*disputes?|boundary\s*disputes?|title\s*disputes?|real\s*estate\s*disputes?|"
    r"land|lands|property|properties|plot|plots|flat|flats|agricultural\s*land|immovable\s*property|"
    r"ancestral\s*property|ancestral\s*land|ancestral\s*assets?|ancestral|coparcenary|coparceners?|"
    r"joint\s*family\s*property|joint\s*family|joint\s*property|huf\b|hindu\s*undivided\s*family|"
    r"partitions?|partition\s*suits?|partition\s*deeds?|suit\s*for\s*partition|"
    r"family\s*settlements?|family\s*arrangements?|family\s*partitions?|family\s*agreements?|"
    r"succession|inheritance|inherit|inherits|inherited|inheriting|heirs?|legal\s*heirs?|coparcenary\s*rights?|"
    r"khasra|khata|patta|khatoni|mutation|revenue\s*records?|jamabandi|7/12|satbara|"
    r"encroachments?|illegal\s*constructions?|demolition|ouster|"
    r"adverse\s*possession|illegal\s*possession|settled\s*possession|recovery\s*of\s*possession|possessory\s*title|"
    r"trespass(?:ing)?|easements?|easementary\s*rights?|right\s*of\s*way|"
    r"title\s*deeds?|sale\s*deeds?|gift\s*deeds?|relinquishment\s*deeds?|release\s*deeds?|settlement\s*deeds?|"
    r"wills?|probate|letters\s*of\s*administration|testamentary|codicils?|intestate|"
    r"power\s*of\s*attorney|poa\b|general\s*power\s*of\s*attorney|gpa\b|special\s*power\s*of\s*attorney|spa\b|"
    r"co-owners?|co-ownership|co-sharers?|undivided\s*shares?|share\s*in\s*property|mesne\s*profits|"
    r"suits?\s*for\s*declaration|declaration\s*of\s*title|"
    # Family & Matrimonial
    r"family\s*disputes?|marital\s*disputes?|matrimonial\s*disputes?|"
    r"families|sibling|siblings|brother|brothers|sister|sisters|father|mother|son|sons|daughter|daughters|"
    r"husband|wife|spouse|divorce|alimony|maintenance|child\s*custody|guardianship|"
    r"domestic\s*violence|dowry|streedhan|stridhan|daughter(?:'s)?\s*share|son(?:'s)?\s*share"
    r")\b",
    re.IGNORECASE,
)

LEGAL_GRIEVANCE_MARKERS = re.compile(
    r"\b("
    r"bail|anticipatory\s*bail|fir\b|police\s*complaints?|charge\s*sheets?|arrest|custody|remand|"
    r"evictions?|tenants?|tenancy|landlords?|rent\s*agreements?|rent\s*controls?|lease|leases|"
    r"cheque\s*bounce|cheque\s*dishonou?r|dishonou?r\s*of\s*cheque|breach\s*of\s*contracts?|legal\s*notices?|notice\s*of\s*demand|"
    r"defamation|cheating|fraud|forgery|impersonation|criminal\s*breach\s*of\s*trust|extortion|"
    r"consumer\s*complaints?|consumer\s*disputes?|cyber\s*crimes?|damages\s*claims?|recovery\s*of\s*money|"
    r"injunctions?|temporary\s*injunctions?|permanent\s*injunctions?|mandatory\s*injunctions?|status\s*quo|ad[- ]interim"
    r")\b",
    re.IGNORECASE,
)

LEGAL_CURRENT_MARKERS = re.compile(
    r"\b("
    r"2025|2026|latest\s*news|breaking|recent\s*judgment|recent\s*verdict|"
    r"recent\s*notification|recent\s*gazette|ongoing\s*trial|today|yesterday|this\s*week"
    r")\b",
    re.IGNORECASE,
)

LEGAL_GENERAL_MARKERS = re.compile(
    r"\b("
    r"laws?|legal(?:ity)?|illegality|illegal|courts?|judges?|judicial|judiciary|justices?|jurisprudence|"
    r"lawyers?|advocates?|attorneys?|bar\s*councils?|counsels?|"
    r"prosecutions?|accused|victims?|complainants?|petitioners?|respondents?|appellants?|"
    r"offences?|crimes?|criminals?|civil\s*suits?|plaints?|plaintiffs?|defendants?|"
    r"rights?|fundamental\s*rights?|liabilit(?:y|ies)|liable|penalt(?:y|ies)|fines?|imprisonments?|punishments?|jurisdictions?|"
    r"constitutions?|constitutional|statutory|litigations?|litigants?|"
    r"disputes?|disputed|disputing|conflicts?|controvers(?:y|ies)|claims?|claimants?|"
    r"tribunals?|settlements?|remed(?:y|ies)|reliefs?|compensations?|arbitrations?|mediations?|conciliations?"
    r")\b",
    re.IGNORECASE,
)

SCENARIO_OVERVIEW_MARKERS = re.compile(
    r"\b("
    r"scenarios?|examples?|types\s*of|kinds\s*of|categories\s*of|common\s*(?:causes?|disputes?|issues?|problems?)|"
    r"grounds?|overviews?|reasons?\s*for|how\s*to\s*resolve|how\s*do\s*courts\s*handle|"
    r"what\s*happens\s*when|what\s*are\s*the\s*options|how\s*to\s*prevent|practical\s*examples?"
    r")\b",
    re.IGNORECASE,
)

NON_LEGAL_MARKERS = re.compile(
    r"\b("
    r"mahabharat|mahabharata|ramayan|ramayana|bhagavad\s*gita|gita\b|vedas|upanishad|puran|purana|shastra|"
    r"krishna|arjuna|arjun\b|karna|pandavas|kauravas|bhishma|yudhishthir|yudhishthira|duryodhan|duryodhana|"
    r"draupadi|ravana|hanuman|rama|sita|lakshmana|valmiki|vyasa|greek\s*mythology|zeus|apollo|athena|thor|odin|mythology|"
    r"photosynthesis|gravity|quantum|relativity|einstein|newton|dna|rna|mitochondria|cell\b|molecule|atom|"
    r"physics|chemistry|biology|astronomy|planet|galaxy|solar\s*system|black\s*hole|calculus|algebra|geometry|"
    r"python|javascript|typescript|java\b|c\+\+|rust\b|html|css|sql\b|react|docker|kubernetes|git\b|linux|windows|"
    r"algorithm|data\s*structure|machine\s*learning|deep\s*learning|neural\s*network|llm\b|ai\b|api\b|programming|"
    r"write\s*code|write\s*a\s*script|software|"
    r"world\s*war|french\s*revolution|industrial\s*revolution|mughal|maurya|harappa|indus\s*valley|"
    r"capital\s*of|currency\s*of|population\s*of|president\s*of|prime\s*minister\s*of|king\s*of|queen\s*of|"
    r"continent|ocean|mount\s*everest|geography|history\s*of|"
    r"cricket|football|fifa|olympics|world\s*cup|virat\s*kohli|messi|ronaldo|ipl\b|movie|film|actor|actress|"
    r"bollywood|hollywood|song|music|guitar|piano|recipe|how\s*to\s*cook|how\s*to\s*bake|chocolate|cake|poem|joke|weather"
    r")\b",
    re.IGNORECASE,
)


# ------------------------------------------------------------------
# Router prompt for LLM fallback
# ------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """You are an intelligent query router for an Indian Legal AI Assistant.

The LOCAL repository contains:
1. Indian Central and State Bare Acts (Statutes, Sections, Definitions)
2. Supreme Court of India Judgments (1950-2025)
3. High Court Judgments (Bombay, Delhi, Calcutta, Allahabad, etc.)
It covers Indian civil law, property/land disputes, partition, family settlements, inheritance, tenancy, contracts, and criminal law.

It does NOT contain general knowledge, world history, ancient epics, science, coding, or non-legal trivia.

Classify the user's question into:

=== INTENT ===
STATUTE            - Indian Bare Acts, specific sections, or legislative definitions (e.g. "Section 138 NI Act").
PRECEDENT          - Indian court judgments, landmark rulings, case law (e.g. "Kesavananda Bharati case").
GENERAL_ADVICE     - Practical Indian legal scenarios, land/property disputes, partition in families, tenancy, or citizen grievances (e.g. "scenarios of land disputes in families", "landlord locked tenant out").
CURRENT_INFORMATION - High-profile ongoing legal trials, breaking 2025/2026 legal notifications.
GENERAL_KNOWLEDGE  - Strictly non-legal queries: ancient epics (Mahabharata, Ramayana), mythology, science, world history, coding, sports, or general trivia.
UNKNOWN            - Unclear or ambiguous queries.

=== SOURCE ===
LOCAL - For specific Indian statutory law, bare acts, sections, or court judgment precedents.
BOTH  - For practical legal scenarios, comparative legal overviews, land/family dispute scenarios, or queries benefiting from both local court jurisprudence and modern web context.
WEB   - Mandatory for General Knowledge, mythology, science, non-legal queries, or breaking non-legal news.

CRITICAL RULES:
1. Land disputes, family partition, property claims, inheritance, and citizen grievances ARE LEGAL QUERIES. Route them to LOCAL or BOTH.
2. Never route general knowledge, epics, mythology, science, or non-legal questions to LOCAL.

Return ONLY valid JSON (no markdown formatting, no code fences):
{
  "intent": "GENERAL_ADVICE",
  "source": "BOTH",
  "reasoning": "One short sentence explaining your decision."
}
"""


# ------------------------------------------------------------------
# LLM call
# ------------------------------------------------------------------

def _call_llm(prompt: str) -> str:
    """Send prompt to available LLM for fast classification (Cloud Gemini or local)."""
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if api_key:
        try:
            from src.llm import _call_gemini_api
            # thinking_budget=0: otherwise reasoning tokens eat the budget and truncate the JSON
            return _call_gemini_api(prompt, model_id="gemini-3.8-flash", max_tokens=256, temperature=0.0, thinking_budget=0)
        except Exception as e:
            print(f"[Router] Gemini API routing attempt failed: {e}")

    from src.llm import chat
    return chat(prompt, max_new_tokens=200, temperature=0.0)


# ------------------------------------------------------------------
# Main classification function
# ------------------------------------------------------------------

def classify_query(question: str) -> dict:
    """
    Classify a user question into routing decisions.
    Returns a dict with keys: intent, source, reasoning.
    """
    q_clean = question.strip()
    q_lower = q_clean.lower()

    # Fast heuristic checks
    has_non_legal = bool(NON_LEGAL_MARKERS.search(q_lower))
    has_statutory = bool(LEGAL_STATUTORY_MARKERS.search(q_lower))
    has_precedent = bool(LEGAL_PRECEDENT_MARKERS.search(q_lower))
    has_property_family = bool(LEGAL_PROPERTY_FAMILY_MARKERS.search(q_lower))
    has_grievance = bool(LEGAL_GRIEVANCE_MARKERS.search(q_lower))
    has_current = bool(LEGAL_CURRENT_MARKERS.search(q_lower))
    has_general_legal = bool(LEGAL_GENERAL_MARKERS.search(q_lower))
    is_broad_scenario = bool(SCENARIO_OVERVIEW_MARKERS.search(q_lower))

    has_any_legal = (
        has_statutory
        or has_precedent
        or has_property_family
        or has_grievance
        or has_general_legal
    )

    # 1. Non-Legal / Mythology / General Knowledge / Science query
    # Only route to WEB if there are NO legal terms at all
    if has_non_legal and not has_any_legal:
        return {
            "intent": "GENERAL_KNOWLEDGE",
            "source": "WEB",
            "reasoning": "Non-legal / general knowledge or mythological inquiry. Routing directly to live web search.",
        }

    # 2. Breaking news / Current legal developments
    if has_current and has_any_legal:
        return {
            "intent": "CURRENT_INFORMATION",
            "source": "BOTH",
            "reasoning": "Recent legal development or breaking notification. Routing to both local foundation and live web.",
        }

    # 3. Broad Legal Scenarios / Practical Dispute Overviews (e.g. "scenarios of land disputes in families")
    # Hybrid (BOTH) ensures local court jurisprudence (High Court / Supreme Court) AND practical web breakdowns
    if is_broad_scenario and has_any_legal:
        return {
            "intent": "GENERAL_ADVICE",
            "source": "BOTH",
            "reasoning": "Inquiry regarding practical dispute scenarios. Routing to both local court precedents and live web information for hybrid legal analysis.",
        }

    # 4. Explicit Statutory Provision
    if has_statutory and not has_precedent:
        return {
            "intent": "STATUTE",
            "source": "LOCAL",
            "reasoning": "Indian statutory bare act / section inquiry. Routing to local legislation repository.",
        }

    # 5. Explicit Court Judgment / Precedent
    if has_precedent:
        return {
            "intent": "PRECEDENT",
            "source": "LOCAL",
            "reasoning": "Court precedent / case law inquiry. Routing to Supreme Court and High Court judgments.",
        }

    # 6. Property / Family / Succession / Citizen Dispute Inquiries
    if has_property_family or has_grievance:
        return {
            "intent": "GENERAL_ADVICE",
            "source": "LOCAL",
            "reasoning": "Property, family, or citizen legal dispute inquiry. Routing to local legal database.",
        }

    # 7. General Legal Concept or Question
    if has_general_legal and not has_non_legal:
        return {
            "intent": "GENERAL_ADVICE",
            "source": "LOCAL",
            "reasoning": "General Indian legal inquiry. Routing to local legal repository.",
        }

    # 8. Pure Non-Legal Fallback (if query has zero legal markers and matches general non-legal patterns)
    if has_non_legal:
        return {
            "intent": "GENERAL_KNOWLEDGE",
            "source": "WEB",
            "reasoning": "Non-legal query. Routing to live web search.",
        }

    # 9. Semantic LLM Routing for Ambiguous Queries
    prompt = (
        f"{ROUTER_SYSTEM_PROMPT}\n\n"
        f"USER QUESTION:\n{q_clean}\n\n"
        f"Respond with JSON only."
    )

    try:
        raw = _call_llm(prompt).strip()

        # Pull the first {...} object out of the reply (handles code fences and chatter around it)
        m = re.search(r"\{[\s\S]*\}", raw) or re.search(r"\{[\s\S]*", raw)  # second form: truncated reply
        if not m:
            raise ValueError(f"no JSON object in router reply: {raw[:120]!r}")
        try:
            parsed = json.loads(m.group(0))
        except json.JSONDecodeError:
            # Malformed JSON (e.g. unescaped quotes in reasoning): read the fields we need directly
            parsed = dict(re.findall(r'"(intent|source|reasoning)"\s*:\s*"([^"]*)"', m.group(0)))
            if "intent" not in parsed:
                raise
        intent = str(parsed.get("intent", "GENERAL_ADVICE")).upper()
        source = str(parsed.get("source", "LOCAL")).upper()
        reasoning = str(parsed.get("reasoning", ""))

        if intent not in VALID_INTENTS:
            intent = "GENERAL_KNOWLEDGE" if has_non_legal or not has_any_legal else "GENERAL_ADVICE"
        if source not in VALID_SOURCES:
            source = "WEB" if intent == "GENERAL_KNOWLEDGE" else "LOCAL"
        # Only clearly non-legal questions skip the legal corpus; for anything else the
        # pipeline checks local evidence first and adds web results only if it is weak.
        if source == "WEB" and intent != "GENERAL_KNOWLEDGE":
            source = "BOTH" if intent == "CURRENT_INFORMATION" else "LOCAL"

        return {
            "intent": intent,
            "source": source,
            "reasoning": reasoning or "Classified by intelligent routing engine.",
        }

    except Exception as e:
        print(f"[Router] Classification exception ({e}), applying safe fallback.")
        if has_any_legal:
            return {
                "intent": "GENERAL_ADVICE",
                "source": "BOTH" if is_broad_scenario else "LOCAL",
                "reasoning": "Legal scenario inquiry. Routed to local legal repository.",
            }
        # No legal or non-legal markers matched: check the legal corpus first,
        # the pipeline escalates to the web if local evidence is weak.
        return {
            "intent": "UNKNOWN",
            "source": "LOCAL",
            "reasoning": "Ambiguous inquiry. Checking the legal corpus first, with web fallback.",
        }


# ------------------------------------------------------------------
# Confidence-based fallback helper
# ------------------------------------------------------------------

def needs_web_fallback(
    reranked_chunks: list[dict],
    threshold: float = 0.5,
) -> bool:
    """
    Return True if the top reranked chunk's score is below `threshold`.

    Cross-encoder scores are raw logits:
      > 2.0  : strong relevant match
      0.5-2.0: moderate match
      < 0.5  : weak or noisy match
      < 0.0  : irrelevant match

    Threshold 0.5 triggers seamless web fallback whenever local evidence
    does not contain a confident, high-relevance match.
    """
    if not reranked_chunks:
        return True

    top_score = reranked_chunks[0].get("rerank_score", -999.0)
    return top_score < threshold
