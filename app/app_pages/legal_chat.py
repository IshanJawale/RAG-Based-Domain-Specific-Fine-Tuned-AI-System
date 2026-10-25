import os
import time

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import chat_store
from src.pipeline import MODE_LABELS, agent_answer, finalize, needs_agent, prepare_query, stream_answer
from src.document_generator import generate_legal_document
from src.legal_formats import BLANK, FORMATS, missing_fields, read_docx_preview
from src.llm import get_available_models, check_hpc_server, HPC_MODEL_NAME
from src.retrieval import HybridRetriever
from src.reranking import Reranker
from src.web_search import TavilySearcher

DOMAIN = "legal"
HISTORY_TURNS = 6  # earlier messages handed to the model for follow-up questions
SOURCE_LABELS = {"LOCAL": ":material/database: Local corpus", "WEB": ":material/language: Web", "BOTH": ":material/hub: Corpus + web"}
INTENT_LABELS = {
    "STATUTE": "Bare act, statute or section",
    "PRECEDENT": "Case law and precedents",
    "GENERAL_ADVICE": "Practical legal scenario",
    "CURRENT_INFORMATION": "Needs live web information",
    "GENERAL_KNOWLEDGE": "General, non-legal query",
    "UNKNOWN": "Unclear intent",
    "DRAFTING": "Document drafting",
}
KIND_ICONS = {"act": ":material/menu_book:", "court": ":material/account_balance:", "web": ":material/language:"}
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

SUGGESTIONS = {
    ":material/lock_open: Anticipatory bail": "What is the procedure for anticipatory bail under Section 438 CrPC and its BNSS equivalent?",
    ":material/account_balance: Article 21": "Explain Article 21 of the Indian Constitution and the landmark cases that expanded it.",
    ":material/receipt_long: Cheque bounce": "What are the remedies and limitation period for a cheque dishonour under Section 138 of the NI Act?",
    ":material/description: Draft a notice": "Draft a legal notice for breach of contract for an unpaid invoice of INR 15,00,000 under Section 73 of the Indian Contract Act.",
    ":material/gavel: Draft a bail application": "Draft a regular bail application for an accused arrested in an FIR for cheating under Section 318 BNS, in custody for 45 days, with the charge-sheet already filed.",
}


# ----------------------------------------------------------------------
# Resources and cached lookups
# ----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading LanceDB retriever and cross-encoder…")
def load_resources():
    hybrid = HybridRetriever()
    reranker = Reranker()
    try:
        web_search = TavilySearcher()
    except Exception:
        web_search = None
    return hybrid, reranker, web_search


@st.cache_data(ttl=20, show_spinner=False)
def backend_status():
    return check_hpc_server(), get_available_models()


# ----------------------------------------------------------------------
# State
# ----------------------------------------------------------------------
if "legal_chats" not in st.session_state:
    st.session_state.legal_chats = chat_store.load_chats(DOMAIN)
    st.session_state.legal_active = None  # None = unsaved blank chat
if "legal_pending" not in st.session_state:
    st.session_state.legal_pending = None


def persist():
    chat_store.save_chats(DOMAIN, st.session_state.legal_chats)


def get_active():
    return next((c for c in st.session_state.legal_chats if c["id"] == st.session_state.legal_active), None)


def start_new_chat():
    st.session_state.legal_active = None


def pick_from_history():
    st.session_state.legal_active = st.session_state.legal_history_pick


def delete_chat(chat_id):
    st.session_state.legal_chats = [c for c in st.session_state.legal_chats if c["id"] != chat_id]
    if st.session_state.legal_active == chat_id:
        st.session_state.legal_active = None
    persist()


def clear_all():
    st.session_state.legal_chats = []
    st.session_state.legal_active = None
    persist()


def pick_suggestion():
    choice = st.session_state.legal_suggest
    if choice:
        st.session_state.legal_pending = SUGGESTIONS[choice]
        st.session_state.legal_suggest = None


def rename_chat(chat_id):
    title = (st.session_state.get("rename_value") or "").strip()
    chat = next((c for c in st.session_state.legal_chats if c["id"] == chat_id), None)
    if chat and title:
        chat["title"] = title
        persist()


# The confirm buttons act through on_click callbacks, which run on the click itself,
# so the change is applied even if the dialog is not drawn again on that rerun.
@st.dialog("Rename chat")
def rename_dialog(chat):
    st.text_input("Title", value=chat["title"], max_chars=80, key="rename_value")
    if st.button("Save", type="primary", on_click=rename_chat, args=(chat["id"],)):
        st.rerun()


@st.dialog("Delete this chat?")
def delete_dialog(chat):
    st.write(f"**{chat['title']}** and its messages will be removed. Generated documents stay in `generated_documents/`.")
    if st.button("Delete chat", type="primary", icon=":material/delete:", on_click=delete_chat, args=(chat["id"],)):
        st.rerun()


try:
    hybrid, reranker, web_search = load_resources()
except Exception as e:
    st.error(f"Failed to load LanceDB or embedding models: {e}", icon=":material/error:")
    st.stop()

hpc_status, available_models = backend_status()
available_models = available_models or ["Qwen3-8B"]

# ----------------------------------------------------------------------
# Sidebar: navigation, history, then settings
# ----------------------------------------------------------------------
with st.sidebar:
    st.page_link("app_pages/home.py", label="All domains", icon=":material/arrow_back:")
    st.markdown("### :material/gavel: Legal assistant")
    st.button("New chat", icon=":material/edit_square:", width="stretch", on_click=start_new_chat)

    chats = sorted(st.session_state.legal_chats, key=lambda c: c["updated"], reverse=True)
    st.caption("Recent chats")
    if not chats:
        st.caption(":gray[No conversations yet. Ask a question to start one.]")
    else:
        titles = {c["id"]: c["title"] for c in chats}
        # Keep the list's selection in sync with the open chat (None while a new chat is blank)
        st.session_state.legal_history_pick = st.session_state.legal_active if st.session_state.legal_active in titles else None
        with st.container(height=300, border=False):
            st.radio(
                "Recent chats",
                options=[c["id"] for c in chats],
                format_func=lambda cid: titles[cid],
                captions=[chat_store.relative_label(c["updated"]) for c in chats],
                key="legal_history_pick",
                on_change=pick_from_history,
                label_visibility="collapsed",
            )
        selected = next((c for c in chats if c["id"] == st.session_state.legal_active), None)
        actions = st.container(horizontal=True, gap="small")
        if actions.button("Rename", icon=":material/edit:", type="tertiary", disabled=selected is None,
                          help="Rename the selected chat", key="side_rename"):
            rename_dialog(selected)
        if actions.button("Delete", icon=":material/delete:", type="tertiary", disabled=selected is None,
                          help="Delete the selected chat", key="side_delete"):
            delete_dialog(selected)
        with actions.popover("Clear all", icon=":material/delete_sweep:", type="tertiary"):
            st.write("Delete every saved legal chat?")
            st.button("Yes, delete all", on_click=clear_all, key="clear_all_confirm")

    with st.expander("Legal knowledge base", icon=":material/database:"):
        st.metric("Total indexed chunks", "5,676,435")
        a, b = st.columns(2)
        a.caption("High Court")
        a.write("5,002,082")
        b.caption("Supreme Court")
        b.write("546,547")
        a.caption("Central/State Acts")
        a.write("127,806")
        b.caption("Vector engine")
        b.write("LanceDB · BGE-small")

    with st.expander("LLM backend", icon=":material/memory:", expanded=True):
        if hpc_status:
            st.success(
                f"HPC online (`{hpc_status.get('host', 'cn02')}`)  \n"
                f"GPU `{hpc_status.get('device', 'RTX PRO 6000')}`  \n"
                f"Model `{hpc_status.get('active_model', 'Qwen3-VL-Legal')}`  \n"
                f"VRAM `{hpc_status.get('vram_allocated_gb', 0)} GB`",
                icon=":material/check_circle:",
            )
        selected_model = st.selectbox(
            "Model",
            options=available_models,
            help="Plaksha HPC cluster (RTX PRO 6000 48GB), local laptop models, or cloud Gemini API.",
        )
        if not hpc_status and selected_model == HPC_MODEL_NAME:
            st.warning("HPC tunnel not connected. Run on your laptop:", icon=":material/link_off:")
            st.code("ssh -N -L 8000:cn02:8000 ishan.jawale.pg26@hpc.plaksha.edu.in", language="powershell")
        st.caption(f"Web search: {'enabled (Tavily)' if web_search else 'disabled (no TAVILY_API_KEY)'}")

    with st.expander("Agent and search settings", icon=":material/tune:"):
        enable_agent = st.toggle(
            "Agent tool calling",
            value=True,
            help="Legal document generator (.docx), Section 34 CPC interest and damages calculator, bare acts search and precedent lookup.",
        )
        use_context = st.toggle(
            "Use chat context",
            value=True,
            help="Resolve follow-ups like 'and the punishment?' using earlier messages in this chat.",
        )
        top_k_retrieval = st.slider("Retrieval candidates", 10, 50, 20, 5)
        top_k_rerank = st.slider("Reranked legal passages", 3, 10, 5, 1)
        show_sources = st.toggle("Show retrieved citations", value=True)
        show_routing = st.toggle("Show routing reasoning", value=True)

    st.caption("Indian Legal RAG Assistant · LanceDB & Qwen3")


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------
def _md(text: str) -> str:
    """Make document text safe for st.markdown and highlight blanks."""
    text = text.replace("$", "\\$").replace("*", "\\*")
    return text.replace(BLANK, ":orange-badge[blank]")


def render_document(msg, key_prefix, chat):
    """Download, preview and a form to fill blanks for a generated document."""
    doc_path = msg.get("document_path")
    if not doc_path or not os.path.exists(doc_path):
        return
    args = msg.get("document_args") or {}
    key = args.get("document_type")
    blanks = missing_fields(key, args) if key in FORMATS else []

    row = st.container(horizontal=True, vertical_alignment="center")
    with open(doc_path, "rb") as f:
        row.download_button(
            f"Download {msg.get('document_type') or 'document'} (.docx)",
            data=f.read(),
            file_name=os.path.basename(doc_path),
            mime=DOCX_MIME,
            icon=":material/download:",
            key=f"dl_{key_prefix}",
        )
    if blanks:
        row.badge(f"{len(blanks)} details to fill", icon=":material/edit_note:", color="orange")
    elif key in FORMATS:
        row.badge("All details filled", icon=":material/check_circle:", color="green")

    with st.expander("Preview document", icon=":material/description:"):
        try:
            body, checklist = read_docx_preview(doc_path)
        except Exception as e:
            st.caption(f"Preview unavailable: {e}")
            body, checklist = [], []
        with st.container(border=True, height=420):
            for text, bold in body[:150]:
                st.markdown(f"**{_md(text)}**" if bold else _md(text))
        if checklist:
            st.caption("Drafter's checklist")
            for item in checklist:
                st.caption(f":material/checklist: {item}")

    if blanks and chat is not None:
        with st.expander(f"Fill in missing details ({len(blanks)})", icon=":material/edit_note:"):
            with st.form(f"fill_{key_prefix}", border=False):
                values = {}
                for fld in blanks:
                    label = fld["name"].replace("_", " ").capitalize()
                    if fld["is_list"]:
                        values[fld["name"]] = st.text_area(label, help=f"{fld['description']}. One item per line.", height=90)
                    else:
                        values[fld["name"]] = st.text_input(label, help=fld["description"])
                if st.form_submit_button("Update document", icon=":material/refresh:", type="primary"):
                    filled = {}
                    for fld in blanks:
                        v = values[fld["name"]].strip()
                        if v:
                            filled[fld["name"]] = [x.strip() for x in v.splitlines() if x.strip()] if fld["is_list"] else v
                    if filled:
                        new_args = {**args, **filled}
                        res = generate_legal_document(key, **{k: v for k, v in new_args.items() if k != "document_type"})
                        if res.get("status") == "success":
                            msg.update(document_path=res["file_path"], document_args=new_args)
                            chat["updated"] = time.time()
                            persist()
                            st.toast(f"Updated {len(filled)} field(s)", icon=":material/check_circle:")
                            st.rerun()
                        else:
                            st.error(res.get("message", "Could not regenerate the document."), icon=":material/error:")


def render_assistant_extras(msg, key_prefix, chat=None):
    """Routing, sources, notices and document for one stored assistant message."""
    routing = msg.get("routing")
    if routing and show_routing:
        with st.expander("Query routing", icon=":material/route:"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Route", SOURCE_LABELS.get(routing["source"], routing["source"]))
            c2.metric("Intent", INTENT_LABELS.get(routing["intent"], routing["intent"]))
            if routing.get("mode"):
                c3.metric("Evidence used", MODE_LABELS.get(routing["mode"], routing["mode"]))
            fmt = routing.get("document_format")
            if fmt:
                st.markdown(f":material/description: **Document format:** {fmt['label']}  \n:gray[{fmt['basis']}]")
            if routing.get("follow_up"):
                st.caption(":material/forum: Treated as a follow-up to the previous question.")
            st.caption(f"Router: {routing['reasoning']}")
            if routing.get("decision"):
                st.caption(f"Evidence: {routing['decision']}")

    sources = msg.get("sources") or []
    if sources and show_sources:
        with st.expander(f"{len(sources)} retrieved authorities", icon=":material/library_books:"):
            for i, s in enumerate(sources, 1):
                with st.container(border=True):
                    top = st.container(horizontal=True, vertical_alignment="center")
                    title = f"[{s['title']}]({s['url']})" if s.get("url") else s["title"]
                    top.markdown(f"**{KIND_ICONS.get(s['kind'], ':material/article:')} {i}.** {title}")
                    top.badge(s["type"], color="gray")
                    label = "rerank" if s.get("scored_by") == "rerank" else "distance"
                    st.caption(f"{s['section']}  ·  {label} score {s['score']:.3f}")
                    st.write(s["text"])

    if msg.get("tool_used"):
        st.caption(f":material/build: Tool executed: `{msg['tool_used']}`")
    mode = (routing or {}).get("mode")
    if mode == "hybrid":
        st.caption(":material/hub: Answer combines the legal corpus with live web sources.")
    elif mode == "web" or (mode is None and msg.get("used_web")):
        st.caption(":material/language: Answer is based on live web sources.")
    elif mode == "none":
        st.caption(":material/warning: No supporting evidence was found for this answer.")

    render_document(msg, key_prefix, chat)
    if msg.get("model"):
        st.caption(msg["model"])


def render_message(msg, key_prefix, chat=None):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_extras(msg, key_prefix, chat)


# ----------------------------------------------------------------------
# Main area
# ----------------------------------------------------------------------
active = get_active()

if active is None or not active["messages"]:
    st.space("medium")
    with st.container(horizontal_alignment="center"):
        st.title("Indian legal assistant", icon=":material/gavel:", text_alignment="center")
        st.markdown(
            "Ask about statutes, precedents or practical disputes. Answers are grounded in "
            "retrieved Acts and judgments, and the agent can draft notices, petitions and applications.",
            text_alignment="center",
        )
    st.space("small")
    st.pills(
        "Try asking",
        list(SUGGESTIONS),
        key="legal_suggest",
        on_change=pick_suggestion,
        label_visibility="collapsed",
    )
else:
    head = st.container(horizontal=True, vertical_alignment="center")
    head.subheader(active["title"])
    head.download_button(
        "",
        data=chat_store.to_markdown(active),
        file_name=f"{active['title'][:40]}.md",
        mime="text/markdown",
        icon=":material/download:",
        type="tertiary",
        help="Export chat as Markdown",
        key="export_chat",
    )
    action = head.menu_button(
        "Options",
        options=["Rename", "Delete chat"],
        icon=":material/more_horiz:",
        type="tertiary",
        help="Chat options",
        key="chat_menu",
    )
    if action == "Rename":
        rename_dialog(active)
    elif action == "Delete chat":
        delete_dialog(active)
    st.caption(f"Model: {selected_model}")
    for idx, msg in enumerate(active["messages"]):
        render_message(msg, f"{active['id']}_{idx}", active)

typed = st.chat_input("Ask a legal question, or ask me to draft a document…", submit_mode="disable")
question = typed or st.session_state.legal_pending
st.session_state.legal_pending = None

if question:
    if active is None:
        active = chat_store.new_chat()
        active["title"] = chat_store.make_title(question)
        st.session_state.legal_chats.append(active)
        st.session_state.legal_active = active["id"]

    history = active["messages"][-HISTORY_TURNS:] if use_context else []
    user_msg = {"role": "user", "content": question, "ts": time.time()}
    active["messages"].append(user_msg)
    active["updated"] = time.time()
    render_message(user_msg, "live_user")

    with st.chat_message("assistant"):
        started = time.time()
        result = None
        try:
            with st.status(":shimmer[Researching]", type="compact") as status:
                prep = prepare_query(
                    question,
                    hybrid=hybrid,
                    reranker=reranker,
                    web_search=web_search,
                    top_k_retrieval=top_k_retrieval,
                    top_k_rerank=top_k_rerank,
                    history=history,
                    step=lambda label: st.write(f":material/check: {label}"),
                )
                if needs_agent(prep, enable_agent):
                    st.write(":material/check: " + ("Drafting the document with the legal agent" if prep["drafting"] else "Running the agent's tools"))
                    agent_res = agent_answer(prep, selected_model)
                status.update(label=f"Researched in {time.time() - started:.1f}s", state="complete")

            if needs_agent(prep, enable_agent):
                st.markdown(agent_res["response"])
                result = finalize(prep, agent_res["response"], selected_model, agent_res)
            else:
                content = st.write_stream(stream_answer(prep, selected_model))
                result = finalize(prep, content if isinstance(content, str) else "".join(map(str, content)), selected_model)
        except Exception as e:
            st.error(f"Something went wrong while answering: {e}", icon=":material/error:")
            result = {"content": f"Sorry, something went wrong while answering:\n\n`{e}`"}

        result.update(role="assistant", ts=time.time())
        if result.get("routing"):
            render_assistant_extras(result, "live", None)

    active["messages"].append(result)
    active["updated"] = time.time()
    persist()
    st.rerun()
