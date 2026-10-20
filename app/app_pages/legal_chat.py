import os
import time
from collections import defaultdict

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

import chat_store
from src.pipeline import answer_query
from src.llm import get_available_models, check_hpc_server, HPC_MODEL_NAME
from src.pipeline import MODE_LABELS
from src.retrieval import HybridRetriever
from src.reranking import Reranker
from src.web_search import TavilySearcher

DOMAIN = "legal"
SOURCE_LABELS = {"LOCAL": ":material/database: Local corpus", "WEB": ":material/language: Web", "BOTH": ":material/hub: Corpus + web"}
INTENT_LABELS = {
    "STATUTE": "Bare act, statute or section",
    "PRECEDENT": "Case law and precedents",
    "GENERAL_ADVICE": "Practical legal scenario",
    "CURRENT_INFORMATION": "Needs live web information",
    "GENERAL_KNOWLEDGE": "General, non-legal query",
    "UNKNOWN": "Unclear intent",
}
KIND_ICONS = {"act": ":material/menu_book:", "court": ":material/account_balance:", "web": ":material/language:"}
DOCX_MIME = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

SUGGESTIONS = {
    ":material/lock_open: Anticipatory bail": "What is the procedure for anticipatory bail under Section 438 CrPC and its BNSS equivalent?",
    ":material/account_balance: Article 21": "Explain Article 21 of the Indian Constitution and the landmark cases that expanded it.",
    ":material/receipt_long: Cheque bounce": "What are the remedies and limitation period for a cheque dishonour under Section 138 of the NI Act?",
    ":material/description: Draft a notice": "Draft a legal notice for breach of contract for an unpaid invoice of INR 15,00,000 under Section 73 of the Indian Contract Act.",
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
    for c in st.session_state.legal_chats:
        if c["id"] == st.session_state.legal_active:
            return c
    return None


def start_new_chat():
    st.session_state.legal_active = None


def open_chat(chat_id):
    st.session_state.legal_active = chat_id


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
    st.button("New chat", icon=":material/edit_square:", type="secondary", width="stretch", on_click=start_new_chat)

    st.caption("Chat history")
    chats = sorted(st.session_state.legal_chats, key=lambda c: c["updated"], reverse=True)
    if not chats:
        st.caption("No conversations yet.")
    groups = defaultdict(list)
    for c in chats:
        groups[chat_store.group_label(c["updated"])].append(c)
    with st.container(height=320, border=False):
        for label in ("Today", "Yesterday", "Previous 7 days", "Older"):
            if label not in groups:
                continue
            st.caption(label)
            for c in groups[label]:
                st.button(
                    c["title"],
                    key=f"open_{c['id']}",
                    type="secondary" if c["id"] == st.session_state.legal_active else "tertiary",
                    width="stretch",
                    on_click=open_chat,
                    args=(c["id"],),
                )
    if chats:
        with st.popover("Clear all history", icon=":material/delete_sweep:", width="stretch"):
            st.write("Delete every saved legal chat?")
            st.button("Yes, delete all", type="secondary", on_click=clear_all, key="clear_all_confirm")

    st.space("small")
    with st.expander("Legal knowledge base", icon=":material/database:"):
        st.metric("Total indexed chunks", "5,647,872")
        a, b = st.columns(2)
        a.caption("High Court")
        a.write("5,002,082")
        b.caption("Supreme Court")
        b.write("546,547")
        a.caption("Central/State Acts")
        a.write("99,243")
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
        top_k_retrieval = st.slider("Retrieval candidates", 10, 50, 20, 5)
        top_k_rerank = st.slider("Reranked legal passages", 3, 10, 5, 1)
        show_sources = st.toggle("Show retrieved citations", value=True)
        show_routing = st.toggle("Show routing reasoning", value=True)

    st.caption("Indian Legal RAG Assistant · LanceDB & Qwen3")


# ----------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------
def render_assistant_extras(msg, key_prefix):
    """Routing, sources, notices and download for one stored assistant message."""
    routing = msg.get("routing")
    if routing and show_routing:
        with st.expander("Query routing", icon=":material/route:"):
            c1, c2, c3 = st.columns(3)
            c1.metric("Route", SOURCE_LABELS.get(routing["source"], routing["source"]))
            c2.metric("Intent", INTENT_LABELS.get(routing["intent"], routing["intent"]))
            if routing.get("mode"):
                c3.metric("Evidence used", MODE_LABELS.get(routing["mode"], routing["mode"]))
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

    doc_path = msg.get("document_path")
    if doc_path and os.path.exists(doc_path):
        with open(doc_path, "rb") as f:
            st.download_button(
                f"Download {msg.get('document_type') or 'document'} (.docx)",
                data=f.read(),
                file_name=os.path.basename(doc_path),
                mime=DOCX_MIME,
                icon=":material/download:",
                key=f"dl_{key_prefix}_{os.path.basename(doc_path)}",
            )


def render_message(msg, key_prefix):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_assistant_extras(msg, key_prefix)
            if msg.get("model"):
                st.caption(f"{msg['model']}")


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
            "retrieved Acts and judgments, and the agent can draft notices, opinions and reports.",
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
    head.button(
        "Delete chat",
        icon=":material/delete:",
        type="tertiary",
        width="content",
        on_click=delete_chat,
        args=(active["id"],),
        key="delete_active",
    )
    st.caption(f"Model: {selected_model}")
    for idx, msg in enumerate(active["messages"]):
        render_message(msg, f"{active['id']}_{idx}")

typed = st.chat_input("Ask a legal question…", submit_mode="disable")
question = typed or st.session_state.legal_pending
st.session_state.legal_pending = None

if question:
    if active is None:
        active = chat_store.new_chat()
        active["title"] = chat_store.make_title(question)
        st.session_state.legal_chats.append(active)
        st.session_state.legal_active = active["id"]

    user_msg = {"role": "user", "content": question, "ts": time.time()}
    active["messages"].append(user_msg)
    active["updated"] = time.time()
    render_message(user_msg, "live_user")

    with st.chat_message("assistant"):
        with st.status(":shimmer[Working]", type="compact") as status:
            steps = []

            def on_step(label):
                steps.append(label)
                st.write(f":material/check: {label}")

            try:
                started = time.time()
                result = answer_query(
                    question,
                    hybrid=hybrid,
                    reranker=reranker,
                    web_search=web_search,
                    model_name=selected_model,
                    enable_agent=enable_agent,
                    top_k_retrieval=top_k_retrieval,
                    top_k_rerank=top_k_rerank,
                    step=on_step,
                )
                status.update(label=f"Done in {time.time() - started:.1f}s", state="complete")
            except Exception as e:
                status.update(label="Failed", state="error")
                result = {"content": f"Sorry, something went wrong while answering:\n\n`{e}`"}

        result.update(role="assistant", ts=time.time())
        st.markdown(result["content"])
        render_assistant_extras(result, "live")
        if result.get("model"):
            st.caption(result["model"])

    active["messages"].append(result)
    active["updated"] = time.time()
    persist()
    st.rerun()
