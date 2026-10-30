import streamlit as st

from mobile_access import render_phone_qr

DOMAINS = [
    {
        "key": "legal",
        "title": "Legal",
        "icon": ":material/gavel:",
        "color": "orange",
        "blurb": "Indian law research grounded in 128K statutory sections and 5.5M court judgment chunks, with a tool-using agent that drafts notices, opinions and reports as Word files.",
        "tags": ["Bare acts", "Case law", "Document drafting"],
        "page": "app_pages/legal_chat.py",
    },
    {
        "key": "medicine",
        "title": "Medicine",
        "icon": ":material/stethoscope:",
        "color": "red",
        "blurb": "Clinical guidelines, drug references and literature search with cited, privacy-preserving answers.",
        "tags": ["Guidelines", "Pharmacology"],
    },
    {
        "key": "engineering",
        "title": "Engineering",
        "icon": ":material/engineering:",
        "color": "blue",
        "blurb": "Standards, specifications and design-code lookup with deterministic calculators.",
        "tags": ["Standards", "Calculations"],
    },
    {
        "key": "finance",
        "title": "Finance",
        "icon": ":material/account_balance:",
        "color": "green",
        "blurb": "Regulatory filings, circulars and compliance research across financial regulations.",
        "tags": ["Regulation", "Compliance"],
    },
]

with st.container(horizontal_alignment="center"):
    st.space("large")
    st.badge("Runs locally · no data leaves your machine", icon=":material/shield:", color="orange")
    st.title("Domain-adapted AI, on your own hardware", text_alignment="center")
    st.markdown(
        "Fine-tuned language models, federated retrieval and tool-using agents, "
        "grounded in primary sources. **Choose a domain to begin.**",
        text_alignment="center",
    )

st.space("medium")

with st.container(horizontal=True):  # wraps to two-by-two or one column on narrow screens
    st.metric("Indexed passages", "5.68M", border=True)
    st.metric("Statutory sections", "128K", border=True)
    st.metric("Fine-tuned params", "0.7%", border=True, help="RS-LoRA trains ~0.7% of the base model weights")
    st.metric("Peak VRAM (8B, 4-bit)", "5.96 GB", border=True)

st.space("small")
render_phone_qr()

st.space("medium")
st.subheader("Domains")

cols = st.columns(2)
for i, d in enumerate(DOMAINS):
    live = "page" in d
    with cols[i % 2].container(border=True, height="stretch"):
        head = st.container(horizontal=True, vertical_alignment="center")
        head.markdown(f"### {d['icon']} {d['title']}")
        if live:
            head.badge("Available", icon=":material/check_circle:", color="green")
        else:
            head.badge("Coming soon", icon=":material/schedule:", color="gray")
        st.write(d["blurb"])
        st.caption("  ·  ".join(d["tags"]))
        if live:
            if st.button(f"Open {d['title'].lower()} assistant", key=f"open_{d['key']}", type="primary", icon=":material/arrow_forward:"):
                st.switch_page(d["page"])
        else:
            st.button("Not yet available", key=f"open_{d['key']}", disabled=True)

st.space("medium")
st.caption("Qwen3-VL · RS-LoRA fine-tuning · LanceDB · cross-encoder reranking · Streamlit")
