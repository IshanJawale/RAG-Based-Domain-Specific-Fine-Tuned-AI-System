"""
app.py — entry point for the multi-domain AI assistant.

Run from project root:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st

# Project root on path so `src.*` imports work from every page
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

st.set_page_config(
    page_title="Sovereign AI · Domain assistants",
    page_icon=":material/balance:",
    layout="wide",
    initial_sidebar_state="expanded",
)

pages = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/legal_chat.py", title="Legal assistant", icon=":material/gavel:", url_path="legal"),
]

# Navigation is driven by the landing-page cards and the sidebar, not a menu
st.navigation(pages, position="hidden").run()
