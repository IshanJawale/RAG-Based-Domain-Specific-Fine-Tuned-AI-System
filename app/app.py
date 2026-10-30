"""
app.py — entry point for the multi-domain AI assistant.

Run from project root:
    streamlit run app/app.py
"""

import sys
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

import mobile_access

# Project root on path so `src.*` imports work from every page
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv(Path(__file__).resolve().parent.parent / ".env")

st.set_page_config(
    page_title="Sovereign AI · Domain assistants",
    page_icon=":material/balance:",
    layout="wide",
    initial_sidebar_state="auto",  # open on desktop, tucked away on phones
)

mobile_access.phone_url()  # first run only: ngrok tunnel + QR code in the terminal
mobile_access.require_password()

# Phones: less padding, smaller headings and metrics, long URLs and citations wrap instead of overflowing
st.html("""<style>
[data-testid="stMarkdownContainer"] { overflow-wrap: anywhere; }
@media (max-width: 640px) {
  [data-testid="stMainBlockContainer"] { padding: 3.5rem 0.9rem 5rem; }
  h1 { font-size: 1.75rem !important; }
  h2 { font-size: 1.4rem !important; }
  h3 { font-size: 1.15rem !important; }
  [data-testid="stMetricValue"] { font-size: 1.35rem; }
  [data-testid="stChatMessage"] { padding: 0.5rem 0.25rem; gap: 0.5rem; }
  [data-testid="stSidebar"][aria-expanded="true"] { width: 86vw !important; min-width: 86vw !important; max-width: 86vw !important; }
}
</style>""")

pages = [
    st.Page("app_pages/home.py", title="Home", icon=":material/home:", default=True),
    st.Page("app_pages/legal_chat.py", title="Legal assistant", icon=":material/gavel:", url_path="legal"),
]

# Navigation is driven by the landing-page cards and the sidebar, not a menu
st.navigation(pages, position="hidden").run()
