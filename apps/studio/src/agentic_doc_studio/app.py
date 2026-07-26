"""Doc Agent — Streamlit UI shell (M2 agentic workflows)."""

import streamlit as st

from agentic_doc_studio.workspace import require_workspace_root

require_workspace_root("studio")

st.set_page_config(
    page_title="Doc Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("Doc Agent")
st.caption(
    "Agentic documentation assistant — multi-step Answer over a grounded RAG index"
)

st.info("UI shell only — Answer workflow form and result panels land in later steps.")
