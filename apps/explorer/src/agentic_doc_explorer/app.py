import streamlit as st

from agentic_doc_core.config import get_phoenix_settings
from agentic_doc_explorer.constants import (
    DEFAULT_TOP_K,
    EXAMPLE_QUERIES,
    PHOENIX_UI_URL,
    PREVIEW_LENGTH,
)
from agentic_doc_explorer.workspace import require_workspace_root
from agentic_doc_rag.config import get_rag_settings
from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.observability import register_tracing
from agentic_doc_rag.vectorstore.factory import create_vector_store

register_tracing(get_phoenix_settings())
require_workspace_root("explorer")

st.set_page_config(
    page_title="Doc Explorer",
    page_icon="📚",
    layout="wide",
)

st.title("Doc Explorer")
st.caption("Semantic search over technical documentation — Milestone 1 RAG core")


@st.cache_resource
def _vectorstore():
    return create_vector_store(get_rag_settings())


@st.cache_resource
def _settings():
    return get_rag_settings()


def _render_hit(hit: SearchResult, index: int) -> None:
    section = hit.chunk.metadata.get("section_path", "—")
    source = hit.chunk.metadata.get("source", "—")
    with st.expander(f"**{index}.** {section}  ·  score `{hit.score:.4f}`", expanded=index == 1):
        st.markdown(f"**Source:** `{source}`")
        text = hit.chunk.text
        if len(text) > PREVIEW_LENGTH:
            st.markdown(text[:PREVIEW_LENGTH] + "...")
            with st.popover("Show full chunk"):
                st.markdown(text)
        else:
            st.markdown(text)


settings = _settings()
phoenix_settings = get_phoenix_settings()
store = _vectorstore()
document_count = store.count()

with st.sidebar:
    st.header("Corpus")
    st.metric("Chunks indexed", document_count)
    st.text(f"Collection: {settings.chroma_collection_name}")
    st.text(f"Store: {settings.chroma_persist_dir}")
    top_k = st.slider("Results (top-k)", min_value=1, max_value=10, value=DEFAULT_TOP_K)

    if phoenix_settings.enabled:
        st.divider()
        st.header("Observability")
        st.link_button("Open Phoenix", PHOENIX_UI_URL, use_container_width=True)
        st.caption(f"Project: {phoenix_settings.project_name}")

    st.divider()
    st.header("Try an example")
    for example in EXAMPLE_QUERIES:
        if st.button(example, use_container_width=True):
            st.session_state["query"] = example
            st.session_state["run_search"] = True

if document_count == 0:
    st.warning("Collection is empty. Index the corpus first:")
    st.code("uv run explorer ingest", language="bash")
    st.stop()

query_input = st.text_input(
    "Ask a question",
    value=st.session_state.get("query", ""),
    placeholder="What is ownership in Rust?",
)
query = query_input or ""

if st.button("Search", type="primary"):
    st.session_state["run_search"] = True

if st.session_state.get("run_search"):
    if not query.strip():
        st.info("Enter a question to search.")
    else:
        with st.spinner("Searching..."):
            hits = store.search(query.strip(), k=top_k)

        st.subheader("Results")
        st.caption("Lower score = closer match (Chroma distance)")
        if not hits:
            st.info("No results found.")
        else:
            for index, hit in enumerate(hits, start=1):
                _render_hit(hit, index)
