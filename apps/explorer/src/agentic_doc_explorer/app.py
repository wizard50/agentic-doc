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
from agentic_doc_rag.retrieval import MetadataFilter, RetrievalRequest, SearchMode, create_retriever

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
def _retriever():
    return create_retriever(get_rag_settings())


@st.cache_resource
def _settings():
    return get_rag_settings()


def _build_metadata_filter(
    source_contains: str,
    source_suffix: str,
    section_path_contains: str,
) -> MetadataFilter | None:
    values = {
        "source_contains": source_contains.strip() or None,
        "source_suffix": source_suffix.strip() or None,
        "section_path_contains": section_path_contains.strip() or None,
    }
    if not any(values.values()):
        return None
    return MetadataFilter(**values)


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
retriever = _retriever()
document_count = retriever.count()

with st.sidebar:
    st.header("Corpus")
    st.metric("Chunks indexed", document_count)
    st.text(f"Collection: {settings.chroma_collection_name}")
    st.text(f"Store: {settings.chroma_persist_dir}")
    top_k = st.slider("Results (top-k)", min_value=1, max_value=10, value=DEFAULT_TOP_K)
    search_mode = st.selectbox(
        "Search mode",
        options=list(SearchMode),
        format_func=lambda mode: mode.value,
        index=list(SearchMode).index(settings.search_mode),
    )

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

with st.expander("Metadata filters", expanded=False):
    filter_cols = st.columns(3)
    with filter_cols[0]:
        source_contains = st.text_input("Source contains")
    with filter_cols[1]:
        source_suffix = st.text_input("Source suffix")
    with filter_cols[2]:
        section_path_contains = st.text_input("Section contains")

if st.button("Search", type="primary"):
    st.session_state["run_search"] = True

if st.session_state.get("run_search"):
    if not query.strip():
        st.info("Enter a question to search.")
    else:
        with st.spinner("Searching..."):
            hits = retriever.retrieve(
                RetrievalRequest(
                    query=query.strip(),
                    top_k=top_k,
                    mode=search_mode,
                    candidate_k=settings.candidate_k,
                    filters=_build_metadata_filter(
                        source_contains,
                        source_suffix,
                        section_path_contains,
                    ),
                )
            )

        st.subheader("Results")
        score_caption = {
            SearchMode.SEMANTIC: "Lower score = closer match (Chroma distance)",
            SearchMode.KEYWORD: "Higher score = stronger BM25 match",
            SearchMode.HYBRID: "Higher score = stronger fused rank (RRF)",
        }[search_mode]
        st.caption(score_caption)
        if not hits:
            st.info("No results found.")
        else:
            for index, hit in enumerate(hits, start=1):
                _render_hit(hit, index)
