"""Doc Agent — Streamlit UI for M2 agentic workflows."""

import streamlit as st

from agentic_doc_agent import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    WorkflowId,
    get_agent_settings,
    run_workflow,
)
from agentic_doc_core.config import get_phoenix_settings
from agentic_doc_rag.config import RagSettings, get_rag_settings
from agentic_doc_rag.ingest import IngestEmptyCorpusError, IngestSourceNotFoundError
from agentic_doc_rag.observability import register_tracing
from agentic_doc_rag.retrieval import Retriever, create_retriever
from agentic_doc_studio.constants import (
    ARCHITECTURE_SUMMARY,
    EXAMPLE_GOALS,
    PHOENIX_UI_URL,
)
from agentic_doc_studio.startup_ingest import maybe_run_startup_ingest
from agentic_doc_studio.workspace import require_workspace_root

register_tracing(get_phoenix_settings())
require_workspace_root("studio")

st.set_page_config(
    page_title="Doc Agent",
    page_icon="🤖",
    layout="wide",
)

st.title("Doc Agent")
st.caption("Agentic documentation assistant — multi-step Answer over a grounded RAG index")


@st.cache_resource
def _bootstrap() -> tuple[RagSettings, Retriever, str | None]:
    settings = get_rag_settings()
    try:
        maybe_run_startup_ingest(settings)
    except (IngestSourceNotFoundError, IngestEmptyCorpusError, OSError) as exc:
        return settings, create_retriever(settings), str(exc)
    return settings, create_retriever(settings), None


def _llm_configured() -> bool:
    key = get_agent_settings().llm_api_key
    return bool(key and key.strip())


def _render_empty_corpus(settings: RagSettings, error: str | None) -> None:
    st.warning("Collection is empty — nothing is indexed yet.")
    if error:
        st.error(error)

    st.caption(
        f"Source: `{settings.ingest_source_dir}` · "
        f"Chroma: `{settings.chroma_persist_dir}` · "
        f"BM25: `{settings.bm25_persist_dir}`"
    )

    if settings.ingest_on_startup:
        st.markdown(
            "Startup ingest is **enabled** (`INGEST_ON_STARTUP`), but the collection is still empty. "
            "Confirm the source path above exists on this host, check the logs, then **reboot** the app. "
            "First build can take several minutes on free-tier hosts."
        )
        return

    st.markdown("#### Local development")
    st.code(
        "uv run explorer ingest\n# or:\nuv run explorer ingest --source corpora/rust-book/src",
        language="bash",
    )

    st.markdown("#### Streamlit Cloud / Docker")
    st.markdown(
        "Shell commands are not available on Streamlit Cloud. "
        "Set **secrets** (or env vars) and reboot so the app can index "
        "`corpora/rust-book` on first start:"
    )
    st.code(
        'INGEST_ON_STARTUP = "true"\nINGEST_SOURCE_DIR = "corpora/rust-book/src"',
        language="toml",
    )


def _render_llm_missing() -> None:
    st.warning("LLM credentials are not configured — Answer runs need an API key.")
    st.markdown(
        "Set **`LLM_API_KEY`** in `.env` (workspace root) or Streamlit secrets. "
        "Optional OpenAI-compatible proxy:"
    )
    st.code(
        "LLM_API_KEY=sk-...\n"
        "# LLM_BASE_URL=https://openrouter.ai/api/v1\n"
        "# LLM_MODEL=openai/gpt-4o-mini",
        language="bash",
    )


def _render_sidebar(
    settings: RagSettings,
    document_count: int,
    *,
    phoenix_enabled: bool,
    phoenix_project: str,
) -> None:
    st.header("Corpus")
    st.metric("Chunks indexed", document_count)
    st.text(f"Collection: {settings.chroma_collection_name}")
    st.caption(f"Store: {settings.chroma_persist_dir}")
    if settings.ingest_on_startup:
        st.caption("INGEST_ON_STARTUP is enabled")

    if phoenix_enabled:
        st.divider()
        st.header("Observability")
        st.link_button("Open Phoenix", PHOENIX_UI_URL, use_container_width=True)
        st.caption(f"Project: {phoenix_project}")

    st.divider()
    st.header("Try an example")
    for example in EXAMPLE_GOALS:
        if st.button(example, use_container_width=True):
            st.session_state["goal"] = example
            st.session_state["run_answer"] = True

    st.divider()
    with st.expander("Architecture", expanded=False):
        st.markdown(ARCHITECTURE_SUMMARY)


def _render_result_preview(result: AgentResult) -> None:
    """Minimal result display; metrics strip and tabs land in later steps."""
    st.subheader("Result")
    if result.status is AgentStatus.SUCCEEDED:
        st.success(f"Status: `{result.status.value}`")
    else:
        st.error(f"Status: `{result.status.value}`")
        if result.error:
            st.error(result.error)

    if result.answer:
        st.markdown(result.answer)
    elif result.status is AgentStatus.SUCCEEDED:
        st.info("Run succeeded but produced no answer text.")

    st.caption(
        f"Steps: {len(result.steps)} · Citations: {len(result.citations)} · "
        f"Retrieved: {len(result.retrieved)} · "
        f"Tool calls: {result.metrics.tool_calls}"
        + (
            f" · Duration: {result.metrics.duration_ms} ms"
            if result.metrics.duration_ms is not None
            else ""
        )
    )
    st.info("Metrics strip, Timeline, Citations, and Evidence tabs land in later steps.")


with st.spinner("Loading index (first start may build the demo corpus)..."):
    settings, retriever, startup_error = _bootstrap()

phoenix_settings = get_phoenix_settings()
document_count = retriever.count()
llm_ready = _llm_configured()

with st.sidebar:
    _render_sidebar(
        settings,
        document_count,
        phoenix_enabled=phoenix_settings.enabled,
        phoenix_project=phoenix_settings.project_name,
    )

if document_count == 0:
    _render_empty_corpus(settings, startup_error)
    st.stop()

if not llm_ready:
    _render_llm_missing()

# Form so Enter in the goal field submits (same as clicking Run answer).
with st.form("answer_form", clear_on_submit=False):
    st.text_input(
        "Goal",
        key="goal",
        placeholder="What is ownership in Rust?",
    )
    submitted = st.form_submit_button(
        "Run answer",
        type="primary",
        disabled=not llm_ready,
    )

if submitted:
    st.session_state["run_answer"] = True

goal = (st.session_state.get("goal") or "").strip()

if st.session_state.pop("run_answer", False):
    if not llm_ready:
        st.session_state.pop("last_result", None)
    elif not goal:
        st.info("Enter a goal to run the Answer workflow.")
    else:
        with st.spinner("Running Answer workflow (plan → retrieve → generate → evaluate)..."):
            result = run_workflow(
                AgentRequest(goal=goal, workflow=WorkflowId.ANSWER),
            )
        st.session_state["last_result"] = result

last_result = st.session_state.get("last_result")
if isinstance(last_result, AgentResult):
    _render_result_preview(last_result)
elif not llm_ready:
    pass
else:
    st.caption("Pick an example from the sidebar or type a goal, then **Run answer**.")
