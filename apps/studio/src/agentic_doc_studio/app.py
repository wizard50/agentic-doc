"""Doc Agent — Streamlit UI for M2 agentic workflows."""

import streamlit as st

from agentic_doc_agent import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    Citation,
    StepEvent,
    WorkflowId,
    get_agent_settings,
    run_workflow,
)
from agentic_doc_core.config import get_phoenix_settings
from agentic_doc_rag.config import RagSettings, get_rag_settings
from agentic_doc_rag.ingest import IngestEmptyCorpusError, IngestSourceNotFoundError
from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.observability import register_tracing
from agentic_doc_rag.retrieval import Retriever, create_retriever
from agentic_doc_studio.constants import (
    ARCHITECTURE_SUMMARY,
    EXAMPLE_GOALS,
    PHOENIX_UI_URL,
    PREVIEW_LENGTH,
    # WORKFLOW_CARDS,  # re-enable with workflow gallery when a second workflow ships
)
from agentic_doc_studio.display import (
    faithfulness_caption,
    format_faithfulness,
    retrieved_by_chunk_id,
    step_title,
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


# Re-enable when a second workflow is available (see WORKFLOW_CARDS in constants).
# def _render_workflow_gallery() -> None:
#     """Product map: Answer is live; Compare / Gap report are non-runnable placeholders."""
#     st.subheader("Workflows")
#     cols = st.columns(len(WORKFLOW_CARDS))
#     for col, card in zip(cols, WORKFLOW_CARDS, strict=True):
#         with col:
#             st.markdown(f"### {card['title']}")
#             if card["status"] == "live":
#                 st.success(card["badge"])
#             elif card["status"] == "coming_soon":
#                 st.warning(card["badge"])
#             else:
#                 st.info(card["badge"])
#             st.caption(card["blurb"])
#     st.caption(
#         "Only **Answer** is runnable in this demo. Compare and Gap report share the same "
#         "`AgentResult` contract when they land — no dead API buttons here."
#     )


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


def _render_chunk_text(text: str) -> None:
    if len(text) > PREVIEW_LENGTH:
        st.markdown(text[:PREVIEW_LENGTH] + "...")
        with st.popover("Show full chunk"):
            st.markdown(text)
    else:
        st.markdown(text)


def _render_timeline(steps: list[StepEvent]) -> None:
    if not steps:
        st.info("No steps recorded for this run.")
        return
    for index, step in enumerate(steps, start=1):
        with st.expander(step_title(index, step), expanded=index == 1):
            if step.payload:
                st.json(step.payload)
            else:
                st.caption("No payload for this step.")


def _render_citations(
    citations: list[Citation],
    retrieved: list[SearchResult],
) -> None:
    if not citations:
        st.info("No citations for this run.")
        return
    by_id = retrieved_by_chunk_id(retrieved)
    for index, citation in enumerate(citations, start=1):
        section = citation.section_path or "—"
        score = f"{citation.score:.4f}" if citation.score is not None else "—"
        with st.expander(
            f"**{index}.** {section}  ·  score `{score}`",
            expanded=index == 1,
        ):
            st.markdown(f"**Chunk id:** `{citation.chunk_id}`")
            if citation.source:
                st.markdown(f"**Source:** `{citation.source}`")
            if citation.quote:
                st.markdown(f"> {citation.quote}")
            hit = by_id.get(citation.chunk_id)
            if hit is None:
                st.caption("Full chunk not found in the retrieved set for this run.")
            else:
                _render_chunk_text(hit.chunk.text)


def _render_evidence(retrieved: list[SearchResult]) -> None:
    if not retrieved:
        st.info("No retrieved passages for this run.")
        return
    st.caption("Chunks available to the generator (may be larger than the citation set).")
    for index, hit in enumerate(retrieved, start=1):
        section = hit.chunk.metadata.get("section_path", "—")
        source = hit.chunk.metadata.get("source", "—")
        with st.expander(
            f"**{index}.** {section}  ·  score `{hit.score:.4f}`",
            expanded=index == 1,
        ):
            st.markdown(f"**Source:** `{source}`")
            st.markdown(f"**Chunk id:** `{hit.chunk.id}`")
            _render_chunk_text(hit.chunk.text)


def _render_result(result: AgentResult) -> None:
    """Status badge, metrics strip, answer, and detail tabs."""
    st.subheader("Result")

    if result.status is AgentStatus.SUCCEEDED:
        st.success(f"**Status:** `{result.status.value}`")
    else:
        st.error(f"**Status:** `{result.status.value}`")
        if result.error:
            st.error(result.error)

    faith = result.metrics.faithfulness
    duration = result.metrics.duration_ms
    cols = st.columns(4)
    cols[0].metric(
        "Faithfulness",
        format_faithfulness(faith),
        help="LLM-as-judge groundedness of the answer vs retrieved context (0-1)",
    )
    cols[0].caption(faithfulness_caption(faith))
    cols[1].metric("Tool calls", result.metrics.tool_calls)
    cols[2].metric(
        "Duration",
        f"{duration} ms" if duration is not None else "—",
    )
    cols[3].metric("Citations", len(result.citations))

    st.markdown("#### Answer")
    if result.answer:
        st.markdown(result.answer)
    elif result.status is AgentStatus.SUCCEEDED:
        st.info("Run succeeded but produced no answer text.")
    else:
        st.caption("No answer text for this run.")

    tab_timeline, tab_citations, tab_evidence = st.tabs(
        [
            f"Timeline ({len(result.steps)})",
            f"Citations ({len(result.citations)})",
            f"Evidence ({len(result.retrieved)})",
        ]
    )
    with tab_timeline:
        _render_timeline(result.steps)
    with tab_citations:
        _render_citations(result.citations, result.retrieved)
    with tab_evidence:
        _render_evidence(result.retrieved)


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

# _render_workflow_gallery()  # re-enable when a second workflow ships

if not llm_ready:
    _render_llm_missing()

st.markdown("#### Run Answer")
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
    _render_result(last_result)
elif not llm_ready:
    pass
else:
    st.caption("Pick an example from the sidebar or type a goal, then **Run answer**.")
