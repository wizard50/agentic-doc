"""Pure retrieve/generate/evaluate node logic for the answer workflow.

No LangGraph compile here — nodes are plain functions for testability.
"""

from __future__ import annotations

from agentic_doc_agent.evaluation.faithfulness import score_faithfulness
from agentic_doc_agent.graphs.answer_models import AnswerDraft
from agentic_doc_agent.graphs.answer_prompts import (
    DEFAULT_MAX_CHUNK_CHARS,
    build_answer_messages,
)
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.llm.models import LlmError
from agentic_doc_agent.llm.protocols import LlmClient
from agentic_doc_agent.models import Citation, StepEvent, StepKind
from agentic_doc_agent.tools.retrieve import RetrieveTool
from agentic_doc_rag.models import SearchResult


def run_answer_retrieve(
    state: AgentGraphState,
    retrieve_tool: RetrieveTool,
) -> AgentGraphState:
    """Run retrieval for ``state.request.goal`` and record a tool step."""
    if state.error is not None:
        return state

    query = state.request.goal.strip()
    try:
        result = retrieve_tool.invoke(query=query)
    except Exception as exc:
        step = StepEvent(
            kind=StepKind.TOOL,
            name="retrieve",
            detail="Retrieval failed",
            payload={"query": query, "error": str(exc)},
        )
        return state.model_copy(
            update={
                "error": f"retrieve failed: {exc}",
                "steps": [*state.steps, step],
            }
        )

    step = StepEvent(
        kind=StepKind.TOOL,
        name="retrieve",
        detail=f"Retrieved {result.count} passage(s)",
        payload={"query": result.query, "count": result.count},
    )
    return state.model_copy(
        update={
            "retrieved": list(result.results),
            "steps": [*state.steps, step],
        }
    )


def run_answer_generate(
    state: AgentGraphState,
    llm: LlmClient,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> AgentGraphState:
    """Generate a grounded answer from retrieved context and record a step."""
    if state.error is not None:
        return state

    try:
        messages = build_answer_messages(
            state.request.goal,
            state.retrieved,
            max_chunk_chars=max_chunk_chars,
        )
        draft = llm.complete_structured(messages, AnswerDraft)
    except LlmError as exc:
        step = StepEvent(
            kind=StepKind.GENERATE,
            name="generate",
            detail="Generation failed",
            payload={"error": str(exc)},
        )
        return state.model_copy(
            update={
                "error": f"generate failed: {exc}",
                "steps": [*state.steps, step],
            }
        )
    except Exception as exc:
        step = StepEvent(
            kind=StepKind.GENERATE,
            name="generate",
            detail="Generation failed",
            payload={"error": str(exc)},
        )
        return state.model_copy(
            update={
                "error": f"generate failed: {exc}",
                "steps": [*state.steps, step],
            }
        )

    citations = citations_from_draft(draft, state.retrieved)
    step = StepEvent(
        kind=StepKind.GENERATE,
        name="generate",
        detail="Generated grounded answer",
        payload={
            "citation_count": len(citations),
            "cited_chunk_ids": [c.chunk_id for c in citations],
        },
    )
    return state.model_copy(
        update={
            "draft_answer": draft.answer,
            "citations": citations,
            "structured": draft.model_dump(),
            "steps": [*state.steps, step],
        }
    )


def run_answer_evaluate(
    state: AgentGraphState,
    llm: LlmClient,
    *,
    enabled: bool = True,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> AgentGraphState:
    """Score answer faithfulness against retrieved context (fail-soft).

    Skips when disabled, when the run already failed, or when there is no draft.
    Judge errors leave ``faithfulness`` unset and do not fail the workflow.
    """
    if not enabled or state.error is not None:
        return state

    answer = (state.draft_answer or "").strip()
    if not answer:
        return state

    try:
        verdict = score_faithfulness(
            llm,
            goal=state.request.goal,
            answer=answer,
            retrieved=state.retrieved,
            max_chunk_chars=max_chunk_chars,
        )
    except LlmError as exc:
        step = StepEvent(
            kind=StepKind.EVALUATE,
            name="evaluate",
            detail="Faithfulness scoring failed",
            payload={"error": str(exc)},
        )
        return state.model_copy(update={"steps": [*state.steps, step]})
    except Exception as exc:
        step = StepEvent(
            kind=StepKind.EVALUATE,
            name="evaluate",
            detail="Faithfulness scoring failed",
            payload={"error": str(exc)},
        )
        return state.model_copy(update={"steps": [*state.steps, step]})

    step = StepEvent(
        kind=StepKind.EVALUATE,
        name="evaluate",
        detail=f"Faithfulness score {verdict.score:.2f}",
        payload={
            "faithfulness": verdict.score,
            "explanation": verdict.explanation,
        },
    )
    return state.model_copy(
        update={
            "faithfulness": verdict.score,
            "steps": [*state.steps, step],
        }
    )


def citations_from_draft(
    draft: AnswerDraft,
    retrieved: list[SearchResult],
) -> list[Citation]:
    """Map draft citation ids to ``Citation`` rows; drop unknowns and duplicates."""
    by_id = {hit.chunk.id: hit for hit in retrieved}
    citations: list[Citation] = []
    seen: set[str] = set()
    for chunk_id in draft.citation_chunk_ids:
        if chunk_id in seen:
            continue
        hit = by_id.get(chunk_id)
        if hit is None:
            continue
        seen.add(chunk_id)
        meta = hit.chunk.metadata or {}
        citations.append(
            Citation(
                chunk_id=chunk_id,
                source=_meta_str(meta.get("source")),
                section_path=_meta_str(meta.get("section_path")),
                score=hit.score,
            )
        )
    return citations


def _meta_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
