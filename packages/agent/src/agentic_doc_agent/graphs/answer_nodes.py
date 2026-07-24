"""Pure plan/retrieve/generate/evaluate node logic for the answer workflow.

No LangGraph compile here — nodes are plain functions for testability.
"""

from __future__ import annotations

from opentelemetry.trace import Span

from agentic_doc_agent.evaluation.faithfulness import score_faithfulness
from agentic_doc_agent.graphs.answer_models import AnswerDraft, PlanDraft
from agentic_doc_agent.graphs.answer_prompts import (
    DEFAULT_MAX_CHUNK_CHARS,
    build_answer_messages,
    build_plan_messages,
)
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.llm.models import LlmError
from agentic_doc_agent.llm.protocols import LlmClient
from agentic_doc_agent.models import Citation, StepEvent, StepKind
from agentic_doc_agent.observability.tracing import (
    get_tracer,
    mark_evaluator_span,
    mark_llm_span,
    set_input_value,
    set_output_value,
    set_span_error,
)
from agentic_doc_agent.tools.retrieve import RetrieveTool
from agentic_doc_rag.models import SearchResult


def run_answer_plan(
    state: AgentGraphState,
    llm: LlmClient,
    *,
    enabled: bool = True,
) -> AgentGraphState:
    """Rewrite the user goal into a retrieval query (fail-soft).

    When disabled, already failed, or the judge errors, leaves ``retrieve_query``
    unset so retrieve can fall back to the goal. Never sets ``state.error``.
    """
    if not enabled or state.error is not None:
        return state

    with get_tracer(__name__).start_as_current_span("agent.plan") as span:
        mark_llm_span(span)
        set_input_value(span, state.request.goal)

        goal = state.request.goal.strip()
        try:
            messages = build_plan_messages(state.request.goal)
            plan = llm.complete_structured(messages, PlanDraft)
        except LlmError as exc:
            return _plan_fallback(state, span, goal=goal, error=str(exc))
        except Exception as exc:
            return _plan_fallback(state, span, goal=goal, error=str(exc))

        search_query = plan.search_query.strip()
        if not search_query:
            return _plan_fallback(state, span, goal=goal, error="empty search_query")

        set_output_value(span, search_query)
        step = StepEvent(
            kind=StepKind.PLAN,
            name="plan",
            detail=f"Planned search query: {search_query}",
            payload={
                "search_query": search_query,
                "rationale": plan.rationale,
            },
        )
        return state.model_copy(
            update={
                "retrieve_query": search_query,
                "steps": [*state.steps, step],
            }
        )


def run_answer_retrieve(
    state: AgentGraphState,
    retrieve_tool: RetrieveTool,
) -> AgentGraphState:
    """Run retrieval and merge hits into state.

    Uses ``state.retrieve_query`` when set, otherwise the user goal. Successful
    rounds increment ``retrieve_rounds`` and clear ``needs_more_context``.
    """
    if state.error is not None:
        return state

    planned = (state.retrieve_query or "").strip()
    query = planned if planned else state.request.goal.strip()
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

    merged = merge_retrieved(state.retrieved, list(result.results))
    next_round = state.retrieve_rounds + 1
    step = StepEvent(
        kind=StepKind.TOOL,
        name="retrieve",
        detail=f"Retrieved {result.count} passage(s) (round {next_round})",
        payload={
            "query": result.query,
            "count": result.count,
            "round": next_round,
            "merged_count": len(merged),
        },
    )
    return state.model_copy(
        update={
            "retrieved": merged,
            "retrieve_rounds": next_round,
            "needs_more_context": False,
            "steps": [*state.steps, step],
        }
    )


def run_answer_generate(
    state: AgentGraphState,
    llm: LlmClient,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> AgentGraphState:
    """Generate a grounded answer from retrieved context and record a step.

    When the draft marks context insufficient and provides a non-empty
    ``follow_up_query``, sets ``needs_more_context`` and ``retrieve_query`` for
    a possible re-retrieve. Missing follow-up stops the loop (fail-safe).
    """
    if state.error is not None:
        return state

    with get_tracer(__name__).start_as_current_span("agent.generate") as span:
        mark_llm_span(span)
        set_input_value(span, state.request.goal)
        span.set_attribute("retrieved_count", len(state.retrieved))

        try:
            messages = build_answer_messages(
                state.request.goal,
                state.retrieved,
                max_chunk_chars=max_chunk_chars,
                retrieve_rounds=state.retrieve_rounds,
            )
            draft = llm.complete_structured(messages, AnswerDraft)
        except LlmError as exc:
            set_span_error(span, str(exc))
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
            set_span_error(span, str(exc))
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
        follow_up = (draft.follow_up_query or "").strip() or None
        needs_more = (not draft.context_sufficient) and follow_up is not None
        next_query = follow_up if needs_more else state.retrieve_query

        set_output_value(span, draft.answer)
        span.set_attribute("citation_count", len(citations))
        span.set_attribute("context_sufficient", draft.context_sufficient)
        span.set_attribute("needs_more_context", needs_more)
        step = StepEvent(
            kind=StepKind.GENERATE,
            name="generate",
            detail="Generated grounded answer",
            payload={
                "citation_count": len(citations),
                "cited_chunk_ids": [c.chunk_id for c in citations],
                "context_sufficient": draft.context_sufficient,
                "needs_more_context": needs_more,
                "follow_up_query": follow_up,
            },
        )
        return state.model_copy(
            update={
                "draft_answer": draft.answer,
                "citations": citations,
                "structured": draft.model_dump(),
                "needs_more_context": needs_more,
                "retrieve_query": next_query,
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

    with get_tracer(__name__).start_as_current_span("agent.evaluate") as span:
        mark_evaluator_span(span)
        set_input_value(span, state.request.goal)

        try:
            verdict = score_faithfulness(
                llm,
                goal=state.request.goal,
                answer=answer,
                retrieved=state.retrieved,
                max_chunk_chars=max_chunk_chars,
            )
        except LlmError as exc:
            set_span_error(span, str(exc))
            step = StepEvent(
                kind=StepKind.EVALUATE,
                name="evaluate",
                detail="Faithfulness scoring failed",
                payload={"error": str(exc)},
            )
            return state.model_copy(update={"steps": [*state.steps, step]})
        except Exception as exc:
            set_span_error(span, str(exc))
            step = StepEvent(
                kind=StepKind.EVALUATE,
                name="evaluate",
                detail="Faithfulness scoring failed",
                payload={"error": str(exc)},
            )
            return state.model_copy(update={"steps": [*state.steps, step]})

        span.set_attribute("agent.faithfulness", verdict.score)
        set_output_value(span, f"score={verdict.score:.2f}; {verdict.explanation}")
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


def _plan_fallback(
    state: AgentGraphState,
    span: Span,
    *,
    goal: str,
    error: str,
) -> AgentGraphState:
    """Fail-soft plan: keep running with the original goal as retrieve_query."""
    set_span_error(span, error)
    # Still set retrieve_query so downstream uses a concrete string (the goal).
    fallback_query = goal if goal else None
    if fallback_query is not None:
        set_output_value(span, fallback_query)
    step = StepEvent(
        kind=StepKind.PLAN,
        name="plan",
        detail="Plan failed; retrieve will use the original goal",
        payload={"error": error, "search_query": fallback_query, "fallback": True},
    )
    return state.model_copy(
        update={
            "retrieve_query": fallback_query,
            "steps": [*state.steps, step],
        }
    )


def merge_retrieved(
    existing: list[SearchResult],
    new: list[SearchResult],
) -> list[SearchResult]:
    """Merge retrieval hits by chunk id; keep first-seen order, append new ids."""
    merged: list[SearchResult] = []
    seen: set[str] = set()
    for hit in [*existing, *new]:
        chunk_id = hit.chunk.id
        if chunk_id in seen:
            continue
        seen.add(chunk_id)
        merged.append(hit)
    return merged


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
