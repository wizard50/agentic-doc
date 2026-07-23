"""Workflow runtime entry points."""

from __future__ import annotations

import time

from agentic_doc_agent.config import AgentSettings, get_agent_settings
from agentic_doc_agent.graphs.answer import build_answer_graph
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.llm.client import create_llm_client
from agentic_doc_agent.llm.models import LlmConfigError, LlmError
from agentic_doc_agent.llm.protocols import LlmClient
from agentic_doc_agent.models import (
    AgentMetrics,
    AgentRequest,
    AgentResult,
    AgentStatus,
    StepKind,
    WorkflowId,
)
from agentic_doc_agent.tools.retrieve import RetrieveTool
from agentic_doc_rag.config import get_rag_settings
from agentic_doc_rag.retrieval import Retriever, create_retriever


def list_workflows() -> list[WorkflowId]:
    """Return workflow ids the runtime intends to support."""
    return list(WorkflowId)


def run_workflow(
    request: AgentRequest,
    *,
    settings: AgentSettings | None = None,
    retriever: Retriever | None = None,
    llm: LlmClient | None = None,
    retrieve_tool: RetrieveTool | None = None,
) -> AgentResult:
    """Execute an agent workflow and return a structured result.

    Dependencies are injectable for tests. When omitted, production defaults
    build a retriever from RAG settings and an LLM client from agent settings.
    """
    resolved = settings or get_agent_settings()
    started = time.perf_counter()

    if request.workflow is not WorkflowId.ANSWER:
        return _failed_result(
            request,
            error=f"Workflow {request.workflow!r} is not implemented yet",
            duration_ms=_elapsed_ms(started),
        )

    try:
        tool = retrieve_tool or _default_retrieve_tool(resolved, retriever)
        client = llm if llm is not None else create_llm_client(resolved)
    except LlmConfigError as exc:
        return _failed_result(request, error=str(exc), duration_ms=_elapsed_ms(started))
    except Exception as exc:
        return _failed_result(
            request,
            error=f"Failed to initialize workflow dependencies: {exc}",
            duration_ms=_elapsed_ms(started),
        )

    try:
        graph = build_answer_graph(
            tool,
            client,
            faithfulness_enabled=resolved.faithfulness_enabled,
        )
        raw = graph.invoke(AgentGraphState(request=request))
        final_state = (
            raw if isinstance(raw, AgentGraphState) else AgentGraphState.model_validate(raw)
        )
    except LlmError as exc:
        return _failed_result(request, error=str(exc), duration_ms=_elapsed_ms(started))
    except Exception as exc:
        return _failed_result(
            request,
            error=f"Workflow execution failed: {exc}",
            duration_ms=_elapsed_ms(started),
        )

    return agent_result_from_state(final_state, duration_ms=_elapsed_ms(started))


def agent_result_from_state(
    state: AgentGraphState,
    *,
    duration_ms: int | None = None,
) -> AgentResult:
    """Map graph state to the public ``AgentResult`` contract."""
    tool_calls = sum(1 for step in state.steps if step.kind is StepKind.TOOL)
    metrics = AgentMetrics(
        faithfulness=state.faithfulness,
        tool_calls=tool_calls,
        duration_ms=duration_ms,
    )

    if state.error is not None:
        status = AgentStatus.FAILED
        error: str | None = state.error
        answer = state.draft_answer
    elif not state.draft_answer:
        status = AgentStatus.FAILED
        error = "generate produced no answer"
        answer = None
    else:
        status = AgentStatus.SUCCEEDED
        error = None
        answer = state.draft_answer

    return AgentResult(
        status=status,
        workflow=state.request.workflow,
        goal=state.request.goal,
        answer=answer,
        structured=state.structured,
        citations=list(state.citations),
        steps=list(state.steps),
        retrieved=list(state.retrieved),
        metrics=metrics,
        error=error,
    )


def _default_retrieve_tool(
    settings: AgentSettings,
    retriever: Retriever | None,
) -> RetrieveTool:
    resolved_retriever = (
        retriever if retriever is not None else create_retriever(get_rag_settings())
    )
    return RetrieveTool(resolved_retriever, default_top_k=settings.default_top_k)


def _failed_result(
    request: AgentRequest,
    *,
    error: str,
    duration_ms: int | None = None,
) -> AgentResult:
    return AgentResult(
        status=AgentStatus.FAILED,
        workflow=request.workflow,
        goal=request.goal,
        metrics=AgentMetrics(duration_ms=duration_ms),
        error=error,
    )


def _elapsed_ms(started: float) -> int:
    return max(0, int((time.perf_counter() - started) * 1000))
