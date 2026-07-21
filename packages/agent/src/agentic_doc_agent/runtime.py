"""Workflow runtime entry points.

Graphs and tools are wired here as M2 implementation progresses.
"""

from __future__ import annotations

from agentic_doc_agent.config import AgentSettings, get_agent_settings
from agentic_doc_agent.models import AgentRequest, AgentResult, WorkflowId


def list_workflows() -> list[WorkflowId]:
    """Return workflow ids the runtime intends to support."""
    return list(WorkflowId)


def run_workflow(
    request: AgentRequest,
    *,
    settings: AgentSettings | None = None,
) -> AgentResult:
    """Execute an agent workflow and return a structured result.

    Not implemented yet — scaffold only. Callers should depend on this
    signature so apps and the future API stay stable while graphs land.
    """
    _ = settings or get_agent_settings()
    raise NotImplementedError(
        f"Workflow {request.workflow!r} is not implemented yet "
        "(agentic-doc-agent M2 scaffold)."
    )
