"""Shared graph state models (filled in as workflows are implemented)."""

from typing import Any

from pydantic import BaseModel, Field

from agentic_doc_agent.models import AgentRequest, Citation, StepEvent
from agentic_doc_rag.models import SearchResult


class AgentGraphState(BaseModel):
    """Typed state carried through LangGraph nodes.

    Multi-step Answer fields (``retrieve_query``, ``retrieve_rounds``,
    ``needs_more_context``) support plan + re-retrieve loops; unused until
    those nodes are wired.
    """

    request: AgentRequest
    retrieved: list[SearchResult] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    steps: list[StepEvent] = Field(default_factory=list)
    draft_answer: str | None = None
    structured: dict[str, Any] | None = None
    faithfulness: float | None = None
    error: str | None = None
    retrieve_query: str | None = Field(
        default=None,
        description="Query for the next retrieve (goal, plan rewrite, or follow-up)",
    )
    retrieve_rounds: int = Field(
        default=0,
        ge=0,
        description="Number of completed retrieve tool invocations",
    )
    needs_more_context: bool = Field(
        default=False,
        description="Set by generate when another retrieve round is requested",
    )
