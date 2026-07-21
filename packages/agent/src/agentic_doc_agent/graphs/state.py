"""Shared graph state models (filled in as workflows are implemented)."""

from typing import Any

from pydantic import BaseModel, Field

from agentic_doc_agent.models import AgentRequest, Citation, StepEvent
from agentic_doc_rag.models import SearchResult


class AgentGraphState(BaseModel):
    """Typed state carried through LangGraph nodes.

    Intentionally small for the scaffold; nodes will extend usage as workflows land.
    """

    request: AgentRequest
    retrieved: list[SearchResult] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    steps: list[StepEvent] = Field(default_factory=list)
    draft_answer: str | None = None
    structured: dict[str, Any] | None = None
    error: str | None = None
