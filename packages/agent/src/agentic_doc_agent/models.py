"""Public data contracts for agent workflows."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from agentic_doc_rag.models import SearchResult


class WorkflowId(StrEnum):
    """Supported developer-focused workflows (M2)."""

    ANSWER = "answer"
    COMPARE = "compare"
    GAP_REPORT = "gap_report"


class AgentStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StepKind(StrEnum):
    PLAN = "plan"
    TOOL = "tool"
    GENERATE = "generate"
    EVALUATE = "evaluate"
    OTHER = "other"


class Citation(BaseModel):
    """Reference from a generated answer back to a retrieved chunk."""

    chunk_id: str
    source: str | None = None
    section_path: str | None = None
    score: float | None = None
    quote: str | None = Field(
        default=None,
        description="Optional short excerpt used to support a claim",
    )


class StepEvent(BaseModel):
    """One observable step in a workflow run (for UI timelines and traces)."""

    kind: StepKind
    name: str
    detail: str | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


class AgentMetrics(BaseModel):
    """Lightweight run metrics; expanded as evaluation lands."""

    faithfulness: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Groundedness of the answer vs retrieved context (0-1)",
    )
    tool_calls: int = Field(default=0, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)


class AgentRequest(BaseModel):
    """Input to run_workflow() — user/app intent for a single agent run.

    Retrieval knobs (top_k, mode, filters, rerank) are not part of this contract;
    the graph and RetrieveTool choose those per tool call (see AgentSettings defaults).
    """

    workflow: WorkflowId = WorkflowId.ANSWER
    goal: str = Field(..., min_length=1, description="User goal or question")
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional app-specific context (not used by core runtime yet)",
    )


class AgentResult(BaseModel):
    """Output from run_workflow()."""

    status: AgentStatus
    workflow: WorkflowId
    goal: str
    answer: str | None = Field(
        default=None,
        description="Primary natural-language result (markdown ok)",
    )
    structured: dict[str, Any] | None = Field(
        default=None,
        description="Workflow-specific structured payload (schema per workflow)",
    )
    citations: list[Citation] = Field(default_factory=list)
    steps: list[StepEvent] = Field(default_factory=list)
    retrieved: list[SearchResult] = Field(
        default_factory=list,
        description="Chunks used as evidence during the run",
    )
    metrics: AgentMetrics = Field(default_factory=AgentMetrics)
    error: str | None = None
