"""Structured outputs for the answer workflow."""

from pydantic import BaseModel, Field


class PlanDraft(BaseModel):
    """LLM-structured plan for retrieval query rewrite (WorkflowId.ANSWER)."""

    search_query: str = Field(
        ...,
        min_length=1,
        description="Documentation search query derived from the user goal",
    )
    rationale: str | None = Field(
        default=None,
        description="Optional short reason for the chosen query",
    )


class AnswerDraft(BaseModel):
    """LLM-structured draft for WorkflowId.ANSWER.

    ``citation_chunk_ids`` must refer to chunk ids present in the provided context.
    Unknown ids are dropped when mapping to public ``Citation`` objects.

    When ``context_sufficient`` is false, the graph may re-retrieve using
    ``follow_up_query`` (if non-empty) until ``max_tool_rounds`` is hit.
    """

    answer: str = Field(
        ...,
        min_length=1,
        description="Grounded natural-language answer (markdown allowed)",
    )
    citation_chunk_ids: list[str] = Field(
        default_factory=list,
        description="Ids of context chunks that support the answer",
    )
    context_sufficient: bool = Field(
        default=True,
        description="False when retrieved context cannot support a solid answer",
    )
    follow_up_query: str | None = Field(
        default=None,
        description="Alternate search query when context is insufficient",
    )
