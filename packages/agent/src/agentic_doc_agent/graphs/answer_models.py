"""Structured outputs for the answer workflow."""

from pydantic import BaseModel, Field


class AnswerDraft(BaseModel):
    """LLM-structured draft for WorkflowId.ANSWER.

    ``citation_chunk_ids`` must refer to chunk ids present in the provided context.
    Unknown ids are dropped when mapping to public ``Citation`` objects.
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
