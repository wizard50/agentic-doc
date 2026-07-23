"""LLM-as-judge faithfulness scoring for grounded answers."""

from __future__ import annotations

from pydantic import BaseModel, Field

from agentic_doc_agent.llm.models import ChatMessage, ChatRole
from agentic_doc_agent.llm.protocols import LlmClient
from agentic_doc_rag.models import SearchResult

# Keep in sync with answer_prompts.DEFAULT_MAX_CHUNK_CHARS (avoid circular import
# via agentic_doc_agent.graphs package init).
DEFAULT_MAX_CHUNK_CHARS = 2000

FAITHFULNESS_SYSTEM_PROMPT = """\
You are a faithfulness judge for technical documentation answers.

Score how well the answer is grounded in the provided context passages ONLY.

Rules:
- score is a float from 0.0 to 1.0 (higher = more grounded).
- Prefer the fraction of answer claims that are supported by the context.
- Unsupported or invented technical claims pull the score down.
- An honest "context is insufficient / I don't know" answer that invents \
no facts should score high (near 1.0).
- Do not use outside knowledge to reward claims missing from the context.
- Return a response that matches the required structured schema.
"""


class FaithfulnessVerdict(BaseModel):
    """Structured output from the faithfulness judge."""

    score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Groundedness of the answer vs context (0-1)",
    )
    explanation: str = Field(
        ...,
        min_length=1,
        description="Brief rationale for the score",
    )


def build_faithfulness_messages(
    goal: str,
    answer: str,
    retrieved: list[SearchResult],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[ChatMessage]:
    """Build system + user messages for faithfulness scoring."""
    # Local import avoids circular import: graphs → answer_nodes → evaluation.
    from agentic_doc_agent.graphs.answer_prompts import format_retrieved_context

    goal_text = goal.strip()
    answer_text = answer.strip()
    if not goal_text:
        raise ValueError("goal must be a non-empty string")
    if not answer_text:
        raise ValueError("answer must be a non-empty string")
    if max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be >= 1")

    context = format_retrieved_context(retrieved, max_chunk_chars=max_chunk_chars)
    user_content = (
        f"## Goal\n\n{goal_text}\n\n"
        f"## Answer to evaluate\n\n{answer_text}\n\n"
        f"## Context passages\n\n{context}\n\n"
        "## Instructions\n\n"
        "Score how faithful the answer is to the context above. "
        "Populate score (0.0-1.0) and a short explanation."
    )
    return [
        ChatMessage(role=ChatRole.SYSTEM, content=FAITHFULNESS_SYSTEM_PROMPT),
        ChatMessage(role=ChatRole.USER, content=user_content),
    ]


def score_faithfulness(
    llm: LlmClient,
    *,
    goal: str,
    answer: str,
    retrieved: list[SearchResult],
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> FaithfulnessVerdict:
    """Score answer groundedness against retrieved context via structured LLM judge."""
    messages = build_faithfulness_messages(
        goal,
        answer,
        retrieved,
        max_chunk_chars=max_chunk_chars,
    )
    return llm.complete_structured(messages, FaithfulnessVerdict)
