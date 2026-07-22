"""Prompt builders for the answer workflow."""

from __future__ import annotations

from agentic_doc_agent.llm.models import ChatMessage, ChatRole
from agentic_doc_rag.models import SearchResult

DEFAULT_MAX_CHUNK_CHARS = 2000

ANSWER_SYSTEM_PROMPT = """\
You are a technical documentation assistant. Answer the user's goal using ONLY \
the provided context passages from the documentation corpus.

Rules:
- Prefer precise, developer-focused language.
- If the context is empty or insufficient, say what is missing; do not invent facts.
- When you use a passage, include its chunk id in citation_chunk_ids.
- Only cite chunk ids that appear in the context. Do not invent ids.
- citation_chunk_ids may be empty only when you truly used no passage \
(e.g. stating that evidence is missing).
- Return a response that matches the required structured schema.
"""


def build_answer_messages(
    goal: str,
    retrieved: list[SearchResult],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> list[ChatMessage]:
    """Build system + user messages for grounded answer generation."""
    goal_text = goal.strip()
    if not goal_text:
        raise ValueError("goal must be a non-empty string")
    if max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be >= 1")

    user_content = _build_user_content(goal_text, retrieved, max_chunk_chars=max_chunk_chars)
    return [
        ChatMessage(role=ChatRole.SYSTEM, content=ANSWER_SYSTEM_PROMPT),
        ChatMessage(role=ChatRole.USER, content=user_content),
    ]


def format_retrieved_context(
    retrieved: list[SearchResult],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
) -> str:
    """Format retrieved hits as a stable context block for the user message."""
    if max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be >= 1")
    if not retrieved:
        return "(No passages retrieved.)"

    blocks: list[str] = []
    for index, hit in enumerate(retrieved, start=1):
        meta = hit.chunk.metadata or {}
        source = meta.get("source", "—")
        section = meta.get("section_path", "—")
        text = _truncate(hit.chunk.text, max_chunk_chars)
        blocks.append(
            "\n".join(
                [
                    f"### Passage {index}",
                    f"- chunk_id: `{hit.chunk.id}`",
                    f"- source: `{source}`",
                    f"- section_path: `{section}`",
                    f"- score: {hit.score:.4f}",
                    "",
                    text,
                ]
            )
        )
    return "\n\n".join(blocks)


def _build_user_content(
    goal: str,
    retrieved: list[SearchResult],
    *,
    max_chunk_chars: int,
) -> str:
    context = format_retrieved_context(retrieved, max_chunk_chars=max_chunk_chars)
    return (
        f"## Goal\n\n{goal}\n\n"
        f"## Context passages\n\n{context}\n\n"
        "## Instructions\n\n"
        "Write the answer using only the context above. "
        "Populate citation_chunk_ids with supporting chunk_id values."
    )


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."
