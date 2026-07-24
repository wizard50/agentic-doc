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
- Set context_sufficient=true when the context supports a solid answer.
- Set context_sufficient=false only when the context cannot support the goal; \
then provide follow_up_query as a different, keyword-oriented search query. \
If you set context_sufficient=false without a useful follow_up_query, the \
workflow will not re-retrieve.
- When context_sufficient=true, leave follow_up_query null or empty.
- Return a response that matches the required structured schema.
"""

PLAN_SYSTEM_PROMPT = """\
You rewrite a user goal into a focused search query for a technical documentation \
corpus (e.g. a programming language book or API docs).

Rules:
- Output a JSON object with keys search_query (string, required) and rationale \
(string or null, optional). Example: \
{"search_query": "Rust ownership rules", "rationale": "keyword focus"}.
- search_query should be concise and keyword-oriented (good for hybrid/semantic search).
- Prefer domain terms from the goal; do not invent product names not implied by the goal.
- Do not answer the goal; only produce a retrieval query (and optional short rationale).
- If the goal is already a good query, you may return it nearly unchanged.
- Do not return a JSON Schema definition; return the filled values only.
"""


def build_plan_messages(goal: str) -> list[ChatMessage]:
    """Build system + user messages for retrieval query planning."""
    goal_text = goal.strip()
    if not goal_text:
        raise ValueError("goal must be a non-empty string")

    user_content = (
        f"## Goal\n\n{goal_text}\n\n"
        "## Instructions\n\n"
        "Return JSON with search_query set to a documentation search string for "
        "this goal. Optionally set rationale. "
        'Example shape: {"search_query": "...", "rationale": null}.'
    )
    return [
        ChatMessage(role=ChatRole.SYSTEM, content=PLAN_SYSTEM_PROMPT),
        ChatMessage(role=ChatRole.USER, content=user_content),
    ]


def build_answer_messages(
    goal: str,
    retrieved: list[SearchResult],
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    retrieve_rounds: int = 0,
) -> list[ChatMessage]:
    """Build system + user messages for grounded answer generation."""
    goal_text = goal.strip()
    if not goal_text:
        raise ValueError("goal must be a non-empty string")
    if max_chunk_chars < 1:
        raise ValueError("max_chunk_chars must be >= 1")
    if retrieve_rounds < 0:
        raise ValueError("retrieve_rounds must be >= 0")

    user_content = _build_user_content(
        goal_text,
        retrieved,
        max_chunk_chars=max_chunk_chars,
        retrieve_rounds=retrieve_rounds,
    )
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
    retrieve_rounds: int = 0,
) -> str:
    context = format_retrieved_context(retrieved, max_chunk_chars=max_chunk_chars)
    rounds_note = ""
    if retrieve_rounds > 1:
        rounds_note = (
            f"## Retrieval\n\n"
            f"This is after {retrieve_rounds} retrieve round(s); "
            "context may include passages from prior searches.\n\n"
        )
    return (
        f"## Goal\n\n{goal}\n\n"
        f"{rounds_note}"
        f"## Context passages\n\n{context}\n\n"
        "## Instructions\n\n"
        "Write the answer using only the context above. "
        "Populate citation_chunk_ids with supporting chunk_id values. "
        "Set context_sufficient and follow_up_query per the system rules."
    )


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."
