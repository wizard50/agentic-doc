import pytest
from pydantic import ValidationError

from agentic_doc_agent.graphs import (
    ANSWER_SYSTEM_PROMPT,
    PLAN_SYSTEM_PROMPT,
    AnswerDraft,
    build_answer_messages,
    build_plan_messages,
    format_retrieved_context,
)
from agentic_doc_agent.llm import ChatRole
from agentic_doc_rag.models import DocumentChunk, SearchResult


def _hit(
    chunk_id: str,
    text: str,
    *,
    source: str = "ch04.md",
    section: str = "Ownership",
    score: float = 0.9,
) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text=text,
            metadata={"source": source, "section_path": section},
        ),
        score=score,
    )


def test_answer_draft_requires_answer() -> None:
    with pytest.raises(ValidationError):
        AnswerDraft(answer="")


def test_answer_draft_defaults_empty_citations() -> None:
    draft = AnswerDraft(answer="Ownership is …")
    assert draft.citation_chunk_ids == []
    assert draft.context_sufficient is True
    assert draft.follow_up_query is None


def test_build_plan_messages_structure() -> None:
    messages = build_plan_messages("What is ownership in Rust?")

    assert len(messages) == 2
    assert messages[0].role is ChatRole.SYSTEM
    assert messages[0].content == PLAN_SYSTEM_PROMPT
    assert "JSON Schema" in messages[0].content or "filled values" in messages[0].content
    assert messages[1].role is ChatRole.USER
    user = messages[1].content
    assert "## Goal" in user
    assert "What is ownership in Rust?" in user
    assert "search_query" in user


def test_build_plan_messages_rejects_blank_goal() -> None:
    with pytest.raises(ValueError, match="goal"):
        build_plan_messages("   ")


def test_build_answer_messages_structure() -> None:
    hits = [_hit("c1", "Ownership rules."), _hit("c2", "Borrowing rules.")]
    messages = build_answer_messages("What is ownership?", hits)

    assert len(messages) == 2
    assert messages[0].role is ChatRole.SYSTEM
    assert messages[0].content == ANSWER_SYSTEM_PROMPT
    assert messages[1].role is ChatRole.USER
    user = messages[1].content
    assert "## Goal" in user
    assert "What is ownership?" in user
    assert "`c1`" in user
    assert "`c2`" in user
    assert "Ownership rules." in user
    assert "Borrowing rules." in user
    assert "citation_chunk_ids" in user
    assert "context_sufficient" in user


def test_build_answer_messages_includes_rounds_note_when_re_retrieved() -> None:
    messages = build_answer_messages(
        "What is ownership?",
        [_hit("c1", "Ownership rules.")],
        retrieve_rounds=2,
    )
    user = messages[1].content
    assert "## Retrieval" in user
    assert "2 retrieve round" in user


def test_build_answer_messages_empty_retrieval() -> None:
    messages = build_answer_messages("Explain lifetimes", [])
    assert "(No passages retrieved.)" in messages[1].content


def test_build_answer_messages_rejects_blank_goal() -> None:
    with pytest.raises(ValueError, match="goal"):
        build_answer_messages("   ", [])


def test_format_retrieved_context_truncates_long_text() -> None:
    long_text = "x" * 100
    formatted = format_retrieved_context([_hit("long", long_text)], max_chunk_chars=20)
    assert "..." in formatted
    assert len(long_text) > 20
    # truncated body appears after metadata lines
    assert "xxx" in formatted


def test_format_retrieved_context_includes_metadata() -> None:
    formatted = format_retrieved_context(
        [_hit("id-1", "body", source="src.md", section="Sec A", score=0.42)]
    )
    assert "chunk_id: `id-1`" in formatted
    assert "source: `src.md`" in formatted
    assert "section_path: `Sec A`" in formatted
    assert "0.4200" in formatted
