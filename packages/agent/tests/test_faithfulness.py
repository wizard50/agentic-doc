from typing import TypeVar

import pytest
from pydantic import BaseModel, ValidationError

from agentic_doc_agent.evaluation import (
    FaithfulnessVerdict,
    build_faithfulness_messages,
    score_faithfulness,
)
from agentic_doc_agent.llm import ChatMessage, ChatResult, ChatRole
from agentic_doc_rag.models import DocumentChunk, SearchResult

T = TypeVar("T", bound=BaseModel)


class FakeLlm:
    def __init__(
        self,
        verdict: FaithfulnessVerdict | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._verdict = verdict
        self._error = error
        self.structured_calls: list[tuple[list[ChatMessage], type[BaseModel]]] = []

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def complete_structured(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> T:
        self.structured_calls.append((messages, schema))
        if self._error is not None:
            raise self._error
        if self._verdict is None:
            raise RuntimeError("FakeLlm has no verdict configured")
        return schema.model_validate(self._verdict.model_dump())


def _hit(chunk_id: str, text: str = "Ownership rules") -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text=text,
            metadata={"source": f"{chunk_id}.md", "section_path": "Ownership"},
        ),
        score=0.9,
    )


def test_build_faithfulness_messages_includes_goal_answer_and_context() -> None:
    messages = build_faithfulness_messages(
        "What is ownership?",
        "Each value has one owner.",
        [_hit("c1", "A value can have only one owner.")],
    )

    assert len(messages) == 2
    assert messages[0].role is ChatRole.SYSTEM
    assert "faithfulness" in messages[0].content.lower()
    user = messages[1].content
    assert "What is ownership?" in user
    assert "Each value has one owner." in user
    assert "c1" in user
    assert "A value can have only one owner." in user


def test_build_faithfulness_messages_rejects_empty_goal_or_answer() -> None:
    with pytest.raises(ValueError, match="goal"):
        build_faithfulness_messages("  ", "answer", [])
    with pytest.raises(ValueError, match="answer"):
        build_faithfulness_messages("goal", "  ", [])


def test_score_faithfulness_happy_path() -> None:
    verdict = FaithfulnessVerdict(score=0.85, explanation="Most claims supported.")
    llm = FakeLlm(verdict)

    result = score_faithfulness(
        llm,
        goal="What is ownership?",
        answer="Each value has one owner.",
        retrieved=[_hit("c1")],
    )

    assert result.score == 0.85
    assert result.explanation == "Most claims supported."
    assert len(llm.structured_calls) == 1
    messages, schema = llm.structured_calls[0]
    assert schema is FaithfulnessVerdict
    assert messages[0].role is ChatRole.SYSTEM


def test_faithfulness_verdict_rejects_out_of_range_score() -> None:
    with pytest.raises(ValidationError):
        FaithfulnessVerdict(score=1.5, explanation="too high")
    with pytest.raises(ValidationError):
        FaithfulnessVerdict(score=-0.1, explanation="too low")
