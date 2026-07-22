"""LLM client protocol for graph nodes and tests."""

from __future__ import annotations

from typing import Protocol, TypeVar, runtime_checkable

from pydantic import BaseModel

from agentic_doc_agent.llm.models import ChatMessage, ChatResult

T = TypeVar("T", bound=BaseModel)


@runtime_checkable
class LlmClient(Protocol):
    """Minimal chat interface used by agent workflows."""

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        """Run a chat completion and return assistant text."""
        ...

    def complete_structured(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> T:
        """Run a chat completion and parse the result as a Pydantic model."""
        ...
