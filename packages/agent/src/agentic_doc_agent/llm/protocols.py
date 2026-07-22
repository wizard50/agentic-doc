"""LLM client protocol for graph nodes and tests."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentic_doc_agent.llm.models import ChatMessage, ChatResult


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
