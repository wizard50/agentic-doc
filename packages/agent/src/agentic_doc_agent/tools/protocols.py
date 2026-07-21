"""Tool protocols for the agent runtime."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class AgentTool(Protocol):
    """Minimal tool interface used by graphs."""

    name: str
    description: str

    def invoke(self, **kwargs: Any) -> Any:
        """Run the tool and return a JSON-serializable or Pydantic result."""
        ...
