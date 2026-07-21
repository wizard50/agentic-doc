"""Public API for the agentic-doc agent package.

Prefer importing from this module for stable entry points. Submodules remain
available for advanced use (custom tools, graphs, etc.).

Exports are resolved lazily so submodule imports do not pull the full
dependency graph or create import cycles.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "AgentMetrics",
    "AgentRequest",
    "AgentResult",
    "AgentSettings",
    "AgentStatus",
    "Citation",
    "StepEvent",
    "StepKind",
    "WorkflowId",
    "get_agent_settings",
    "list_workflows",
    "run_workflow",
]

# name -> (module, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    "AgentMetrics": ("agentic_doc_agent.models", "AgentMetrics"),
    "AgentRequest": ("agentic_doc_agent.models", "AgentRequest"),
    "AgentResult": ("agentic_doc_agent.models", "AgentResult"),
    "AgentSettings": ("agentic_doc_agent.config", "AgentSettings"),
    "AgentStatus": ("agentic_doc_agent.models", "AgentStatus"),
    "Citation": ("agentic_doc_agent.models", "Citation"),
    "StepEvent": ("agentic_doc_agent.models", "StepEvent"),
    "StepKind": ("agentic_doc_agent.models", "StepKind"),
    "WorkflowId": ("agentic_doc_agent.models", "WorkflowId"),
    "get_agent_settings": ("agentic_doc_agent.config", "get_agent_settings"),
    "list_workflows": ("agentic_doc_agent.runtime", "list_workflows"),
    "run_workflow": ("agentic_doc_agent.runtime", "run_workflow"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
