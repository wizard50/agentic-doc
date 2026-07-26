"""Pure display helpers for Doc Agent (testable without Streamlit)."""

from __future__ import annotations

from agentic_doc_agent import StepEvent
from agentic_doc_rag.models import SearchResult


def format_faithfulness(score: float | None) -> str:
    """Display value for the faithfulness metric (em dash when unset)."""
    if score is None:
        return "—"
    return f"{score:.2f}"


def faithfulness_caption(score: float | None) -> str:
    if score is None:
        return "Disabled or judge skipped"
    if score >= 0.8:
        return "Strong groundedness"
    if score >= 0.5:
        return "Moderate groundedness"
    return "Weak groundedness — check citations"


def retrieved_by_chunk_id(retrieved: list[SearchResult]) -> dict[str, SearchResult]:
    """Map chunk id -> hit (last write wins if duplicates)."""
    return {hit.chunk.id: hit for hit in retrieved}


def step_title(index: int, step: StepEvent) -> str:
    kind = step.kind.value if hasattr(step.kind, "value") else str(step.kind)
    detail = f" — {step.detail}" if step.detail else ""
    return f"**{index}.** `{kind}` · {step.name}{detail}"
