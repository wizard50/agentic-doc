from typing import Protocol

from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.retrieval.models import RetrievalRequest


class RetrievalStage(Protocol):
    """One step in a retrieval pipeline."""

    def run(
        self,
        request: RetrievalRequest,
        results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        """Apply this stage, optionally transforming prior stage output."""
        ...


class Retriever(Protocol):
    """Public retrieval API for search callers."""

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        """Run the configured retrieval pipeline for a query."""
        ...

    def count(self) -> int:
        """Return number of indexed chunks (health checks / eval guard)."""
        ...