from typing import Protocol

from agentic_doc_rag.models import DocumentChunk, SearchResult


class SparseIndex(Protocol):
    """Keyword search index over document chunks."""

    def build(self, chunks: list[DocumentChunk]) -> None:
        """Build or rebuild the index from the given chunks."""
        ...

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return top-k chunks ranked by keyword relevance."""
        ...

    def count(self) -> int:
        """Return number of indexed chunks."""
        ...
