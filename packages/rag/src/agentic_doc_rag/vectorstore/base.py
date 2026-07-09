from typing import Protocol

from agentic_doc_rag.models import DocumentChunk, SearchResult


class VectorStore(Protocol):
    """Storage and similarity search for document chunks."""

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        """Insert or update chunks (idempotent on chunk id)."""
        ...

    def search(self, query: str, k: int) -> list[SearchResult]:
        """Return top-k chunks ranked by relevance."""
        ...

    def delete(self, ids: list[str]) -> None:
        """Remove chunks by id."""
        ...

    def count(self) -> int:
        """Return number of stored chunks (useful for health checks / eval)."""
        ...
