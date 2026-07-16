from typing import Literal

from agentic_doc_rag.models import DocumentChunk, SearchResult


class PassthroughReranker:
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        del query
        return list(results)


class UnusedSparseIndex:
    def build(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        raise AssertionError("sparse search should not be called")

    def count(self) -> int:
        return 0


class StubVectorStore:
    def __init__(
        self,
        responses: dict[str, list[SearchResult]] | None = None,
        *,
        count: int = 0,
        match: Literal["exact", "contains"] = "exact",
        fixed_results: list[SearchResult] | None = None,
    ) -> None:
        self._responses = responses or {}
        self._count = count
        self._match = match
        self._fixed_results = fixed_results
        self.search_calls: list[tuple[str, int]] = []

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.search_calls.append((query, k))
        if self._fixed_results is not None:
            return self._fixed_results[:k]
        if self._match == "exact":
            return self._responses.get(query, [])
        for key, results in self._responses.items():
            if key in query or query in key:
                return results
        return next(iter(self._responses.values()), [])

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        return self._count


class TrackingVectorStore:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.search_calls.append((query, k))
        return [SearchResult(chunk=DocumentChunk(id="dense-1", text="dense"), score=0.2)]

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        return 1


class TrackingSparseIndex:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.search_calls.append((query, k))
        return [SearchResult(chunk=DocumentChunk(id="sparse-1", text="sparse"), score=1.5)]

    def count(self) -> int:
        return 1