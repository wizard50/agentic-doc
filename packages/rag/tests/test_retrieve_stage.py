from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval import PipelineRetriever, RetrievalRequest, SearchMode
from agentic_doc_rag.retrieval.retrieve import RetrieveStage


class _StubVectorStore:
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


class _StubSparseIndex:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, int]] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        self.search_calls.append((query, k))
        return [SearchResult(chunk=DocumentChunk(id="sparse-1", text="sparse"), score=1.5)]

    def count(self) -> int:
        return 1


def _retriever(store: _StubVectorStore, sparse: _StubSparseIndex) -> PipelineRetriever:
    stages = [RetrieveStage(store, sparse)]
    return PipelineRetriever(stages=stages, vectorstore=store)


def test_retrieve_stage_uses_semantic_backend() -> None:
    store = _StubVectorStore()
    sparse = _StubSparseIndex()
    retriever = _retriever(store, sparse)

    results = retriever.retrieve(RetrievalRequest(query="ownership", mode=SearchMode.SEMANTIC, top_k=3))

    assert results[0].chunk.id == "dense-1"
    assert store.search_calls == [("ownership", 3)]
    assert sparse.search_calls == []


def test_retrieve_stage_uses_keyword_backend() -> None:
    store = _StubVectorStore()
    sparse = _StubSparseIndex()
    retriever = _retriever(store, sparse)

    results = retriever.retrieve(RetrievalRequest(query="borrowing", mode=SearchMode.KEYWORD, top_k=4))

    assert results[0].chunk.id == "sparse-1"
    assert sparse.search_calls == [("borrowing", 4)]
    assert store.search_calls == []


def test_retrieve_stage_hybrid_queries_both_backends_with_candidate_k() -> None:
    store = _StubVectorStore()
    sparse = _StubSparseIndex()
    retriever = _retriever(store, sparse)

    results = retriever.retrieve(
        RetrievalRequest(query="ownership", mode=SearchMode.HYBRID, top_k=2, candidate_k=10)
    )

    assert store.search_calls == [("ownership", 10)]
    assert sparse.search_calls == [("ownership", 10)]
    assert {result.chunk.id for result in results} == {"dense-1", "sparse-1"}
    assert all(result.score > 0 for result in results)


def test_retrieve_stage_hybrid_uses_top_k_when_larger_than_candidate_k() -> None:
    store = _StubVectorStore()
    sparse = _StubSparseIndex()
    retriever = _retriever(store, sparse)

    retriever.retrieve(
        RetrievalRequest(query="ownership", mode=SearchMode.HYBRID, top_k=8, candidate_k=3)
    )

    assert store.search_calls == [("ownership", 8)]
    assert sparse.search_calls == [("ownership", 8)]