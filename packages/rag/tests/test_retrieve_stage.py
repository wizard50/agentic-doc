from support.fakes import TrackingSparseIndex, TrackingVectorStore

from agentic_doc_rag.retrieval import RetrievalRequest, SearchMode
from agentic_doc_rag.retrieval.retrieve import RetrieveStage


def test_retrieve_stage_uses_semantic_backend_with_pool_k() -> None:
    store = TrackingVectorStore()
    sparse = TrackingSparseIndex()
    stage = RetrieveStage(store, sparse)

    results = stage.run(RetrievalRequest(query="ownership", mode=SearchMode.SEMANTIC, top_k=3))

    assert results[0].chunk.id == "dense-1"
    assert store.search_calls == [("ownership", 20)]
    assert sparse.search_calls == []


def test_retrieve_stage_uses_keyword_backend_with_pool_k() -> None:
    store = TrackingVectorStore()
    sparse = TrackingSparseIndex()
    stage = RetrieveStage(store, sparse)

    results = stage.run(RetrievalRequest(query="borrowing", mode=SearchMode.KEYWORD, top_k=4))

    assert results[0].chunk.id == "sparse-1"
    assert sparse.search_calls == [("borrowing", 20)]
    assert store.search_calls == []


def test_retrieve_stage_hybrid_queries_both_backends_with_candidate_k() -> None:
    store = TrackingVectorStore()
    sparse = TrackingSparseIndex()
    stage = RetrieveStage(store, sparse)

    results = stage.run(
        RetrievalRequest(query="ownership", mode=SearchMode.HYBRID, top_k=2, candidate_k=10)
    )

    assert store.search_calls == [("ownership", 10)]
    assert sparse.search_calls == [("ownership", 10)]
    assert {result.chunk.id for result in results} == {"dense-1", "sparse-1"}
    assert all(result.score > 0 for result in results)


def test_retrieve_stage_hybrid_uses_top_k_when_larger_than_candidate_k() -> None:
    store = TrackingVectorStore()
    sparse = TrackingSparseIndex()
    stage = RetrieveStage(store, sparse)

    stage.run(RetrievalRequest(query="ownership", mode=SearchMode.HYBRID, top_k=8, candidate_k=3))

    assert store.search_calls == [("ownership", 8)]
    assert sparse.search_calls == [("ownership", 8)]
