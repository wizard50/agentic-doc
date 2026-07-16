from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval.models import RetrievalRequest
from agentic_doc_rag.retrieval.rerank import CrossEncoderReranker, RerankStage


def _result(chunk_id: str, score: float) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(id=chunk_id, text=f"text-{chunk_id}", metadata={}),
        score=score,
    )


class _ReverseReranker:
    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        del query
        return list(reversed(results))


def test_rerank_stage_passes_through_when_disabled() -> None:
    stage = RerankStage(_ReverseReranker(), default_enabled=False)
    results = [_result("1", 1.0), _result("2", 0.5)]

    output = stage.run(RetrievalRequest(query="ownership", top_k=2), results)

    assert output == results


def test_rerank_stage_uses_request_override_when_enabled() -> None:
    stage = RerankStage(_ReverseReranker(), default_enabled=False)
    results = [_result("1", 1.0), _result("2", 0.5)]
    request = RetrievalRequest(query="ownership", top_k=2, rerank=True)

    output = stage.run(request, results)

    assert [result.chunk.id for result in output] == ["2", "1"]


def test_rerank_stage_uses_default_enabled_when_request_is_none() -> None:
    stage = RerankStage(_ReverseReranker(), default_enabled=True)
    results = [_result("1", 1.0), _result("2", 0.5)]

    output = stage.run(RetrievalRequest(query="ownership", top_k=2), results)

    assert [result.chunk.id for result in output] == ["2", "1"]


def test_rerank_stage_returns_empty_for_empty_input() -> None:
    stage = RerankStage(_ReverseReranker(), default_enabled=True)

    assert stage.run(RetrievalRequest(query="ownership", top_k=2), []) == []


def test_cross_encoder_reranker_skips_model_for_single_result() -> None:
    reranker = CrossEncoderReranker("cross-encoder/ms-marco-MiniLM-L-6-v2")
    results = [_result("1", 1.0)]

    assert reranker.rerank("ownership", results) == results