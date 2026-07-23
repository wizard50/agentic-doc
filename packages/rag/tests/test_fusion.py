from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval.fusion import reciprocal_rank_fusion


def _result(chunk_id: str, score: float = 0.0) -> SearchResult:
    return SearchResult(chunk=DocumentChunk(id=chunk_id, text=chunk_id), score=score)


def test_reciprocal_rank_fusion_promotes_chunks_present_in_both_lists() -> None:
    dense = [_result("a"), _result("shared")]
    sparse = [_result("shared"), _result("c")]

    fused = reciprocal_rank_fusion([dense, sparse], top_k=3)

    assert [result.chunk.id for result in fused] == ["shared", "a", "c"]
    assert fused[0].score > fused[1].score > fused[2].score


def test_reciprocal_rank_fusion_respects_top_k() -> None:
    dense = [_result("a"), _result("b"), _result("c")]

    fused = reciprocal_rank_fusion([dense], top_k=2)

    assert len(fused) == 2
    assert [result.chunk.id for result in fused] == ["a", "b"]


def test_reciprocal_rank_fusion_returns_empty_for_no_results() -> None:
    assert reciprocal_rank_fusion([[], []], top_k=5) == []
    assert reciprocal_rank_fusion([[_result("a")]], top_k=0) == []
