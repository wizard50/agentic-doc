from collections import defaultdict

from agentic_doc_rag.models import SearchResult

DEFAULT_RRF_K = 60


def reciprocal_rank_fusion(
    ranked_lists: list[list[SearchResult]],
    *,
    top_k: int,
    rrf_k: int = DEFAULT_RRF_K,
) -> list[SearchResult]:
    """Merge ranked result lists with reciprocal rank fusion (higher score is better)."""
    if top_k <= 0:
        return []

    fused_scores: dict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, SearchResult] = {}

    for results in ranked_lists:
        for rank, result in enumerate(results, start=1):
            chunk_id = result.chunk.id
            fused_scores[chunk_id] += 1.0 / (rrf_k + rank)
            if chunk_id not in chunks_by_id:
                chunks_by_id[chunk_id] = result

    if not fused_scores:
        return []

    sorted_ids = sorted(fused_scores, key=lambda chunk_id: fused_scores[chunk_id], reverse=True)[
        :top_k
    ]
    return [
        SearchResult(chunk=chunks_by_id[chunk_id].chunk, score=fused_scores[chunk_id])
        for chunk_id in sorted_ids
    ]