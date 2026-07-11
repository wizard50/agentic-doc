from collections import defaultdict
from collections.abc import Callable
from typing import Any

from agentic_doc_rag.evaluation.models import EvalQuery, EvalReport, QueryEvalResult, TagMetrics
from agentic_doc_rag.models import SearchResult

MetadataPredicate = Callable[[dict[str, Any]], bool]


def _source_matches(metadata: dict, expected_sources: list[str]) -> str | None:
    source = str(metadata.get("source", ""))
    for expected in expected_sources:
        if source.endswith(expected) or expected in source:
            return expected
    return None


def _section_matches(metadata: dict, expected_sections: list[str]) -> str | None:
    section_path = str(metadata.get("section_path", ""))
    for expected in expected_sections:
        if expected in section_path:
            return expected
    return None


def _first_match_rank(
    results: list[SearchResult],
    predicate: MetadataPredicate,
) -> int | None:
    for rank, result in enumerate(results, start=1):
        if predicate(result.chunk.metadata):
            return rank
    return None


def _recall_at_k(
    results: list[SearchResult],
    expected_sources: list[str],
    expected_sections: list[str],
) -> tuple[float, list[str], list[str]]:
    matched_sources = {
        matched
        for result in results
        if (matched := _source_matches(result.chunk.metadata, expected_sources)) is not None
    }
    matched_sections = {
        matched
        for result in results
        if (matched := _section_matches(result.chunk.metadata, expected_sections)) is not None
    }

    if expected_sources:
        return len(matched_sources) / len(expected_sources), sorted(matched_sources), []
    if expected_sections:
        return len(matched_sections) / len(expected_sections), [], sorted(matched_sections)
    return 0.0, [], []


def evaluate_query(
    query: EvalQuery,
    results: list[SearchResult],
    *,
    top_k: int,
) -> QueryEvalResult:
    """Evaluate retrieval results for a single golden query."""
    top_results = results[:top_k]

    first_match_rank = _first_match_rank(
        top_results,
        lambda metadata: (
            _source_matches(metadata, query.expected_sources) is not None
            or _section_matches(metadata, query.expected_section_paths) is not None
        ),
    )
    source_hit_at_k = (
        _first_match_rank(
            top_results,
            lambda metadata: _source_matches(metadata, query.expected_sources) is not None,
        )
        is not None
    )
    section_hit_at_k = (
        _first_match_rank(
            top_results,
            lambda metadata: _section_matches(metadata, query.expected_section_paths) is not None,
        )
        is not None
    )
    recall_at_k, matched_sources, matched_sections = _recall_at_k(
        top_results,
        query.expected_sources,
        query.expected_section_paths,
    )

    return QueryEvalResult(
        query_id=query.id,
        query=query.query,
        tags=query.tags,
        hit_at_k=first_match_rank is not None,
        source_hit_at_k=source_hit_at_k,
        section_hit_at_k=section_hit_at_k,
        first_match_rank=first_match_rank,
        recall_at_k=recall_at_k,
        matched_sources=matched_sources,
        matched_sections=matched_sections,
    )


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _mrr(results: list[QueryEvalResult]) -> float:
    if not results:
        return 0.0
    reciprocal_ranks = [
        1 / result.first_match_rank if result.first_match_rank is not None else 0.0
        for result in results
    ]
    return _mean(reciprocal_ranks)


def _tag_metrics(results: list[QueryEvalResult]) -> list[TagMetrics]:
    grouped: dict[str, list[QueryEvalResult]] = defaultdict(list)
    for result in results:
        for tag in result.tags:
            grouped[tag].append(result)

    metrics: list[TagMetrics] = []
    for tag in sorted(grouped):
        tag_results = grouped[tag]
        metrics.append(
            TagMetrics(
                tag=tag,
                query_count=len(tag_results),
                hit_at_k=_mean([float(result.hit_at_k) for result in tag_results]),
                mrr=_mrr(tag_results),
                recall_at_k=_mean([result.recall_at_k for result in tag_results]),
            )
        )
    return metrics


def compute_report(
    queries: list[EvalQuery],
    results_by_query_id: dict[str, list[SearchResult]],
    *,
    top_k: int,
    dataset_name: str,
) -> EvalReport:
    """Compute aggregate retrieval metrics for a batch of golden queries."""
    per_query_results = [
        evaluate_query(
            query,
            results_by_query_id.get(query.id, []),
            top_k=top_k,
        )
        for query in queries
    ]

    return EvalReport(
        dataset_name=dataset_name,
        query_count=len(queries),
        top_k=top_k,
        hit_at_k=_mean([float(result.hit_at_k) for result in per_query_results]),
        mrr=_mrr(per_query_results),
        recall_at_k=_mean([result.recall_at_k for result in per_query_results]),
        source_match_at_k=_mean([float(result.source_hit_at_k) for result in per_query_results]),
        section_match_at_k=_mean([float(result.section_hit_at_k) for result in per_query_results]),
        by_tag=_tag_metrics(per_query_results),
        results=per_query_results,
    )
