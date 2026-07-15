from typing import Any

from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span
from agentic_doc_rag.retrieval.models import MetadataFilter, RetrievalRequest


def _contains(haystack: str, needle: str) -> bool:
    return needle.casefold() in haystack.casefold()


def _has_suffix(haystack: str, suffix: str) -> bool:
    return haystack.casefold().endswith(suffix.casefold())


def matches_filter(metadata: dict[str, Any], filters: MetadataFilter) -> bool:
    source = str(metadata.get("source", ""))
    section_path = str(metadata.get("section_path", ""))

    if filters.source_contains is not None and not _contains(source, filters.source_contains):
        return False
    if filters.source_suffix is not None and not _has_suffix(source, filters.source_suffix):
        return False
    if filters.section_path_contains is not None and not _contains(
        section_path, filters.section_path_contains
    ):
        return False
    return True


class MetadataFilterStage:
    """Drop retrieved chunks that do not match optional request metadata filters."""

    def run(
        self,
        request: RetrievalRequest,
        results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        if not results or request.filters is None:
            return results or []

        with get_tracer(__name__).start_as_current_span("retriever.filter") as span:
            mark_chain_span(span)
            span.set_attribute("input_count", len(results))
            filtered = [result for result in results if matches_filter(result.chunk.metadata, request.filters)]
            span.set_attribute("output_count", len(filtered))
            return filtered