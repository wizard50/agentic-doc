import json

from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

from agentic_doc_rag.models import SearchResult


def get_tracer(name: str) -> Tracer:
    return trace.get_tracer(name)


def mark_chain_span(span: Span) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value
    )


def mark_retriever_span(span: Span, *, query: str, top_k: int) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.RETRIEVER.value
    )
    span.set_attribute(SpanAttributes.INPUT_VALUE, query)
    span.set_attribute("top_k", top_k)


def record_search_results(span: Span, results: list[SearchResult]) -> None:
    span.set_attribute("result_count", len(results))
    if not results:
        return

    for index, result in enumerate(results):
        base = f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{index}"
        span.set_attribute(f"{base}.{DocumentAttributes.DOCUMENT_ID}", result.chunk.id)
        span.set_attribute(f"{base}.{DocumentAttributes.DOCUMENT_CONTENT}", result.chunk.text)
        span.set_attribute(f"{base}.{DocumentAttributes.DOCUMENT_SCORE}", result.score)
        if result.chunk.metadata:
            span.set_attribute(
                f"{base}.{DocumentAttributes.DOCUMENT_METADATA}",
                json.dumps(result.chunk.metadata),
            )

    span.set_attribute("top_scores", [result.score for result in results])
