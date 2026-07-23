from collections.abc import Sequence

from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span, record_search_results
from agentic_doc_rag.retrieval.models import RetrievalRequest
from agentic_doc_rag.retrieval.protocols import RetrievalStage
from agentic_doc_rag.vectorstore.base import VectorStore


class PipelineRetriever:
    """Run a fixed sequence of retrieval stages for each request."""

    def __init__(self, stages: Sequence[RetrievalStage], vectorstore: VectorStore) -> None:
        self._stages = stages
        self._vectorstore = vectorstore

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        with get_tracer(__name__).start_as_current_span("retriever.retrieve") as span:
            mark_chain_span(span)
            span.set_attribute("query", request.query)
            span.set_attribute("top_k", request.top_k)
            span.set_attribute("mode", request.mode.value)

            results: list[SearchResult] | None = None
            for stage in self._stages:
                results = stage.run(request, results)

            final_results = results or []
            record_search_results(span, final_results)
            return final_results

    def count(self) -> int:
        return self._vectorstore.count()
