from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.models import SearchResult

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span
from agentic_doc_rag.retrieval.models import RetrievalRequest


class Reranker(Protocol):
    """Rescore retrieved chunks for query-level relevance."""

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        ...


class CrossEncoderReranker:
    """Local cross-encoder reranker with lazy model loading."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: CrossEncoder | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    def rerank(self, query: str, results: list[SearchResult]) -> list[SearchResult]:
        if len(results) <= 1:
            return list(results)

        model = self._get_model()
        pairs = [(query, result.chunk.text) for result in results]
        raw_scores = model.predict(pairs)
        scored_results = [
            SearchResult(chunk=result.chunk, score=float(score))
            for result, score in zip(results, raw_scores, strict=True)
        ]
        return sorted(scored_results, key=lambda result: result.score, reverse=True)

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder as CrossEncoderModel

            self._model = CrossEncoderModel(self._model_name)
        return self._model


class RerankStage:
    """Optionally rerank filtered candidates before the final top-k trim."""

    def __init__(self, reranker: Reranker, *, default_enabled: bool) -> None:
        self._reranker = reranker
        self._default_enabled = default_enabled

    def run(
        self,
        request: RetrievalRequest,
        results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        if not results:
            return []

        enabled = request.rerank if request.rerank is not None else self._default_enabled
        if not enabled:
            return results

        with get_tracer(__name__).start_as_current_span("retriever.rerank") as span:
            mark_chain_span(span)
            span.set_attribute("input_count", len(results))
            if isinstance(self._reranker, CrossEncoderReranker):
                span.set_attribute("model", self._reranker.model_name)

            reranked = self._reranker.rerank(request.query, results)
            span.set_attribute("output_count", len(reranked))
            return reranked


def create_reranker(settings: RagSettings) -> CrossEncoderReranker:
    return CrossEncoderReranker(settings.rerank_model)
