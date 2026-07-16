from __future__ import annotations

from typing import TYPE_CHECKING

from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer


class SentenceTransformerEmbeddings:
    """Local sentence-transformer embeddings with lazy model loading."""

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: SentenceTransformer | None = None
        self._dimensions: int | None = None

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            dims = self._get_model().get_embedding_dimension()
            if dims is None:
                msg = f"Could not determine embedding dimensions for {self._model_name}"
                raise RuntimeError(msg)
            self._dimensions = dims
        return self._dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with get_tracer(__name__).start_as_current_span("embeddings.embed") as span:
            mark_chain_span(span)
            span.set_attribute("model", self.model_name)
            span.set_attribute("kind", "document")
            span.set_attribute("input_count", len(texts))
            vectors = self._get_model().encode(texts, convert_to_numpy=True)
            span.set_attribute("dimensions", len(vectors[0]))
            return vectors.tolist()

    def embed_query(self, query: str) -> list[float]:
        with get_tracer(__name__).start_as_current_span("embeddings.embed") as span:
            mark_chain_span(span)
            span.set_attribute("model", self.model_name)
            span.set_attribute("kind", "query")
            span.set_attribute("input_count", 1)
            vector = self._get_model().encode(query, convert_to_numpy=True)
            span.set_attribute("dimensions", len(vector))
            return vector.tolist()

    def _get_model(self) -> SentenceTransformer:
        if self._model is None:
            from sentence_transformers import SentenceTransformer as SentenceTransformerModel

            self._model = SentenceTransformerModel(self._model_name)
        return self._model