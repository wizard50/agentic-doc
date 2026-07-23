from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span

_DEFAULT_MODEL_NAME = "all-MiniLM-L6-v2"
_DEFAULT_DIMENSIONS = 384


class ChromaDefaultEmbeddings:
    """Chroma's bundled ONNX embedding model (all-MiniLM-L6-v2)."""

    def __init__(self) -> None:
        self._model: DefaultEmbeddingFunction | None = None

    @property
    def model_name(self) -> str:
        return _DEFAULT_MODEL_NAME

    @property
    def dimensions(self) -> int:
        return _DEFAULT_DIMENSIONS

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        with get_tracer(__name__).start_as_current_span("embeddings.embed") as span:
            mark_chain_span(span)
            span.set_attribute("model", self.model_name)
            span.set_attribute("kind", "document")
            span.set_attribute("input_count", len(texts))
            span.set_attribute("dimensions", self.dimensions)
            return [vector.tolist() for vector in self._get_model()(texts)]

    def embed_query(self, query: str) -> list[float]:
        with get_tracer(__name__).start_as_current_span("embeddings.embed") as span:
            mark_chain_span(span)
            span.set_attribute("model", self.model_name)
            span.set_attribute("kind", "query")
            span.set_attribute("input_count", 1)
            span.set_attribute("dimensions", self.dimensions)
            return self._get_model()([query])[0].tolist()

    def _get_model(self) -> DefaultEmbeddingFunction:
        if self._model is None:
            self._model = DefaultEmbeddingFunction()
        return self._model
