from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.retrieval.models import RetrievalRequest, SearchMode
from agentic_doc_rag.vectorstore.base import VectorStore


class SemanticStage:
    """Dense vector search via the configured vector store."""

    def __init__(self, vectorstore: VectorStore) -> None:
        self._vectorstore = vectorstore

    def run(
        self,
        request: RetrievalRequest,
        results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        del results
        if request.mode is not SearchMode.SEMANTIC:
            msg = f"Search mode {request.mode!r} is not implemented yet"
            raise NotImplementedError(msg)
        return self._vectorstore.search(request.query, k=request.top_k)