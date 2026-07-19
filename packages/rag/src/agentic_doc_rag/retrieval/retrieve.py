from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.retrieval.fusion import reciprocal_rank_fusion
from agentic_doc_rag.retrieval.models import RetrievalRequest, SearchMode
from agentic_doc_rag.retrieval.utils import pool_k
from agentic_doc_rag.sparse.protocols import SparseIndex
from agentic_doc_rag.vectorstore.base import VectorStore


class RetrieveStage:
    """Dispatch retrieval to semantic or keyword backends based on request mode."""

    def __init__(self, vectorstore: VectorStore, sparse: SparseIndex) -> None:
        self._vectorstore = vectorstore
        self._sparse = sparse

    def run(
        self,
        request: RetrievalRequest,
        results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        del results
        prefetch_k = pool_k(request)
        match request.mode:
            case SearchMode.SEMANTIC:
                return self._vectorstore.search(request.query, k=prefetch_k)
            case SearchMode.KEYWORD:
                return self._sparse.search(request.query, k=prefetch_k)
            case SearchMode.HYBRID:
                dense = self._vectorstore.search(request.query, k=prefetch_k)
                sparse = self._sparse.search(request.query, k=prefetch_k)
                return reciprocal_rank_fusion([dense, sparse], top_k=prefetch_k)
