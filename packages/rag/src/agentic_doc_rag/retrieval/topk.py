from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.retrieval.models import RetrievalRequest


class TopKStage:
    """Trim the candidate list to the requested top-k after filtering."""

    def run(
        self,
        request: RetrievalRequest,
        results: list[SearchResult] | None = None,
    ) -> list[SearchResult]:
        return (results or [])[: request.top_k]
