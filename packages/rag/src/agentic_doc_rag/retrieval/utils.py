from agentic_doc_rag.retrieval.models import RetrievalRequest


def pool_k(request: RetrievalRequest) -> int:
    """Prefetch size used before metadata filtering and final top-k trimming."""
    return max(request.candidate_k, request.top_k)
