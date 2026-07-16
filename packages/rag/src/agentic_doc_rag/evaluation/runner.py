from pydantic import BaseModel, Field

from agentic_doc_rag.evaluation.metrics import compute_report
from agentic_doc_rag.evaluation.models import EvalQuery, EvalReport
from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.retrieval.models import RetrievalRequest, SearchMode
from agentic_doc_rag.retrieval.protocols import Retriever


class EmptyVectorStoreError(Exception):
    """Raised when evaluation is run against an empty vector store."""


class RetrievalEvalRun(BaseModel):
    """Deterministic retrieval evaluation output including raw search results."""

    report: EvalReport
    results_by_query_id: dict[str, list[SearchResult]] = Field(default_factory=dict)


def run_retrieval_eval(
    retriever: Retriever,
    queries: list[EvalQuery],
    *,
    top_k: int,
    dataset_name: str,
    search_mode: SearchMode = SearchMode.SEMANTIC,
    candidate_k: int = 20,
    rerank: bool | None = None,
) -> RetrievalEvalRun:
    """Run golden queries against a retriever and compute retrieval metrics."""
    if retriever.count() == 0:
        msg = "Vector store collection is empty. Index the corpus first:\n  uv run explorer ingest"
        raise EmptyVectorStoreError(msg)

    results_by_query_id = {
        query.id: retriever.retrieve(
            RetrievalRequest(
                query=query.query,
                top_k=top_k,
                mode=search_mode,
                candidate_k=candidate_k,
                rerank=rerank,
            )
        )
        for query in queries
    }
    report = compute_report(
        queries,
        results_by_query_id,
        top_k=top_k,
        dataset_name=dataset_name,
    )
    return RetrievalEvalRun(report=report, results_by_query_id=results_by_query_id)
