from pydantic import BaseModel, Field

from agentic_doc_rag.evaluation.metrics import compute_report
from agentic_doc_rag.evaluation.models import EvalQuery, EvalReport
from agentic_doc_rag.models import SearchResult
from agentic_doc_rag.vectorstore.base import VectorStore


class EmptyVectorStoreError(Exception):
    """Raised when evaluation is run against an empty vector store."""


class RetrievalEvalRun(BaseModel):
    """Deterministic retrieval evaluation output including raw search results."""

    report: EvalReport
    results_by_query_id: dict[str, list[SearchResult]] = Field(default_factory=dict)


def run_retrieval_eval(
    vectorstore: VectorStore,
    queries: list[EvalQuery],
    *,
    top_k: int,
    dataset_name: str,
) -> RetrievalEvalRun:
    """Run golden queries against a vector store and compute retrieval metrics."""
    if vectorstore.count() == 0:
        msg = "Vector store collection is empty. Index the corpus first:\n  uv run explorer ingest"
        raise EmptyVectorStoreError(msg)

    results_by_query_id = {query.id: vectorstore.search(query.query, k=top_k) for query in queries}
    report = compute_report(
        queries,
        results_by_query_id,
        top_k=top_k,
        dataset_name=dataset_name,
    )
    return RetrievalEvalRun(report=report, results_by_query_id=results_by_query_id)
