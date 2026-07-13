from agentic_doc_rag.evaluation.metrics import compute_report
from agentic_doc_rag.evaluation.models import EvalQuery, EvalReport
from agentic_doc_rag.vectorstore.base import VectorStore


class EmptyVectorStoreError(Exception):
    """Raised when evaluation is run against an empty vector store."""


def run_retrieval_eval(
    vectorstore: VectorStore,
    queries: list[EvalQuery],
    *,
    top_k: int,
    dataset_name: str,
) -> EvalReport:
    """Run golden queries against a vector store and compute retrieval metrics."""
    if vectorstore.count() == 0:
        msg = (
            "Vector store collection is empty. Index the corpus first:\n"
            "  uv run explorer ingest"
        )
        raise EmptyVectorStoreError(msg)

    results_by_query_id = {
        query.id: vectorstore.search(query.query, k=top_k) for query in queries
    }
    return compute_report(
        queries,
        results_by_query_id,
        top_k=top_k,
        dataset_name=dataset_name,
    )