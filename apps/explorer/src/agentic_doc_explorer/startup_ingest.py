"""Optional first-start ingest for deploy environments (Streamlit Cloud, Docker).

Indexes from a shipped corpus (``INGEST_SOURCE_DIR``) when the vector store is empty.
Does not download remote content — vendor the demo under ``corpora/`` instead.
"""

from __future__ import annotations

from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.ingest import IngestResult, resolve_ingest_settings, run_ingestion
from agentic_doc_rag.sparse import create_sparse_index
from agentic_doc_rag.vectorstore.factory import create_vector_store


def maybe_run_startup_ingest(settings: RagSettings) -> IngestResult | None:
    """Index the corpus when ``ingest_on_startup`` is set and the store is empty.

    Returns the ingest result when a run happened, otherwise ``None``.
    """
    if not settings.ingest_on_startup:
        return None

    vectorstore = create_vector_store(settings)
    if vectorstore.count() > 0:
        return None

    sparse_index = create_sparse_index(settings)
    return run_ingestion(
        vectorstore,
        sparse_index,
        resolve_ingest_settings(settings),
    )
