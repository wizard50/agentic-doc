"""Public API for the agentic-doc RAG package.

Prefer importing from this module for stable entry points. Submodules remain
available for advanced use (custom stages, parsers, etc.).
"""

from __future__ import annotations

from typing import Any

from agentic_doc_rag.config import EmbeddingType, RagSettings, VectorStoreType, get_rag_settings
from agentic_doc_rag.embeddings import (
    EmbeddingModelMismatchError,
    Embeddings,
    create_embeddings,
)
from agentic_doc_rag.evaluation import (
    EmptyVectorStoreError,
    EvalQuery,
    EvalReport,
    load_eval_dataset,
    run_retrieval_eval,
)
from agentic_doc_rag.ingest import (
    IngestEmptyCorpusError,
    IngestResult,
    IngestSettings,
    IngestSourceNotFoundError,
    ingest_settings_from_rag,
    resolve_ingest_settings,
)
from agentic_doc_rag.models import DocumentChunk, SearchMode, SearchResult
from agentic_doc_rag.observability import get_tracer, register_tracing
from agentic_doc_rag.retrieval import (
    MetadataFilter,
    RetrievalRequest,
    Retriever,
    create_retriever,
)
from agentic_doc_rag.sparse import SparseIndex, create_sparse_index
from agentic_doc_rag.vectorstore.base import VectorStore
from agentic_doc_rag.vectorstore.factory import create_vector_store

__all__ = [
    "DocumentChunk",
    "EmbeddingModelMismatchError",
    "EmbeddingType",
    "Embeddings",
    "EmptyVectorStoreError",
    "EvalQuery",
    "EvalReport",
    "IngestEmptyCorpusError",
    "IngestResult",
    "IngestSettings",
    "IngestSourceNotFoundError",
    "MetadataFilter",
    "RagSettings",
    "RetrievalRequest",
    "Retriever",
    "SearchMode",
    "SearchResult",
    "SparseIndex",
    "VectorStore",
    "VectorStoreType",
    "create_embeddings",
    "create_retriever",
    "create_sparse_index",
    "create_vector_store",
    "get_rag_settings",
    "get_tracer",
    "ingest_settings_from_rag",
    "load_eval_dataset",
    "register_tracing",
    "resolve_ingest_settings",
    "run_ingestion",
    "run_retrieval_eval",
]


def __getattr__(name: str) -> Any:
    # Lazy: avoids import cycles with parsers → ingest.models → package surface.
    if name == "run_ingestion":
        from agentic_doc_rag.ingest.runner import run_ingestion

        return run_ingestion
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
