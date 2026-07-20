"""Public API for the agentic-doc RAG package.

Prefer importing from this module for stable entry points. Submodules remain
available for advanced use (custom stages, parsers, etc.).

Exports are resolved lazily so `from agentic_doc_rag.config import ...` (and
similar submodule imports) do not pull the full dependency graph or create
import cycles (e.g. package → evaluation → retrieval → vectorstore → package).
"""

from __future__ import annotations

from typing import Any

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

# name -> (module, attribute)
_EXPORTS: dict[str, tuple[str, str]] = {
    "DocumentChunk": ("agentic_doc_rag.models", "DocumentChunk"),
    "EmbeddingModelMismatchError": (
        "agentic_doc_rag.embeddings",
        "EmbeddingModelMismatchError",
    ),
    "EmbeddingType": ("agentic_doc_rag.config", "EmbeddingType"),
    "Embeddings": ("agentic_doc_rag.embeddings", "Embeddings"),
    "EmptyVectorStoreError": ("agentic_doc_rag.evaluation", "EmptyVectorStoreError"),
    "EvalQuery": ("agentic_doc_rag.evaluation", "EvalQuery"),
    "EvalReport": ("agentic_doc_rag.evaluation", "EvalReport"),
    "IngestEmptyCorpusError": ("agentic_doc_rag.ingest", "IngestEmptyCorpusError"),
    "IngestResult": ("agentic_doc_rag.ingest", "IngestResult"),
    "IngestSettings": ("agentic_doc_rag.ingest", "IngestSettings"),
    "IngestSourceNotFoundError": ("agentic_doc_rag.ingest", "IngestSourceNotFoundError"),
    "MetadataFilter": ("agentic_doc_rag.retrieval", "MetadataFilter"),
    "RagSettings": ("agentic_doc_rag.config", "RagSettings"),
    "RetrievalRequest": ("agentic_doc_rag.retrieval", "RetrievalRequest"),
    "Retriever": ("agentic_doc_rag.retrieval", "Retriever"),
    "SearchMode": ("agentic_doc_rag.models", "SearchMode"),
    "SearchResult": ("agentic_doc_rag.models", "SearchResult"),
    "SparseIndex": ("agentic_doc_rag.sparse", "SparseIndex"),
    "VectorStore": ("agentic_doc_rag.vectorstore.base", "VectorStore"),
    "VectorStoreType": ("agentic_doc_rag.config", "VectorStoreType"),
    "create_embeddings": ("agentic_doc_rag.embeddings", "create_embeddings"),
    "create_retriever": ("agentic_doc_rag.retrieval", "create_retriever"),
    "create_sparse_index": ("agentic_doc_rag.sparse", "create_sparse_index"),
    "create_vector_store": ("agentic_doc_rag.vectorstore.factory", "create_vector_store"),
    "get_rag_settings": ("agentic_doc_rag.config", "get_rag_settings"),
    "get_tracer": ("agentic_doc_rag.observability", "get_tracer"),
    "ingest_settings_from_rag": ("agentic_doc_rag.ingest", "ingest_settings_from_rag"),
    "load_eval_dataset": ("agentic_doc_rag.evaluation", "load_eval_dataset"),
    "register_tracing": ("agentic_doc_rag.observability", "register_tracing"),
    "resolve_ingest_settings": ("agentic_doc_rag.ingest", "resolve_ingest_settings"),
    "run_ingestion": ("agentic_doc_rag.ingest.runner", "run_ingestion"),
    "run_retrieval_eval": ("agentic_doc_rag.evaluation", "run_retrieval_eval"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None

    from importlib import import_module

    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
