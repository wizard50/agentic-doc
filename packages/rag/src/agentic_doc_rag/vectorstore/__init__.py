"""Vector store protocol and backend factory."""

from agentic_doc_rag.vectorstore.base import VectorStore
from agentic_doc_rag.vectorstore.factory import create_vector_store

__all__ = [
    "VectorStore",
    "create_vector_store",
]
