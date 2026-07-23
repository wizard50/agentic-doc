from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.sparse.bm25 import Bm25Index
from agentic_doc_rag.sparse.protocols import SparseIndex


def create_sparse_index(settings: RagSettings) -> SparseIndex:
    return Bm25Index(settings.bm25_persist_dir)
