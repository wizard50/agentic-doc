from agentic_doc_rag.config import RagSettings, VectorStoreType
from agentic_doc_rag.vectorstore.base import VectorStore
from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore


def create_vector_store(settings: RagSettings) -> VectorStore:
    match settings.vector_store_type:
        case VectorStoreType.CHROMA:
            return ChromaVectorStore(settings.chroma_persist_dir, settings.chroma_collection_name)
