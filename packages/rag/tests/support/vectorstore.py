from pathlib import Path

from agentic_doc_rag.embeddings.chroma_default import ChromaDefaultEmbeddings
from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore


def chroma_vector_store(persist_dir: Path, collection_name: str) -> ChromaVectorStore:
    return ChromaVectorStore(persist_dir, collection_name, ChromaDefaultEmbeddings())