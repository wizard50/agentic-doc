from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_doc_rag.models import SearchMode


class VectorStoreType(StrEnum):
    CHROMA = "chroma"


class EmbeddingType(StrEnum):
    CHROMA_DEFAULT = "chroma_default"
    SENTENCE_TRANSFORMERS = "sentence_transformers"


class RagSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    vector_store_type: VectorStoreType = Field(
        default=VectorStoreType.CHROMA, description="Which vector store to use"
    )

    chroma_persist_dir: Path = Field(
        default=Path("data/chroma"), description="Directory to persist Chroma vector store"
    )
    chroma_collection_name: str = Field(
        default="rust_book", description="Name of the Chroma collection"
    )
    embedding_type: EmbeddingType = Field(
        default=EmbeddingType.CHROMA_DEFAULT,
        description="Embedding backend for semantic search",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Embedding model name (sentence-transformers backend)",
    )
    bm25_persist_dir: Path = Field(
        default=Path("data/bm25"), description="Directory to persist the BM25 sparse index"
    )
    ingest_source_dir: Path = Field(
        default=Path("corpora/rust-book/src"),
        description="Root directory of markdown files to index (shipped demo corpus by default)",
    )
    ingest_skip_files: str = Field(
        default="SUMMARY.md,title-page.md",
        description="Comma-separated markdown filenames to skip during ingest",
    )
    ingest_on_startup: bool = Field(
        default=False,
        description=(
            "When true, apps may index the corpus on first start if the vector store is empty "
            "(Streamlit Cloud / Docker). Set via INGEST_ON_STARTUP."
        ),
    )
    search_mode: SearchMode = Field(
        default=SearchMode.SEMANTIC,
        description="Default retrieval mode (semantic, keyword, or hybrid)",
    )
    candidate_k: int = Field(
        default=20,
        ge=1,
        description="Prefetch size for hybrid search before fusion",
    )
    rerank_enabled: bool = Field(
        default=False,
        description="Enable cross-encoder reranking by default",
    )
    rerank_model: str = Field(
        default="cross-encoder/ms-marco-MiniLM-L-6-v2",
        description="Cross-encoder model used for reranking",
    )


def get_rag_settings() -> RagSettings:
    return RagSettings()
