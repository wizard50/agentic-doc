from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_doc_rag.models import SearchMode


class VectorStoreType(StrEnum):
    CHROMA = "chroma"


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
    bm25_persist_dir: Path = Field(
        default=Path("data/bm25"), description="Directory to persist the BM25 sparse index"
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


def get_rag_settings() -> RagSettings:
    return RagSettings()
