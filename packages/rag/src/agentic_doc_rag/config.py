from enum import StrEnum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


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


def get_rag_settings() -> RagSettings:
    return RagSettings()
