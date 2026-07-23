from pydantic import BaseModel, Field

from agentic_doc_rag.models import SearchMode

__all__ = ["MetadataFilter", "RetrievalRequest", "SearchMode"]


class MetadataFilter(BaseModel):
    source_contains: str | None = Field(
        default=None, description="Substring match on chunk source path"
    )
    source_suffix: str | None = Field(
        default=None,
        description="Expected source path suffix (e.g. ch04-02-references-and-borrowing.md)",
    )
    section_path_contains: str | None = Field(
        default=None,
        description="Substring match on chunk section_path metadata",
    )


class RetrievalRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.SEMANTIC
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)
    filters: MetadataFilter | None = None
    rerank: bool | None = None
