from enum import StrEnum

from pydantic import BaseModel, Field


class SearchMode(StrEnum):
    SEMANTIC = "semantic"
    KEYWORD = "keyword"
    HYBRID = "hybrid"


class RetrievalRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.SEMANTIC
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)