from pydantic import BaseModel, Field

from agentic_doc_rag.models import SearchMode

__all__ = ["RetrievalRequest", "SearchMode"]


class RetrievalRequest(BaseModel):
    query: str
    mode: SearchMode = SearchMode.SEMANTIC
    top_k: int = Field(default=5, ge=1)
    candidate_k: int = Field(default=20, ge=1)