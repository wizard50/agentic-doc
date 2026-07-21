"""Retrieve tool — wraps the M1 RAG retriever for agent workflows."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from agentic_doc_rag.models import SearchMode, SearchResult
from agentic_doc_rag.retrieval import MetadataFilter, RetrievalRequest, Retriever

TOOL_NAME = "retrieve"
TOOL_DESCRIPTION = (
    "Search the documentation corpus for passages relevant to a query. "
    "Returns ranked chunks with source and section metadata for grounding answers."
)


class RetrieveArgs(BaseModel):
    """Arguments accepted by the retrieve tool."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(..., min_length=1, description="Search query")
    top_k: int | None = Field(default=None, ge=1, description="Max results (uses tool default if omitted)")
    search_mode: SearchMode | None = Field(
        default=None,
        description="semantic | keyword | hybrid (uses tool default if omitted)",
    )
    filters: MetadataFilter | None = None
    rerank: bool | None = Field(
        default=None,
        description="Override rerank for this call; None uses pipeline default",
    )


class RetrieveResult(BaseModel):
    """Structured tool output for graph state and LLM tool messages."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    count: int = Field(default=0, ge=0)


class RetrieveTool:
    """Agent tool that runs the configured RAG retrieval pipeline.

    ``invoke`` takes ``**kwargs`` so the class satisfies :class:`AgentTool`.
    Prefer :meth:`invoke_args` when you already have a typed :class:`RetrieveArgs`.
    """

    name: str = TOOL_NAME
    description: str = TOOL_DESCRIPTION

    def __init__(
        self,
        retriever: Retriever,
        *,
        default_top_k: int = 5,
        default_search_mode: SearchMode = SearchMode.SEMANTIC,
    ) -> None:
        if default_top_k < 1:
            raise ValueError("default_top_k must be >= 1")
        self._retriever = retriever
        self._default_top_k = default_top_k
        self._default_search_mode = default_search_mode

    def invoke(self, **kwargs: Any) -> RetrieveResult:
        """Run retrieval from keyword args (validated as :class:`RetrieveArgs`)."""
        return self.invoke_args(RetrieveArgs.model_validate(kwargs))

    def invoke_args(self, args: RetrieveArgs) -> RetrieveResult:
        """Run retrieval from a validated args model."""
        request = RetrievalRequest(
            query=args.query,
            mode=args.search_mode if args.search_mode is not None else self._default_search_mode,
            top_k=args.top_k if args.top_k is not None else self._default_top_k,
            filters=args.filters,
            rerank=args.rerank,
        )
        results = self._retriever.retrieve(request)
        return RetrieveResult(query=args.query, results=results, count=len(results))
