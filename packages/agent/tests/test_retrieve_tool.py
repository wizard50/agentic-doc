from typing import Any

import pytest
from pydantic import ValidationError

from agentic_doc_agent.tools import RetrieveArgs, RetrieveResult, RetrieveTool
from agentic_doc_agent.tools.protocols import AgentTool
from agentic_doc_rag.models import DocumentChunk, SearchMode, SearchResult
from agentic_doc_rag.retrieval import MetadataFilter, RetrievalRequest


class FakeRetriever:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []
        self.calls: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        self.calls.append(request)
        return list(self._results)

    def count(self) -> int:
        return len(self._results)


def _hit(chunk_id: str, text: str = "body", *, score: float = 0.9) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text=text,
            metadata={"source": f"{chunk_id}.md", "section_path": "Ch 1"},
        ),
        score=score,
    )


def test_retrieve_tool_satisfies_agent_tool_protocol() -> None:
    tool = RetrieveTool(FakeRetriever())
    assert isinstance(tool, AgentTool)


def test_invoke_builds_request_with_defaults() -> None:
    retriever = FakeRetriever(results=[_hit("c1")])
    tool = RetrieveTool(retriever, default_top_k=7, default_search_mode=SearchMode.HYBRID)

    result = tool.invoke(query="What is ownership?")

    assert isinstance(result, RetrieveResult)
    assert result.query == "What is ownership?"
    assert result.count == 1
    assert result.results[0].chunk.id == "c1"
    assert len(retriever.calls) == 1
    request = retriever.calls[0]
    assert request.query == "What is ownership?"
    assert request.top_k == 7
    assert request.mode is SearchMode.HYBRID
    assert request.filters is None
    assert request.rerank is None


def test_invoke_overrides_defaults() -> None:
    retriever = FakeRetriever(results=[_hit("a"), _hit("b")])
    tool = RetrieveTool(retriever, default_top_k=5, default_search_mode=SearchMode.SEMANTIC)
    filters = MetadataFilter(source_contains="ch04")

    result = tool.invoke(
        query="borrowing",
        top_k=3,
        search_mode=SearchMode.KEYWORD,
        filters=filters,
        rerank=True,
    )

    assert result.count == 2
    request = retriever.calls[0]
    assert request.top_k == 3
    assert request.mode is SearchMode.KEYWORD
    assert request.filters == filters
    assert request.rerank is True


def test_invoke_args_round_trip() -> None:
    retriever = FakeRetriever(results=[_hit("x", text="ownership rules")])
    tool = RetrieveTool(retriever)

    args = RetrieveArgs(query="ownership", top_k=2, search_mode=SearchMode.HYBRID)
    result = tool.invoke_args(args)

    assert result.model_dump()["count"] == 1
    assert result.results[0].chunk.text == "ownership rules"
    assert retriever.calls[0].top_k == 2
    assert retriever.calls[0].mode is SearchMode.HYBRID


def test_empty_query_rejected() -> None:
    tool = RetrieveTool(FakeRetriever())
    with pytest.raises(ValidationError):
        tool.invoke(query="")


def test_default_top_k_must_be_positive() -> None:
    with pytest.raises(ValueError, match="default_top_k"):
        RetrieveTool(FakeRetriever(), default_top_k=0)


def test_invoke_matches_agent_tool_protocol_signature() -> None:
    """RetrieveTool is assignable to AgentTool (invoke takes **kwargs)."""
    retriever = FakeRetriever(results=[_hit("p")])
    tool: AgentTool = RetrieveTool(retriever)

    out: Any = tool.invoke(query="protocol path")
    assert out.count == 1


def test_invoke_rejects_unknown_kwargs() -> None:
    tool = RetrieveTool(FakeRetriever())
    with pytest.raises(ValidationError):
        tool.invoke(query="ok", unknown_flag=True)
