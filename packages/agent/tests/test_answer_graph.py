from typing import TypeVar

from pydantic import BaseModel

from agentic_doc_agent.graphs.answer import build_answer_graph
from agentic_doc_agent.graphs.answer_models import AnswerDraft
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.llm import ChatMessage, ChatResult
from agentic_doc_agent.models import AgentRequest, StepKind, WorkflowId
from agentic_doc_agent.tools import RetrieveTool
from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval import RetrievalRequest

T = TypeVar("T", bound=BaseModel)


class FakeRetriever:
    def __init__(self, results: list[SearchResult]) -> None:
        self._results = results
        self.calls: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        self.calls.append(request)
        return list(self._results)

    def count(self) -> int:
        return len(self._results)


class FakeLlm:
    def __init__(self, draft: AnswerDraft) -> None:
        self._draft = draft

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def complete_structured(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> T:
        return schema.model_validate(self._draft.model_dump())


def test_build_answer_graph_runs_retrieve_then_generate() -> None:
    hit = SearchResult(
        chunk=DocumentChunk(
            id="c1",
            text="Rust ownership rules",
            metadata={"source": "ch04.md", "section_path": "Ownership"},
        ),
        score=0.95,
    )
    graph = build_answer_graph(
        RetrieveTool(FakeRetriever([hit])),
        FakeLlm(AnswerDraft(answer="Each value has an owner.", citation_chunk_ids=["c1"])),
    )

    raw = graph.invoke(
        AgentGraphState(request=AgentRequest(workflow=WorkflowId.ANSWER, goal="ownership?"))
    )
    state = AgentGraphState.model_validate(raw)

    assert state.error is None
    assert state.draft_answer == "Each value has an owner."
    assert [c.chunk_id for c in state.citations] == ["c1"]
    assert [s.kind for s in state.steps] == [StepKind.TOOL, StepKind.GENERATE]
