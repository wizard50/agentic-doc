from typing import TypeVar

from pydantic import BaseModel

from agentic_doc_agent.graphs.answer_models import AnswerDraft
from agentic_doc_agent.graphs.answer_nodes import (
    citations_from_draft,
    run_answer_generate,
    run_answer_retrieve,
)
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.llm import ChatMessage, ChatResult, LlmRequestError
from agentic_doc_agent.models import AgentRequest, StepKind, WorkflowId
from agentic_doc_agent.tools import RetrieveTool
from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval import RetrievalRequest

T = TypeVar("T", bound=BaseModel)


class FakeRetriever:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._results = results or []
        self._error = error
        self.calls: list[RetrievalRequest] = []

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return list(self._results)

    def count(self) -> int:
        return len(self._results)


class FakeLlm:
    def __init__(
        self,
        draft: AnswerDraft | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._draft = draft
        self._error = error
        self.structured_calls: list[tuple[list[ChatMessage], type[BaseModel]]] = []

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
        self.structured_calls.append((messages, schema))
        if self._error is not None:
            raise self._error
        if self._draft is None:
            raise RuntimeError("FakeLlm has no draft configured")
        # Re-validate so the return type is schema/T, not a fixed AnswerDraft.
        return schema.model_validate(self._draft.model_dump())


def _hit(chunk_id: str, text: str = "body", *, score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text=text,
            metadata={"source": f"{chunk_id}.md", "section_path": f"Sec {chunk_id}"},
        ),
        score=score,
    )


def _state(*, goal: str = "What is ownership?", error: str | None = None) -> AgentGraphState:
    return AgentGraphState(
        request=AgentRequest(workflow=WorkflowId.ANSWER, goal=goal),
        error=error,
    )


def test_run_answer_retrieve_happy_path() -> None:
    hits = [_hit("a"), _hit("b")]
    tool = RetrieveTool(FakeRetriever(hits), default_top_k=3)
    state = run_answer_retrieve(_state(), tool)

    assert state.error is None
    assert [h.chunk.id for h in state.retrieved] == ["a", "b"]
    assert len(state.steps) == 1
    assert state.steps[0].kind is StepKind.TOOL
    assert state.steps[0].name == "retrieve"
    assert state.steps[0].payload["count"] == 2
    assert state.steps[0].payload["query"] == "What is ownership?"


def test_run_answer_retrieve_records_failure() -> None:
    tool = RetrieveTool(FakeRetriever(error=RuntimeError("index down")))
    state = run_answer_retrieve(_state(), tool)

    assert state.error is not None
    assert "retrieve failed" in state.error
    assert state.retrieved == []
    assert state.steps[0].kind is StepKind.TOOL
    assert "error" in state.steps[0].payload


def test_run_answer_retrieve_skips_when_error_set() -> None:
    tool = RetrieveTool(FakeRetriever([_hit("x")]))
    state = run_answer_retrieve(_state(error="already failed"), tool)
    assert state.retrieved == []
    assert state.steps == []


def test_run_answer_generate_happy_path() -> None:
    draft = AnswerDraft(
        answer="Ownership ensures each value has one owner.",
        citation_chunk_ids=["a", "missing", "a", "b"],
    )
    llm = FakeLlm(draft)
    initial = _state().model_copy(update={"retrieved": [_hit("a", "own"), _hit("b", "borrow")]})

    state = run_answer_generate(initial, llm)

    assert state.error is None
    assert state.draft_answer == draft.answer
    assert state.structured == draft.model_dump()
    assert [c.chunk_id for c in state.citations] == ["a", "b"]
    assert state.citations[0].source == "a.md"
    assert state.citations[0].section_path == "Sec a"
    assert state.steps[-1].kind is StepKind.GENERATE
    assert state.steps[-1].payload["citation_count"] == 2
    assert len(llm.structured_calls) == 1
    messages, schema = llm.structured_calls[0]
    assert schema is AnswerDraft
    assert messages[1].content  # user message includes context


def test_run_answer_generate_llm_error() -> None:
    llm = FakeLlm(error=LlmRequestError("provider down"))
    initial = _state().model_copy(update={"retrieved": [_hit("a")]})

    state = run_answer_generate(initial, llm)

    assert state.error is not None
    assert "generate failed" in state.error
    assert state.draft_answer is None
    assert state.steps[-1].kind is StepKind.GENERATE


def test_run_answer_generate_skips_when_error_set() -> None:
    llm = FakeLlm(AnswerDraft(answer="nope", citation_chunk_ids=[]))
    state = run_answer_generate(_state(error="prior"), llm)
    assert state.draft_answer is None
    assert llm.structured_calls == []


def test_citations_from_draft_drops_unknown_and_dedupes() -> None:
    draft = AnswerDraft(answer="x", citation_chunk_ids=["z", "a", "a", "nope"])
    citations = citations_from_draft(draft, [_hit("a"), _hit("b")])
    assert [c.chunk_id for c in citations] == ["a"]
