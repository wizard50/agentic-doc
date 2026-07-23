from typing import TypeVar

from pydantic import BaseModel

from agentic_doc_agent import (
    AgentRequest,
    AgentSettings,
    AgentStatus,
    ChatMessage,
    ChatResult,
    StepKind,
    WorkflowId,
    run_workflow,
)
from agentic_doc_agent.evaluation import FaithfulnessVerdict
from agentic_doc_agent.graphs.answer_models import AnswerDraft
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
        verdict: FaithfulnessVerdict | None = None,
        error: Exception | None = None,
        evaluate_error: Exception | None = None,
    ) -> None:
        self._draft = draft
        self._verdict = verdict or FaithfulnessVerdict(
            score=0.88,
            explanation="Mostly grounded.",
        )
        self._error = error
        self._evaluate_error = evaluate_error
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
        if schema is FaithfulnessVerdict:
            if self._evaluate_error is not None:
                raise self._evaluate_error
            return schema.model_validate(self._verdict.model_dump())
        if self._error is not None:
            raise self._error
        if self._draft is None:
            raise RuntimeError("FakeLlm has no draft configured")
        return schema.model_validate(self._draft.model_dump())


def _hit(chunk_id: str, text: str = "body") -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text=text,
            metadata={"source": f"{chunk_id}.md", "section_path": f"Sec {chunk_id}"},
        ),
        score=0.9,
    )


def test_run_workflow_answer_happy_path() -> None:
    draft = AnswerDraft(
        answer="Ownership means each value has a single owner.",
        citation_chunk_ids=["c1"],
    )
    result = run_workflow(
        AgentRequest(workflow=WorkflowId.ANSWER, goal="What is ownership?"),
        retrieve_tool=RetrieveTool(FakeRetriever([_hit("c1", "owner rules")])),
        llm=FakeLlm(
            draft,
            verdict=FaithfulnessVerdict(score=0.91, explanation="Grounded."),
        ),
        settings=AgentSettings(faithfulness_enabled=True),
    )

    assert result.status is AgentStatus.SUCCEEDED
    assert result.workflow is WorkflowId.ANSWER
    assert result.goal == "What is ownership?"
    assert result.answer == draft.answer
    assert result.structured == draft.model_dump()
    assert [c.chunk_id for c in result.citations] == ["c1"]
    assert len(result.retrieved) == 1
    assert [s.name for s in result.steps] == ["retrieve", "generate", "evaluate"]
    assert result.steps[0].kind is StepKind.TOOL
    assert result.steps[1].kind is StepKind.GENERATE
    assert result.steps[2].kind is StepKind.EVALUATE
    assert result.metrics.tool_calls == 1
    assert result.metrics.faithfulness == 0.91
    assert result.metrics.duration_ms is not None
    assert result.metrics.duration_ms >= 0
    assert result.error is None


def test_run_workflow_answer_faithfulness_disabled() -> None:
    draft = AnswerDraft(answer="Ownership rules apply.", citation_chunk_ids=["c1"])
    result = run_workflow(
        AgentRequest(goal="What is ownership?"),
        retrieve_tool=RetrieveTool(FakeRetriever([_hit("c1")])),
        llm=FakeLlm(draft),
        settings=AgentSettings(faithfulness_enabled=False),
    )

    assert result.status is AgentStatus.SUCCEEDED
    assert result.metrics.faithfulness is None
    assert [s.name for s in result.steps] == ["retrieve", "generate"]


def test_run_workflow_answer_faithfulness_fail_soft() -> None:
    from agentic_doc_agent.llm import LlmRequestError

    draft = AnswerDraft(answer="Ownership rules apply.", citation_chunk_ids=["c1"])
    result = run_workflow(
        AgentRequest(goal="What is ownership?"),
        retrieve_tool=RetrieveTool(FakeRetriever([_hit("c1")])),
        llm=FakeLlm(draft, evaluate_error=LlmRequestError("judge down")),
        settings=AgentSettings(faithfulness_enabled=True),
    )

    assert result.status is AgentStatus.SUCCEEDED
    assert result.answer == draft.answer
    assert result.metrics.faithfulness is None
    assert [s.name for s in result.steps] == ["retrieve", "generate", "evaluate"]
    assert "error" in result.steps[-1].payload


def test_run_workflow_answer_retrieve_failure() -> None:
    result = run_workflow(
        AgentRequest(goal="x"),
        retrieve_tool=RetrieveTool(FakeRetriever(error=RuntimeError("down"))),
        llm=FakeLlm(AnswerDraft(answer="unused", citation_chunk_ids=[])),
        settings=AgentSettings(faithfulness_enabled=True),
    )

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert "retrieve failed" in result.error
    assert result.answer is None
    assert result.metrics.faithfulness is None
    # evaluate no-ops on error, so no evaluate step is recorded
    assert len(result.steps) == 1
    assert result.steps[0].name == "retrieve"


def test_run_workflow_answer_generate_failure() -> None:
    from agentic_doc_agent.llm import LlmRequestError

    result = run_workflow(
        AgentRequest(goal="x"),
        retrieve_tool=RetrieveTool(FakeRetriever([_hit("a")])),
        llm=FakeLlm(error=LlmRequestError("provider down")),
        settings=AgentSettings(faithfulness_enabled=True),
    )

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert "generate failed" in result.error
    assert len(result.retrieved) == 1
    assert result.metrics.faithfulness is None
    assert [s.name for s in result.steps] == ["retrieve", "generate"]


def test_run_workflow_unimplemented_returns_failed() -> None:
    result = run_workflow(
        AgentRequest(workflow=WorkflowId.COMPARE, goal="compare A and B"),
        retrieve_tool=RetrieveTool(FakeRetriever()),
        llm=FakeLlm(AnswerDraft(answer="nope", citation_chunk_ids=[])),
    )

    assert result.status is AgentStatus.FAILED
    assert result.error is not None
    assert "not implemented" in result.error
    assert result.steps == []
