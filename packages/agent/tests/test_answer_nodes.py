from typing import TypeVar

from pydantic import BaseModel

from agentic_doc_agent.evaluation import FaithfulnessVerdict
from agentic_doc_agent.graphs.answer_models import AnswerDraft, PlanDraft
from agentic_doc_agent.graphs.answer_nodes import (
    citations_from_draft,
    run_answer_evaluate,
    run_answer_generate,
    run_answer_plan,
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
        plan: PlanDraft | None = None,
        verdict: FaithfulnessVerdict | None = None,
        error: Exception | None = None,
        plan_error: Exception | None = None,
        evaluate_error: Exception | None = None,
    ) -> None:
        self._draft = draft
        self._plan = plan
        self._verdict = verdict
        self._error = error
        self._plan_error = plan_error
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
        if schema is PlanDraft:
            if self._plan_error is not None:
                raise self._plan_error
            if self._plan is None:
                raise RuntimeError("FakeLlm has no plan configured")
            return schema.model_validate(self._plan.model_dump())
        if schema is FaithfulnessVerdict:
            if self._evaluate_error is not None:
                raise self._evaluate_error
            if self._verdict is None:
                raise RuntimeError("FakeLlm has no verdict configured")
            return schema.model_validate(self._verdict.model_dump())
        if self._error is not None:
            raise self._error
        if self._draft is None:
            raise RuntimeError("FakeLlm has no draft configured")
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


def test_run_answer_plan_happy_path() -> None:
    plan = PlanDraft(search_query="Rust ownership rules", rationale="Focus keywords")
    llm = FakeLlm(plan=plan)

    state = run_answer_plan(_state(), llm)

    assert state.error is None
    assert state.retrieve_query == "Rust ownership rules"
    assert state.steps[-1].kind is StepKind.PLAN
    assert state.steps[-1].name == "plan"
    assert state.steps[-1].payload["search_query"] == "Rust ownership rules"
    assert state.steps[-1].payload["rationale"] == "Focus keywords"
    assert len(llm.structured_calls) == 1
    assert llm.structured_calls[0][1] is PlanDraft


def test_run_answer_plan_disabled_is_noop() -> None:
    llm = FakeLlm(plan=PlanDraft(search_query="unused"))
    state = run_answer_plan(_state(), llm, enabled=False)

    assert state.retrieve_query is None
    assert state.steps == []
    assert llm.structured_calls == []


def test_run_answer_plan_skips_when_error_set() -> None:
    llm = FakeLlm(plan=PlanDraft(search_query="unused"))
    state = run_answer_plan(_state(error="prior"), llm)

    assert state.retrieve_query is None
    assert llm.structured_calls == []


def test_run_answer_plan_fail_soft_on_llm_error() -> None:
    llm = FakeLlm(plan_error=LlmRequestError("provider down"))
    state = run_answer_plan(_state(goal="What is ownership?"), llm)

    assert state.error is None
    # Fall back to the original goal so retrieve still has a concrete query.
    assert state.retrieve_query == "What is ownership?"
    assert state.steps[-1].kind is StepKind.PLAN
    assert "error" in state.steps[-1].payload
    assert state.steps[-1].payload.get("fallback") is True


def test_run_answer_retrieve_happy_path() -> None:
    hits = [_hit("a"), _hit("b")]
    tool = RetrieveTool(FakeRetriever(hits), default_top_k=3)
    state = run_answer_retrieve(_state(), tool)

    assert state.error is None
    assert [h.chunk.id for h in state.retrieved] == ["a", "b"]
    assert state.retrieve_rounds == 1
    assert state.needs_more_context is False
    assert len(state.steps) == 1
    assert state.steps[0].kind is StepKind.TOOL
    assert state.steps[0].name == "retrieve"
    assert state.steps[0].payload["count"] == 2
    assert state.steps[0].payload["query"] == "What is ownership?"
    assert state.steps[0].payload["round"] == 1
    assert state.steps[0].payload["merged_count"] == 2


def test_run_answer_retrieve_uses_retrieve_query() -> None:
    retriever = FakeRetriever([_hit("a")])
    tool = RetrieveTool(retriever, default_top_k=3)
    initial = _state().model_copy(update={"retrieve_query": "Rust ownership rules"})

    state = run_answer_retrieve(initial, tool)

    assert state.steps[0].payload["query"] == "Rust ownership rules"
    assert retriever.calls[0].query == "Rust ownership rules"
    assert state.retrieve_rounds == 1


def test_run_answer_retrieve_merges_across_rounds() -> None:
    first_tool = RetrieveTool(FakeRetriever([_hit("a"), _hit("b")]))
    after_first = run_answer_retrieve(_state(), first_tool)
    assert [h.chunk.id for h in after_first.retrieved] == ["a", "b"]
    assert after_first.retrieve_rounds == 1

    second = after_first.model_copy(
        update={
            "retrieve_query": "borrowing",
            "needs_more_context": True,
        }
    )
    second_tool = RetrieveTool(FakeRetriever([_hit("b", "B2"), _hit("c")]))
    after_second = run_answer_retrieve(second, second_tool)

    assert [h.chunk.id for h in after_second.retrieved] == ["a", "b", "c"]
    assert after_second.retrieved[1].chunk.text == "body"  # first-seen "b" kept
    assert after_second.retrieve_rounds == 2
    assert after_second.needs_more_context is False
    assert after_second.steps[-1].payload["round"] == 2
    assert after_second.steps[-1].payload["merged_count"] == 3
    assert after_second.steps[-1].payload["query"] == "borrowing"


def test_run_answer_retrieve_records_failure() -> None:
    tool = RetrieveTool(FakeRetriever(error=RuntimeError("index down")))
    state = run_answer_retrieve(_state(), tool)

    assert state.error is not None
    assert "retrieve failed" in state.error
    assert state.retrieved == []
    assert state.retrieve_rounds == 0
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
    initial = _state().model_copy(
        update={
            "retrieved": [_hit("a", "own"), _hit("b", "borrow")],
            "retrieve_rounds": 1,
        }
    )

    state = run_answer_generate(initial, llm)

    assert state.error is None
    assert state.draft_answer == draft.answer
    assert state.structured == draft.model_dump()
    assert [c.chunk_id for c in state.citations] == ["a", "b"]
    assert state.citations[0].source == "a.md"
    assert state.citations[0].section_path == "Sec a"
    assert state.needs_more_context is False
    assert state.steps[-1].kind is StepKind.GENERATE
    assert state.steps[-1].payload["citation_count"] == 2
    assert state.steps[-1].payload["context_sufficient"] is True
    assert state.steps[-1].payload["needs_more_context"] is False
    assert len(llm.structured_calls) == 1
    messages, schema = llm.structured_calls[0]
    assert schema is AnswerDraft
    assert messages[1].content  # user message includes context


def test_run_answer_generate_insufficient_with_follow_up() -> None:
    draft = AnswerDraft(
        answer="Context does not cover borrowing.",
        context_sufficient=False,
        follow_up_query="Rust borrowing and references",
    )
    llm = FakeLlm(draft)
    initial = _state().model_copy(
        update={"retrieved": [_hit("a")], "retrieve_rounds": 1, "retrieve_query": "ownership"}
    )

    state = run_answer_generate(initial, llm)

    assert state.needs_more_context is True
    assert state.retrieve_query == "Rust borrowing and references"
    assert state.draft_answer == draft.answer
    assert state.steps[-1].payload["needs_more_context"] is True
    assert state.steps[-1].payload["follow_up_query"] == "Rust borrowing and references"


def test_run_answer_generate_insufficient_without_follow_up_stops() -> None:
    draft = AnswerDraft(
        answer="Not enough context.",
        context_sufficient=False,
        follow_up_query=None,
    )
    llm = FakeLlm(draft)
    initial = _state().model_copy(update={"retrieved": [_hit("a")], "retrieve_rounds": 1})

    state = run_answer_generate(initial, llm)

    assert state.needs_more_context is False
    assert state.steps[-1].payload["needs_more_context"] is False


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


def test_run_answer_evaluate_happy_path() -> None:
    verdict = FaithfulnessVerdict(score=0.9, explanation="Well grounded.")
    llm = FakeLlm(verdict=verdict)
    initial = _state().model_copy(
        update={
            "retrieved": [_hit("a", "own")],
            "draft_answer": "Each value has one owner.",
        }
    )

    state = run_answer_evaluate(initial, llm)

    assert state.faithfulness == 0.9
    assert state.steps[-1].kind is StepKind.EVALUATE
    assert state.steps[-1].name == "evaluate"
    assert state.steps[-1].payload["faithfulness"] == 0.9
    assert state.steps[-1].payload["explanation"] == "Well grounded."
    assert len(llm.structured_calls) == 1
    assert llm.structured_calls[0][1] is FaithfulnessVerdict


def test_run_answer_evaluate_disabled_is_noop() -> None:
    llm = FakeLlm(verdict=FaithfulnessVerdict(score=1.0, explanation="n/a"))
    initial = _state().model_copy(update={"draft_answer": "answer"})

    state = run_answer_evaluate(initial, llm, enabled=False)

    assert state.faithfulness is None
    assert state.steps == []
    assert llm.structured_calls == []


def test_run_answer_evaluate_skips_when_error_set() -> None:
    llm = FakeLlm(verdict=FaithfulnessVerdict(score=1.0, explanation="n/a"))
    state = run_answer_evaluate(
        _state(error="prior").model_copy(update={"draft_answer": "answer"}),
        llm,
    )
    assert state.faithfulness is None
    assert llm.structured_calls == []


def test_run_answer_evaluate_skips_when_no_draft() -> None:
    llm = FakeLlm(verdict=FaithfulnessVerdict(score=1.0, explanation="n/a"))
    state = run_answer_evaluate(_state(), llm)
    assert state.faithfulness is None
    assert llm.structured_calls == []


def test_run_answer_evaluate_fail_soft_on_llm_error() -> None:
    llm = FakeLlm(evaluate_error=LlmRequestError("judge down"))
    initial = _state().model_copy(update={"draft_answer": "Some answer."})

    state = run_answer_evaluate(initial, llm)

    assert state.error is None
    assert state.draft_answer == "Some answer."
    assert state.faithfulness is None
    assert state.steps[-1].kind is StepKind.EVALUATE
    assert "error" in state.steps[-1].payload
