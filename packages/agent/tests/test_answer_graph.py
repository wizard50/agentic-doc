from typing import TypeVar

from pydantic import BaseModel

from agentic_doc_agent.evaluation import FaithfulnessVerdict
from agentic_doc_agent.graphs.answer import build_answer_graph, route_after_generate
from agentic_doc_agent.graphs.answer_models import AnswerDraft, PlanDraft
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.llm import ChatMessage, ChatResult
from agentic_doc_agent.models import AgentRequest, StepKind, WorkflowId
from agentic_doc_agent.tools import RetrieveTool
from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval import RetrievalRequest

T = TypeVar("T", bound=BaseModel)


class FakeRetriever:
    def __init__(self, results: list[SearchResult] | None = None) -> None:
        self._results = results or []
        self.calls: list[RetrievalRequest] = []
        self.results_by_query: dict[str, list[SearchResult]] = {}

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        self.calls.append(request)
        if request.query in self.results_by_query:
            return list(self.results_by_query[request.query])
        return list(self._results)

    def count(self) -> int:
        return len(self._results)


class FakeLlm:
    def __init__(
        self,
        draft: AnswerDraft,
        *,
        plan: PlanDraft | None = None,
        verdict: FaithfulnessVerdict | None = None,
        drafts: list[AnswerDraft] | None = None,
    ) -> None:
        self._draft = draft
        self._drafts = list(drafts) if drafts is not None else None
        self._draft_index = 0
        self._plan = plan or PlanDraft(search_query="planned ownership query")
        self._verdict = verdict or FaithfulnessVerdict(
            score=0.8,
            explanation="Supported by context.",
        )
        self.structured_calls: list[type[BaseModel]] = []

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
        self.structured_calls.append(schema)
        if schema is PlanDraft:
            return schema.model_validate(self._plan.model_dump())
        if schema is FaithfulnessVerdict:
            return schema.model_validate(self._verdict.model_dump())
        if self._drafts is not None:
            if self._draft_index >= len(self._drafts):
                raise RuntimeError("FakeLlm has no more answer drafts")
            draft = self._drafts[self._draft_index]
            self._draft_index += 1
            return schema.model_validate(draft.model_dump())
        return schema.model_validate(self._draft.model_dump())


def _hit(chunk_id: str = "c1", text: str = "Rust ownership rules") -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text=text,
            metadata={"source": "ch04.md", "section_path": "Ownership"},
        ),
        score=0.95,
    )


def _request(goal: str = "ownership?") -> AgentRequest:
    return AgentRequest(workflow=WorkflowId.ANSWER, goal=goal)


def test_route_after_generate_re_retrieve_when_needed() -> None:
    state = AgentGraphState(
        request=_request(),
        needs_more_context=True,
        retrieve_rounds=1,
        retrieve_query="borrowing",
    )
    assert route_after_generate(state, max_tool_rounds=5) == "retrieve"


def test_route_after_generate_stops_at_max_rounds() -> None:
    state = AgentGraphState(
        request=_request(),
        needs_more_context=True,
        retrieve_rounds=2,
        retrieve_query="borrowing",
    )
    assert route_after_generate(state, max_tool_rounds=2) == "evaluate"


def test_route_after_generate_stops_without_follow_up() -> None:
    state = AgentGraphState(
        request=_request(),
        needs_more_context=True,
        retrieve_rounds=1,
        retrieve_query=None,
    )
    assert route_after_generate(state, max_tool_rounds=5) == "evaluate"


def test_build_answer_graph_plan_retrieve_generate_evaluate() -> None:
    llm = FakeLlm(
        AnswerDraft(answer="Each value has an owner.", citation_chunk_ids=["c1"]),
        plan=PlanDraft(search_query="ownership rules"),
        verdict=FaithfulnessVerdict(score=0.95, explanation="Grounded."),
    )
    retriever = FakeRetriever([_hit()])
    graph = build_answer_graph(
        RetrieveTool(retriever),
        llm,
        faithfulness_enabled=True,
        plan_enabled=True,
    )

    raw = graph.invoke(AgentGraphState(request=_request()))
    state = AgentGraphState.model_validate(raw)

    assert state.error is None
    assert state.draft_answer == "Each value has an owner."
    assert [c.chunk_id for c in state.citations] == ["c1"]
    assert state.faithfulness == 0.95
    assert [s.kind for s in state.steps] == [
        StepKind.PLAN,
        StepKind.TOOL,
        StepKind.GENERATE,
        StepKind.EVALUATE,
    ]
    assert retriever.calls[0].query == "ownership rules"
    assert llm.structured_calls == [PlanDraft, AnswerDraft, FaithfulnessVerdict]


def test_build_answer_graph_plan_disabled_skips_plan_step() -> None:
    llm = FakeLlm(AnswerDraft(answer="Each value has an owner.", citation_chunk_ids=["c1"]))
    retriever = FakeRetriever([_hit()])
    graph = build_answer_graph(
        RetrieveTool(retriever),
        llm,
        faithfulness_enabled=False,
        plan_enabled=False,
    )

    raw = graph.invoke(AgentGraphState(request=_request(goal="raw goal")))
    state = AgentGraphState.model_validate(raw)

    assert state.faithfulness is None
    assert [s.kind for s in state.steps] == [StepKind.TOOL, StepKind.GENERATE]
    assert retriever.calls[0].query == "raw goal"
    assert llm.structured_calls == [AnswerDraft]


def test_build_answer_graph_re_retrieve_once_then_evaluate() -> None:
    drafts = [
        AnswerDraft(
            answer="Need more on borrowing.",
            context_sufficient=False,
            follow_up_query="Rust borrowing",
            citation_chunk_ids=["c1"],
        ),
        AnswerDraft(
            answer="Borrowing is temporary access.",
            context_sufficient=True,
            citation_chunk_ids=["c1", "c2"],
        ),
    ]
    llm = FakeLlm(
        drafts[1],
        plan=PlanDraft(search_query="ownership"),
        drafts=drafts,
        verdict=FaithfulnessVerdict(score=0.9, explanation="ok"),
    )
    retriever = FakeRetriever()
    retriever.results_by_query = {
        "ownership": [_hit("c1", "ownership")],
        "Rust borrowing": [_hit("c2", "borrowing")],
    }
    graph = build_answer_graph(
        RetrieveTool(retriever),
        llm,
        faithfulness_enabled=True,
        plan_enabled=True,
        max_tool_rounds=5,
    )

    raw = graph.invoke(AgentGraphState(request=_request()))
    state = AgentGraphState.model_validate(raw)

    assert state.error is None
    assert state.draft_answer == "Borrowing is temporary access."
    assert [h.chunk.id for h in state.retrieved] == ["c1", "c2"]
    assert state.retrieve_rounds == 2
    assert [s.name for s in state.steps] == [
        "plan",
        "retrieve",
        "generate",
        "retrieve",
        "generate",
        "evaluate",
    ]
    assert [c.query for c in retriever.calls] == ["ownership", "Rust borrowing"]
    assert llm.structured_calls == [
        PlanDraft,
        AnswerDraft,
        AnswerDraft,
        FaithfulnessVerdict,
    ]


def test_build_answer_graph_max_tool_rounds_blocks_re_retrieve() -> None:
    llm = FakeLlm(
        AnswerDraft(
            answer="Still insufficient.",
            context_sufficient=False,
            follow_up_query="another query",
            citation_chunk_ids=["c1"],
        ),
        plan=PlanDraft(search_query="first"),
    )
    retriever = FakeRetriever([_hit()])
    graph = build_answer_graph(
        RetrieveTool(retriever),
        llm,
        faithfulness_enabled=False,
        plan_enabled=True,
        max_tool_rounds=1,
    )

    raw = graph.invoke(AgentGraphState(request=_request()))
    state = AgentGraphState.model_validate(raw)

    assert state.retrieve_rounds == 1
    assert len(retriever.calls) == 1
    assert [s.name for s in state.steps] == ["plan", "retrieve", "generate"]
    assert state.needs_more_context is True
    assert state.draft_answer == "Still insufficient."
