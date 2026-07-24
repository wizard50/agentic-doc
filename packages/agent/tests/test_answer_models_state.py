import pytest
from pydantic import ValidationError

from agentic_doc_agent.graphs.answer_models import AnswerDraft, PlanDraft
from agentic_doc_agent.graphs.answer_nodes import merge_retrieved
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.models import AgentRequest, WorkflowId
from agentic_doc_rag.models import DocumentChunk, SearchResult


def _hit(chunk_id: str, text: str = "body", *, score: float = 0.8) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text=text,
            metadata={"source": f"{chunk_id}.md"},
        ),
        score=score,
    )


def test_plan_draft_requires_search_query() -> None:
    with pytest.raises(ValidationError):
        PlanDraft(search_query="")


def test_plan_draft_defaults_rationale_none() -> None:
    plan = PlanDraft(search_query="Rust ownership rules")
    assert plan.search_query == "Rust ownership rules"
    assert plan.rationale is None


def test_answer_draft_multi_step_defaults() -> None:
    draft = AnswerDraft(answer="Ownership is …")
    assert draft.citation_chunk_ids == []
    assert draft.context_sufficient is True
    assert draft.follow_up_query is None


def test_answer_draft_insufficient_with_follow_up() -> None:
    draft = AnswerDraft(
        answer="Context is incomplete.",
        context_sufficient=False,
        follow_up_query="Rust borrowing and references",
    )
    assert draft.context_sufficient is False
    assert draft.follow_up_query == "Rust borrowing and references"


def test_agent_graph_state_multi_step_defaults() -> None:
    state = AgentGraphState(
        request=AgentRequest(workflow=WorkflowId.ANSWER, goal="What is ownership?"),
    )
    assert state.retrieve_query is None
    assert state.retrieve_rounds == 0
    assert state.needs_more_context is False
    assert state.retrieved == []
    assert state.faithfulness is None


def test_agent_graph_state_rejects_negative_retrieve_rounds() -> None:
    with pytest.raises(ValidationError):
        AgentGraphState(
            request=AgentRequest(goal="x"),
            retrieve_rounds=-1,
        )


def test_merge_retrieved_preserves_order_and_dedupes() -> None:
    first = [_hit("a", "A1", score=0.9), _hit("b", "B1", score=0.8)]
    second = [_hit("b", "B2", score=0.99), _hit("c", "C1", score=0.7)]

    merged = merge_retrieved(first, second)

    assert [h.chunk.id for h in merged] == ["a", "b", "c"]
    # First-seen hit wins for duplicates
    assert merged[1].chunk.text == "B1"
    assert merged[1].score == 0.8


def test_merge_retrieved_empty_sides() -> None:
    only_new = merge_retrieved([], [_hit("x")])
    assert [h.chunk.id for h in only_new] == ["x"]

    only_existing = merge_retrieved([_hit("y")], [])
    assert [h.chunk.id for h in only_existing] == ["y"]

    assert merge_retrieved([], []) == []
