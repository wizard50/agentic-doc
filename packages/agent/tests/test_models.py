import pytest
from pydantic import ValidationError

from agentic_doc_agent.models import (
    AgentRequest,
    AgentResult,
    AgentStatus,
    WorkflowId,
)


def test_workflow_ids() -> None:
    assert WorkflowId.ANSWER.value == "answer"
    assert WorkflowId.COMPARE.value == "compare"
    assert WorkflowId.GAP_REPORT.value == "gap_report"


def test_agent_request_requires_goal() -> None:
    with pytest.raises(ValidationError):
        AgentRequest(goal="")


def test_agent_request_defaults() -> None:
    request = AgentRequest(goal="What is ownership in Rust?")
    assert request.workflow is WorkflowId.ANSWER
    assert request.top_k is None
    assert request.filters is None
    assert request.metadata == {}


def test_agent_result_round_trip() -> None:
    result = AgentResult(
        status=AgentStatus.SUCCEEDED,
        workflow=WorkflowId.ANSWER,
        goal="What is ownership?",
        answer="Ownership is …",
    )
    restored = AgentResult.model_validate(result.model_dump())
    assert restored.status is AgentStatus.SUCCEEDED
    assert restored.answer == "Ownership is …"
    assert restored.citations == []
    assert restored.metrics.tool_calls == 0
