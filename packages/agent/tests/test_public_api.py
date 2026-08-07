import agentic_doc_agent as agent
from agentic_doc_agent import WorkflowId


def test_public_api_exports_core_symbols() -> None:
    assert agent.WorkflowId.ANSWER.value == "answer"
    assert agent.AgentRequest is not None
    assert agent.AgentResult is not None
    assert agent.AgentSettings is not None
    assert agent.AgentStatus is not None
    assert agent.Citation is not None
    assert agent.StepEvent is not None
    assert agent.StepKind is not None
    assert agent.AgentMetrics is not None
    assert agent.RetrieveTool is not None
    assert agent.RetrieveArgs is not None
    assert agent.RetrieveResult is not None
    assert agent.LlmClient is not None
    assert agent.ChatMessage is not None
    assert agent.ChatResult is not None
    assert agent.ChatRole is not None
    assert agent.TokenUsage is not None
    assert agent.LlmError is not None
    assert agent.LlmConfigError is not None
    assert agent.LlmRequestError is not None
    assert agent.LlmResponseError is not None
    assert callable(agent.create_llm_client)
    assert callable(agent.get_agent_settings)
    assert callable(agent.list_workflows)
    assert callable(agent.run_workflow)


def test_public_api_all_names_are_importable() -> None:
    for name in agent.__all__:
        assert hasattr(agent, name), name
        assert getattr(agent, name) is not None


def test_list_workflows_returns_all_ids() -> None:
    workflows = agent.list_workflows()
    assert workflows == [WorkflowId.ANSWER]
