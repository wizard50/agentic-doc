"""Compiled LangGraph for the answer workflow."""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_doc_agent.graphs.answer_nodes import (
    run_answer_evaluate,
    run_answer_generate,
    run_answer_retrieve,
)
from agentic_doc_agent.graphs.answer_prompts import DEFAULT_MAX_CHUNK_CHARS
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.llm.protocols import LlmClient
from agentic_doc_agent.tools.retrieve import RetrieveTool


def build_answer_graph(
    retrieve_tool: RetrieveTool,
    llm: LlmClient,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    faithfulness_enabled: bool = True,
) -> CompiledStateGraph:
    """Compile retrieve → generate → evaluate for WorkflowId.ANSWER."""

    def retrieve_node(state: AgentGraphState) -> AgentGraphState:
        return run_answer_retrieve(state, retrieve_tool)

    def generate_node(state: AgentGraphState) -> AgentGraphState:
        return run_answer_generate(state, llm, max_chunk_chars=max_chunk_chars)

    def evaluate_node(state: AgentGraphState) -> AgentGraphState:
        return run_answer_evaluate(
            state,
            llm,
            enabled=faithfulness_enabled,
            max_chunk_chars=max_chunk_chars,
        )

    graph = StateGraph(AgentGraphState)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "evaluate")
    graph.add_edge("evaluate", END)
    return graph.compile()
