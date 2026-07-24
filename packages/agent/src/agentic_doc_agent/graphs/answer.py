"""Compiled LangGraph for the answer workflow."""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_doc_agent.graphs.answer_nodes import (
    run_answer_evaluate,
    run_answer_generate,
    run_answer_plan,
    run_answer_retrieve,
)
from agentic_doc_agent.graphs.answer_prompts import DEFAULT_MAX_CHUNK_CHARS
from agentic_doc_agent.graphs.state import AgentGraphState
from agentic_doc_agent.llm.protocols import LlmClient
from agentic_doc_agent.tools.retrieve import RetrieveTool

AfterGenerate = Literal["retrieve", "evaluate"]


def route_after_generate(
    state: AgentGraphState,
    *,
    max_tool_rounds: int,
) -> AfterGenerate:
    """Choose re-retrieve vs evaluate after generate."""
    if state.error is not None:
        return "evaluate"
    follow_up = (state.retrieve_query or "").strip()
    if state.needs_more_context and state.retrieve_rounds < max_tool_rounds and follow_up:
        return "retrieve"
    return "evaluate"


def build_answer_graph(
    retrieve_tool: RetrieveTool,
    llm: LlmClient,
    *,
    max_chunk_chars: int = DEFAULT_MAX_CHUNK_CHARS,
    faithfulness_enabled: bool = True,
    plan_enabled: bool = True,
    max_tool_rounds: int = 5,
) -> CompiledStateGraph:
    """Compile plan → retrieve → generate ⇄ retrieve → evaluate for ANSWER.

    When ``plan_enabled`` is false, plan is a no-op node (no LLM call, no step).
    Re-retrieve runs only when generate sets ``needs_more_context`` with a
    follow-up query and ``retrieve_rounds < max_tool_rounds``.
    """
    if max_tool_rounds < 1:
        raise ValueError("max_tool_rounds must be >= 1")

    def plan_node(state: AgentGraphState) -> AgentGraphState:
        return run_answer_plan(state, llm, enabled=plan_enabled)

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

    def after_generate(state: AgentGraphState) -> AfterGenerate:
        return route_after_generate(state, max_tool_rounds=max_tool_rounds)

    graph = StateGraph(AgentGraphState)
    graph.add_node("plan", plan_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("evaluate", evaluate_node)
    graph.add_edge(START, "plan")
    graph.add_edge("plan", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_conditional_edges(
        "generate",
        after_generate,
        {
            "retrieve": "retrieve",
            "evaluate": "evaluate",
        },
    )
    graph.add_edge("evaluate", END)
    return graph.compile()
