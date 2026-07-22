# agentic-doc-agent

Agentic intelligence layer for **agentic-doc** — LangGraph workflows, retrieval tools, grounded generation, structured outputs, and generation-quality evaluation.

Built on top of [`agentic-doc-rag`](../rag/) (M1). Apps and the future production API should prefer the public package API.

## Status

M2 scaffold — public contracts and module layout are in place; workflow graphs and runtime are next.

## Layout

```
src/agentic_doc_agent/
  config.py          # AgentSettings
  models.py          # AgentRequest, AgentResult, WorkflowId, citations, steps
  runtime.py         # run_workflow(), list_workflows()
  llm/               # OpenAI-compatible LlmClient (complete + complete_structured)
  tools/             # Tool protocol + RetrieveTool (M1 retriever wrapper)
  graphs/            # Answer workflow (prompts, nodes, compiled LangGraph)
  evaluation/        # Faithfulness and generation eval (stub)
  observability/     # Tracing helpers (stub)
```

## Public API

Stable entry points are re-exported from `agentic_doc_agent`:

```python
from agentic_doc_agent import (
    AgentRequest,
    AgentResult,
    AgentSettings,
    ChatMessage,
    ChatRole,
    RetrieveTool,
    WorkflowId,
    create_llm_client,
    get_agent_settings,
    list_workflows,
    run_workflow,
)

# End-to-end answer workflow (inject fakes in tests; defaults use RAG + LLM env)
# result = run_workflow(AgentRequest(goal="What is ownership?"))
# # or: run_workflow(request, retrieve_tool=..., llm=...)

# Retrieve tool (inject an M1 Retriever)
# tool = RetrieveTool(create_retriever(get_rag_settings()))
# result = tool.invoke(query="What is ownership?", top_k=5)

# LLM client (requires LLM_API_KEY; optional LLM_BASE_URL for OpenRouter-compatible APIs)
# llm = create_llm_client()
# out = llm.complete([ChatMessage(role=ChatRole.USER, content="Summarize ownership")])
# typed = llm.complete_structured(messages, MyPydanticModel)
```

## Dependencies

```
agent → rag → core
```
