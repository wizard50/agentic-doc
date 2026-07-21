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
  llm/               # LLM client + prompts (stub)
  tools/             # Tool protocol + retrieve wrapper (stub)
  graphs/            # LangGraph workflows + shared state (stub)
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
    WorkflowId,
    get_agent_settings,
    list_workflows,
    run_workflow,
)
```

## Dependencies

```
agent → rag → core
```
