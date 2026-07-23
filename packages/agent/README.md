# agentic-doc-agent

Agentic intelligence layer for **agentic-doc** — LangGraph workflows, retrieval tools, grounded generation, structured outputs, and generation-quality evaluation.

Built on top of [`agentic-doc-rag`](../rag/) (M1). Apps and the future production API should prefer the public package API:

```python
from agentic_doc_agent import AgentRequest, run_workflow
```

## Status

**Answer workflow is runnable** end-to-end via `run_workflow` (retrieve → generate → optional faithfulness evaluate → `AgentResult`).

Faithfulness scoring (LLM-as-judge) is on by default (`FAITHFULNESS_ENABLED=true`) and populates `metrics.faithfulness` (0–1). Disable with `FAITHFULNESS_ENABLED=false`. Judge failures are fail-soft: the answer still succeeds with `faithfulness=None`.

**Phoenix tracing:** Answer runs emit OpenInference spans (`agent.run_workflow`, `agent.tool.retrieve`, `agent.generate`, `agent.evaluate`) when tracing is registered. Reuse M1 settings (`PHOENIX_ENABLED`, etc.):

```bash
# Terminal 1
uv run phoenix serve

# Terminal 2
PHOENIX_ENABLED=true uv run python scripts/smoke_answer.py
```

Still stubbed / not implemented:

- Compare and gap-report workflows
- Offline generation-eval CLI / golden set
- Demo UI (planned as a separate app)

## Prerequisites (live runs)

Run commands from the **workspace root**.

1. **Index the corpus** (empty index → 0 retrieved passages; the model may still “succeed” with an insufficient-context answer):

   ```bash
   uv run explorer ingest
   ```

2. **LLM credentials** in `.env` (see workspace [`.env.example`](../../.env.example)):

   ```env
   LLM_API_KEY=sk-...
   # Optional OpenAI-compatible proxy (e.g. OpenRouter):
   # LLM_BASE_URL=https://openrouter.ai/api/v1
   # LLM_MODEL=openai/gpt-4o-mini
   ```

## Live smoke

```bash
# Default goal: ownership in Rust — prints full answer; exit 0 on success
uv run python scripts/smoke_answer.py

uv run python scripts/smoke_answer.py --goal "What is borrowing?"
```

The script fails fast if the index is empty or the workflow does not succeed with retrieved context. It prints the **full** answer (no truncated preview).

## Layout

```
src/agentic_doc_agent/
  config.py          # AgentSettings
  models.py          # AgentRequest, AgentResult, WorkflowId, citations, steps
  runtime.py         # run_workflow(), list_workflows()
  llm/               # OpenAI-compatible LlmClient (complete + complete_structured)
  tools/             # Tool protocol + RetrieveTool (M1 retriever wrapper)
  graphs/            # Answer workflow (prompts, nodes, compiled LangGraph)
  evaluation/        # Faithfulness LLM judge (runtime score on AgentResult)
  observability/     # OpenInference / OTEL span helpers for agent runs
```

## Observability

Register Phoenix once at process start (same helper as M1):

```python
from agentic_doc_core.config import get_phoenix_settings
from agentic_doc_rag.observability import register_tracing

register_tracing(get_phoenix_settings())
```

Spans are always opened; they are no-ops until a tracer provider is registered. `scripts/smoke_answer.py` calls `register_tracing` automatically.


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

# Production defaults: RAG index from settings + LLM from LLM_API_KEY
result = run_workflow(AgentRequest(goal="What is ownership?"))

# Tests: inject fakes
# result = run_workflow(request, retrieve_tool=..., llm=...)
```

## Dependencies

```
agent → rag → core
```
