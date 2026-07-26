# agentic-doc-agent

Agentic intelligence layer for **agentic-doc** — LangGraph workflows, retrieval tools, grounded generation, structured outputs, and generation-quality evaluation.

Built on top of [`agentic-doc-rag`](../rag/) (M1). Apps and the future production API should prefer the public package API:

```python
from agentic_doc_agent import AgentRequest, run_workflow
```

## Status

**Answer workflow is runnable** end-to-end via `run_workflow`:

```
plan → retrieve → generate ⇄ retrieve → evaluate → AgentResult
```

| Setting | Default | Role |
|---------|---------|------|
| `PLAN_ENABLED` | `true` | Rewrite goal into a retrieval query before the first search |
| `MAX_TOOL_ROUNDS` | `5` | Cap on retrieve tool invocations (blocks infinite re-retrieve) |
| `FAITHFULNESS_ENABLED` | `true` | LLM-as-judge groundedness score on `metrics.faithfulness` (0–1) |

Plan and faithfulness failures are **fail-soft** (answer can still succeed). Generate may request another retrieve when context is insufficient and a `follow_up_query` is provided.

**Phoenix tracing:** Answer runs emit OpenInference spans (`agent.run_workflow`, `agent.plan`, `agent.tool.retrieve`, `agent.generate`, `agent.evaluate`) when tracing is registered. Reuse M1 settings (`PHOENIX_ENABLED`, etc.):

```bash
# Terminal 1
uv run phoenix serve

# Terminal 2
PHOENIX_ENABLED=true uv run python scripts/smoke_answer.py
```

Still stubbed / not implemented:

- Compare and gap-report workflows
- Offline generation-eval CLI / golden set

**Demo UI:** [`apps/studio`](../../apps/studio/) — **Doc Agent** Streamlit app (`uv run studio`) for live Answer runs with timeline, citations, evidence, and faithfulness. **Live:** [https://agentic-doc.streamlit.app/](https://agentic-doc.streamlit.app/). Uses `LLM_API_KEY` / `LLM_BASE_URL` / `LLM_MODEL` (not `EVAL_LLM_MODEL`).

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
  graphs/            # Answer workflow: plan, retrieve, generate loop, evaluate
  evaluation/        # Faithfulness LLM judge (runtime score on AgentResult)
  observability/     # OpenInference / OTEL span helpers for agent runs
```

## Multi-step Answer path

1. **plan** (optional) — rewrite the user goal into a search query  
2. **retrieve** — M1 RAG tool; merges chunks across rounds by chunk id  
3. **generate** — structured answer; may set `needs_more_context` + `follow_up_query`  
4. **re-retrieve** — if needed and under `MAX_TOOL_ROUNDS`  
5. **evaluate** (optional) — faithfulness score  

Disable planning with `PLAN_ENABLED=false` for lower latency (retrieve uses the raw goal).

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
