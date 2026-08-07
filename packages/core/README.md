# agentic-doc-core

Shared configuration for the **agentic-doc** workspace.

## What it provides

- **`CoreSettings`** — environment, log level, shared LLM credential env (`LLM_API_KEY`, `LLM_BASE_URL`)
- **`PhoenixSettings`** — Phoenix/OpenTelemetry tracing toggles (`PHOENIX_*`)

Used by `agentic-doc-rag`, `agentic-doc-agent`, and the apps.

## Layout

```
src/agentic_doc_core/
  config.py    # CoreSettings, PhoenixSettings, getters
```
