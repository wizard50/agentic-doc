# studio

M2 portfolio app — **Doc Agent** Streamlit UI for agentic documentation workflows.

Demonstrates the multi-step **Answer** workflow: plan → retrieve → generate (optional re-retrieve) → faithfulness evaluate, with timeline, citations, and evidence tabs.

## Run from the workspace root

```bash
cd path/to/agentic-doc
uv sync --dev

# index the default corpus (required for grounded answers)
uv run explorer ingest

# set LLM_API_KEY in .env (optional LLM_BASE_URL / LLM_MODEL for OpenRouter, etc.)
uv run studio
```

### Prerequisites

| Need | How |
|------|-----|
| Indexed corpus | `uv run explorer ingest` (or `INGEST_ON_STARTUP` on deploy) |
| LLM credentials | `LLM_API_KEY` in `.env` or Streamlit secrets |
| Phoenix (optional) | `PHOENIX_ENABLED=true` + `uv run phoenix serve` |

Configure defaults via workspace [`.env.example`](../../.env.example).

### What you can do

- Run **Answer** on a documentation goal (sidebar examples or free text)
- Inspect **faithfulness**, tool calls, duration, and citation count
- Open **Timeline** / **Citations** / **Evidence** tabs for the last run
- See roadmap cards for **Compare** and **Gap report** (not runnable yet)

## Deploy (Streamlit Community Cloud)

- **Main file:** `apps/studio/src/agentic_doc_studio/app.py`
- **Working directory:** repository root
- Demo markdown is under `corpora/rust-book/`

### Secrets

```toml
INGEST_ON_STARTUP = "true"
INGEST_SOURCE_DIR = "corpora/rust-book/src"
LLM_API_KEY = "sk-..."
# Optional OpenAI-compatible proxy:
# LLM_BASE_URL = "https://openrouter.ai/api/v1"
# LLM_MODEL = "openai/gpt-4o-mini"
PHOENIX_ENABLED = "false"
```

On first boot with an empty index, the app builds Chroma + BM25 from the shipped corpus (can take several minutes). Leave `INGEST_ON_STARTUP` off locally unless you want the same behavior.

## Related

- M1 search UI: [`apps/explorer`](../explorer/)
- Agent library: [`packages/agent`](../../packages/agent/)
