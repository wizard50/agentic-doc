# Agentic Doc API

M3 production FastAPI backend — HTTP boundary for Agentic Doc.

Phase 0 is the **runnable skeleton**: settings, health probes, CORS, request IDs, OpenAPI.

The M3 finish is `POST /v1/answer`: call `run_workflow` and return the answer, citations, faithfulness, latency, token cost, trace id, and abstention. The open demo has no signup. An in-process rate limit and a max question length protect the LLM budget. Phoenix spans are the run log. Doc Agent Studio stays the UI.

## Run locally

From the **workspace root**:

```bash
uv sync --dev
uv run api
```

Equivalent:

```bash
uv run uvicorn agentic_doc_api.main:app --reload --port 8000
```

Open:

- Health: <http://localhost:8000/health>
- Ready: <http://localhost:8000/ready>
- OpenAPI UI: <http://localhost:8000/docs>

## Configuration

Settings use pydantic-settings (see `agentic_doc_api.config.ApiSettings`). Useful env vars (also in root [`.env.example`](../../.env.example)):

| Variable | Default | Role |
|----------|---------|------|
| `API_HOST` | `0.0.0.0` | Bind host |
| `API_PORT` | `8000` | Bind port |
| `ALLOWED_ORIGINS` | `http://localhost:5173` | CORS origins (comma-separated) |
| `API_DOCS_ENABLED` | `true` | Expose `/docs` and `/redoc` |
| `LOG_LEVEL` | `INFO` | Logging level |
| `ENVIRONMENT` | `dev` | `dev` enables reload by default in `uv run api` |

Phase 0 does not need `LLM_API_KEY`. `/v1/answer` will use the same `LLM_API_KEY`, `LLM_BASE_URL`, and `LLM_MODEL` as the Answer workflow. Set `PHOENIX_ENABLED=true` to export spans.

## Tests

```bash
uv run pytest apps/api -q
```

## Docker

From the workspace root:

```bash
docker build -f apps/api/Dockerfile -t agentic-doc-api .
docker run --rm -p 8000:8000 agentic-doc-api
```

## Layout

```text
src/agentic_doc_api/
  main.py           # create_app(), ASGI app
  config.py         # ApiSettings
  cli.py            # uv run api
  logging.py
  deps.py
  middleware/       # X-Request-ID
  api/              # routers (health today; /v1/answer is the finish)
```

See root [AGENTS.md](../../AGENTS.md) for the M3 finish.
