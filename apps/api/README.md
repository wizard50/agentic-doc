# Agentic Doc API

M3 production FastAPI backend — HTTP boundary for Agentic Doc.

Phase 0 delivers a **runnable skeleton**: settings, health probes, CORS, request IDs, OpenAPI. Auth, Supabase, retrieval, and agent workflows land in later M3 phases.

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

Supabase / LLM secrets are **not** required for Phase 0.

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
  api/              # routers (health today; /v1 later)
```

## Intentionally not here yet

- Supabase Auth / JWT
- SQLAlchemy / Alembic
- `/v1/retrieve` or `/v1/workflows/*`
- Background jobs

See root [AGENTS.md](../../AGENTS.md) for the M3 roadmap.
