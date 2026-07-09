# Agentic Doc

Production-grade Agentic RAG for technical documentation — with observability, evaluation, and clean architecture.

The workspace is **domain-agnostic**: packages and apps are built to work with any technical documentation corpus (Markdown, code, PDFs, etc.). For local development, [The Rust Programming Language](https://github.com/rust-lang/book) will be the first reference corpus — not yet wired up.

> **Status:** WIP scaffold. Workspace layout, tooling, and settings are in place. Ingestion, retrieval, vector store, UI, and production backend are planned next.

Observability (Phoenix) and evaluation will be added as the RAG pipeline matures. See [AGENTS.md](AGENTS.md) for milestones and architecture principles.

## Project structure

```
packages/
  core/     # Shared config and foundational types
  rag/      # RAG retrieval layer (chunking, embeddings, vector store)
apps/
  explorer/ # M1 RAG explorer — Streamlit UI + ingest CLI
```

## Setup

```bash
uv sync --dev
pre-commit install
```

## Development

```bash
uv run pytest
uv run ruff check .
uv run ty check
```

## M1 explorer

![Doc Explorer search UI](assets/doc-explorer-screenhot.png)

```bash
uv run explorer ingest   # index corpus into data/chroma
uv run explorer          # launch Streamlit search UI
```

## Observability (Phoenix)

Phoenix runs as a local dev server (included in dev dependencies). Tracing is off by default — set `PHOENIX_ENABLED=true` in `.env` once instrumentation is wired up.

```bash
# Terminal 1 — UI at http://localhost:6006, collector at http://localhost:4317
uv run phoenix serve

# Terminal 2 — index and search (with tracing enabled)
PHOENIX_ENABLED=true uv run explorer ingest
PHOENIX_ENABLED=true uv run explorer
```

## License

MIT — see [LICENSE](LICENSE).
