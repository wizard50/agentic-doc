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
  ingest/   # Document ingestion worker
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

## License

MIT — see [LICENSE](LICENSE).
