# AGENTS.md

## Project Overview
- **Name**: agentic-doc
- **Goal**: Build a production-grade Agentic RAG system for technical documentation with strong observability, evaluation, and clean architecture.
- **Domain**: Technical documentation and developer knowledge bases (Markdown, PDFs, and similar). Initial corpus: The Rust Programming Language book.
- **Approach**: Incremental delivery across three milestones (see below).

## Current Focus
**Milestone 2 – Agentic Intelligence Layer**

Build agentic capabilities on top of the completed M1 documentation RAG core:

- Multi-step reasoning, planning, and tool use (LangGraph)
- Answer generation grounded in retrieved context
- Faithfulness and related answer-quality metrics
- Structured outputs with Pydantic models
- Developer-focused workflows (analysis, comparison, gap detection, report generation)
- Keep agent logic testable, observable, and visible in Phoenix

**Foundation (M1, closed):** see Milestone 1 below. Prefer `from agentic_doc_rag import ...` for library consumers.

**M2 library:** Prefer `from agentic_doc_agent import run_workflow, AgentRequest` for agent workflows (`packages/agent`). The **Answer** workflow is implemented (LangGraph retrieve → generate). Live smoke: `uv run explorer ingest` then `uv run python scripts/smoke_answer.py` (requires `LLM_API_KEY`).

## Milestones

### Milestone 1 – RAG Core (Foundation) ← Closed

Delivered documentation RAG:

- **Ingest:** Markdown and PDF (pymupdf4llm → markdown, then header-aware chunking); configurable source directory and skip list via settings/CLI
- **Embeddings:** swappable backends (Chroma default, sentence-transformers) via settings
- **Retrieval modes:** semantic (vector), **keyword (BM25)**, and **hybrid (RRF fusion)** of both
- **Pipeline stages:** retrieve → path/section metadata filters → optional cross-encoder re-ranking → top-k
- **Storage:** Chroma vector store (local) + BM25 sparse index, with abstractions for future backends
- **Observability:** Phoenix / OpenTelemetry tracing on ingest and retrieve
- **Evaluation:** golden-set retrieval metrics (hit@k, MRR, recall@k, per-tag) and optional LLM document relevance
- **Apps / API:** Streamlit explorer + `ingest` / `eval` / `ui` CLI; public `agentic_doc_rag` exports

### Milestone 2 – Agentic Intelligence Layer ← Current
- Agentic capabilities on top of Milestone 1
- Multi-step reasoning, planning, and tool use (LangGraph)
- **Answer workflow:** `run_workflow(AgentRequest(...))` — retrieve tool + structured generation
- Answer generation grounded in retrieved context
- Faithfulness and related answer-quality metrics (planned)
- Structured outputs with Pydantic models
- Developer-focused workflows (analysis, comparison, gap detection, report generation) — Answer done; others planned
- Focus on Software Engineering / technical documentation domain

### Milestone 3 – Production Backend
- FastAPI backend orchestrating M1 + M2
- Background job processing
- PostgreSQL + Alembic for persistence
- Guardrails, cost tracking, and production observability
- Docker + deployment setup
- Clean API for both simple RAG and complex agentic workflows

## Tech Stack & Key Decisions
- **Orchestration**: LangGraph (primary)
- **Observability**: Arize Phoenix (preferred)
- **Vector Store**: Chroma (local development) — abstraction layer for future backends (pgvector, LanceDB, Qdrant, etc.)
- **LLM**: OpenAI-compatible APIs (via LiteLLM when needed)
- **Frontend**: Streamlit (rapid iteration; M1 explorer)
- **Data Models**: Pydantic v2 (mandatory for all structured data)
- **Python version**: 3.13+

## Architecture Principles
- Clean separation of concerns (ingestion / retrieval / agent / observability / evaluation)
- Prefer composition over inheritance
- All LLM inputs/outputs should use Pydantic models when possible
- Tracing should be added early and be visible in Phoenix
- Keep agent logic testable and observable
- Build for swapability (especially the VectorStore and Embeddings layers)
- Prefer the public `agentic_doc_rag` API for app and library consumers; use submodules only for advanced customization

## Coding Standards
- Use type hints everywhere
- Prefer `pydantic` models over raw dicts for structured data
- Write clear, self-documenting code
- Add docstrings for public functions and classes
- Keep functions relatively small and focused
- Use `ruff` for linting and formatting

## How to Work With Me
- Always read `AGENTS.md` before starting a task
- Ask clarifying questions if requirements are ambiguous
- Propose small, incremental changes when possible
- Show the diff or key changes after implementing something
- Prioritize clean architecture, observability, and evaluation over quick hacks
- When writing, reviewing, or refactoring code, follow [`.agents/skills/karpathy-guidelines/SKILL.md`](.agents/skills/karpathy-guidelines/SKILL.md) (think before coding, simplicity first, surgical changes, goal-driven execution)

## Agent Skills
Project skills live under [`.agents/skills/`](.agents/skills/) (portable Agent Skills format). Start with:
- **karpathy-guidelines** — behavioral rules to avoid overcomplication, drive-by refactors, and unverified work
