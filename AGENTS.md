# AGENTS.md

## Project Overview
- **Name**: agentic-doc
- **Goal**: Build a production-grade Agentic RAG system for technical documentation with strong observability, evaluation, and clean architecture.
- **Domain**: Technical documentation and developer knowledge bases (Markdown, PDFs, and similar). Initial corpus: The Rust Programming Language book.
- **Approach**: Incremental delivery across three milestones (see below).

## Current Focus
**Milestone 3 – Production Backend**

Build a production API and backend on top of the completed M1 RAG core and M2 Answer agent:

- FastAPI orchestration of RAG + agent workflows
- Auth, persistence, background jobs (planned)
- Guardrails, cost tracking, production observability
- Docker + deployment

**Foundation (M1, closed):** see Milestone 1 below. Prefer `from agentic_doc_rag import ...` for library consumers.

**Agent layer (M2, closed):** Prefer `from agentic_doc_agent import run_workflow, AgentRequest` (`packages/agent`). The **Answer** workflow is multi-step (LangGraph plan → retrieve → generate ⇄ re-retrieve → optional faithfulness evaluate) with Phoenix/OpenInference spans when tracing is registered. Live smoke: `uv run explorer ingest` then `uv run python scripts/smoke_answer.py` (requires `LLM_API_KEY`; `PHOENIX_ENABLED=true` for traces). **Doc Agent UI:** `uv run studio` (`apps/studio`) — Answer demo with timeline, citations, evidence, and faithfulness metrics.

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

### Milestone 2 – Agentic Intelligence Layer ← Closed

Delivered agentic layer on top of Milestone 1:

- Multi-step reasoning, planning, and tool use (LangGraph)
- **Answer workflow:** `run_workflow(AgentRequest(...))` — plan (query rewrite), retrieve tool, structured generation, optional re-retrieve (`MAX_TOOL_ROUNDS`), optional faithfulness score
- Answer generation grounded in retrieved context
- Faithfulness scoring on Answer runs (`metrics.faithfulness`; `FAITHFULNESS_ENABLED`)
- Multi-step Answer path (`PLAN_ENABLED`, re-retrieve when context is insufficient)
- Agent Phoenix/OpenInference spans on Answer runs (`agent.run_workflow` + plan/tool/generate/evaluate)
- Structured outputs with Pydantic models
- **Doc Agent UI** (`apps/studio`, `uv run studio`): Streamlit demo for Answer with metrics, step timeline, citations, and evidence
- Focus on Software Engineering / technical documentation domain

### Milestone 3 – Production Backend ← Current
- **Phase 0 (skeleton):** `apps/api` — FastAPI app factory, settings, `/health` + `/ready`, CORS, request IDs, OpenAPI, Dockerfile (`uv run api`)
- FastAPI backend orchestrating M1 + M2 (planned)
- Supabase (Auth + Postgres) + Alembic for persistence (planned)
- Background job processing
- Guardrails, cost tracking, and production observability
- Docker + deployment setup
- Clean API for both simple RAG and complex agentic workflows

## Tech Stack & Key Decisions
- **Orchestration**: LangGraph (primary)
- **Observability**: Arize Phoenix (preferred)
- **Vector Store**: Chroma (local development) — abstraction layer for future backends (pgvector, LanceDB, Qdrant, etc.)
- **LLM**: OpenAI-compatible APIs (via LiteLLM when needed)
- **Frontend**: Streamlit (rapid iteration; M1 explorer, M2 Doc Agent studio)
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
