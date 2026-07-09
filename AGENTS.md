# AGENTS.md

## Project Overview
- **Name**: agentic-doc
- **Goal**: Build a production-grade Agentic RAG system for technical documentation with strong observability, evaluation, and clean architecture.
- **Domain**: Technical documentation and developer knowledge bases (Markdown, code, PDFs). Initial corpus: The Rust Programming Language book.
- **Approach**: Incremental delivery across three milestones (see below).

## Current Focus
**Milestone 1 – RAG Core (Foundation)**

We are currently building the foundational RAG system:
- Robust document ingestion pipeline (Markdown, code files, PDFs, etc.)
- Advanced RAG techniques (smart chunking, embeddings, hybrid search, re-ranking)
- Vector database layer with clean abstraction (Chroma for local dev, pgvector/LanceDB planned)
- Observability and evaluation from day one (Phoenix tracing + retrieval/faithfulness metrics)
- Exposes a clean, usable API

## Milestones

### Milestone 1 – RAG Core (Foundation) ← Current
- Robust ingestion pipeline for technical documentation
- Advanced RAG (chunking, embeddings, hybrid search, re-ranking)
- Vector store abstraction (Chroma local → production backends)
- Full observability (Phoenix) and automated evaluation
- Clean standalone API

### Milestone 2 – Agentic Intelligence Layer
- Agentic capabilities on top of Milestone 1
- Multi-step reasoning, planning, and tool use
- Structured outputs with Pydantic models
- Developer-focused workflows (analysis, comparison, gap detection, report generation)
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
- **LLM**: OpenAI (via LiteLLM when needed)
- **Frontend**: Streamlit (rapid iteration)
- **Data Models**: Pydantic v2 (mandatory for all structured data)
- **Python version**: 3.13+

## Architecture Principles
- Clean separation of concerns (ingestion / retrieval / agent / observability / evaluation)
- Prefer composition over inheritance
- All LLM inputs/outputs should use Pydantic models when possible
- Tracing should be added early and be visible in Phoenix
- Keep the agent logic testable and observable
- Build for swapability (especially the VectorStore layer)

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
