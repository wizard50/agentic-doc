# explorer

M1 portfolio app — **Streamlit search UI** and **ingestion CLI** for the Agentic RAG system.

## Run from the workspace root

```bash
cd path/to/agentic-doc
uv sync --dev

# index the Rust book corpus (once, or after updates)
uv run explorer ingest

# launch the search UI (default)
uv run explorer
# or explicitly:
uv run explorer ui

# run retrieval evaluation (requires ingest first)
uv run explorer eval
uv run explorer eval --llm   # optional LLM relevance scoring (LLM_API_KEY)
```

## Deploy (Streamlit Community Cloud)

- **Main file:** `apps/explorer/src/agentic_doc_explorer/app.py`
- **Working directory:** repository root
- Pre-index with `uv run explorer ingest` locally, or run ingest in your deploy pipeline and ship `data/chroma/`.