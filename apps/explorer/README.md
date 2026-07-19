# explorer

M1 portfolio app — **Streamlit search UI** and **ingestion / eval CLI** for the Agentic RAG system.

## Run from the workspace root

```bash
cd path/to/agentic-doc
uv sync --dev

# index the default corpus (Rust book Markdown under INGEST_SOURCE_DIR)
uv run explorer ingest

# index any documentation tree (Markdown + PDF text layer)
uv run explorer ingest --source path/to/docs
uv run explorer ingest --source path/to/docs --skip NOTES.md

# launch the search UI (default)
uv run explorer
# or explicitly:
uv run explorer ui

# run retrieval evaluation (requires ingest first)
uv run explorer eval
uv run explorer eval --search-mode hybrid --rerank
uv run explorer eval --llm   # optional LLM relevance scoring (LLM_API_KEY)
```

Ingest picks up `.md` and `.pdf` files recursively. PDFs use the text layer only (no OCR). Configure defaults via `.env` (`INGEST_SOURCE_DIR`, `INGEST_SKIP_FILES`, search/embedding/rerank settings) — see the workspace [`.env.example`](../../.env.example).

## Deploy (Streamlit Community Cloud)

- **Main file:** `apps/explorer/src/agentic_doc_explorer/app.py`
- **Working directory:** repository root
- Pre-index with `uv run explorer ingest` locally, or run ingest in your deploy pipeline and ship `data/chroma/`.
