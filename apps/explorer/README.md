# explorer

M1 portfolio app — **Streamlit search UI** and **ingestion / eval CLI** for the Agentic RAG system.

## Run from the workspace root

```bash
cd path/to/agentic-doc
uv sync --dev

# index the default corpus (corpora/rust-book/src, or INGEST_SOURCE_DIR)
uv run explorer ingest

# index any documentation tree (Markdown + PDF text layer)
uv run explorer ingest --source path/to/docs
uv run explorer ingest --source path/to/docs --skip NOTES.md
uv run explorer ingest --source data/download/rust-book/src   # local full clone

# launch the search UI (default)
uv run explorer
# or explicitly:
uv run explorer ui

# run retrieval evaluation (requires ingest first)
uv run explorer eval
uv run explorer eval --search-mode hybrid --rerank
uv run explorer eval --llm   # optional LLM relevance scoring (LLM_API_KEY)
```

Ingest picks up `.md` and `.pdf` files recursively. PDFs use pymupdf4llm (layout-aware markdown, OCR off). Configure defaults via `.env` (`INGEST_SOURCE_DIR`, `INGEST_SKIP_FILES`, search/embedding/rerank settings) — see the workspace [`.env.example`](../../.env.example).

### Demo corpus

- Shipped snapshot: `corpora/rust-book/` (default `INGEST_SOURCE_DIR`)
- Local full clone (optional): `python scripts/download.py` → `data/download/rust-book`
- Refresh shipped snapshot from the local clone: `python scripts/sync_demo_corpus.py`

## Deploy (Streamlit Community Cloud)

- **Main file:** `apps/explorer/src/agentic_doc_explorer/app.py`
- **Working directory:** repository root
- Demo markdown is under `corpora/rust-book/`; no shell access is required for ingest

### First-start ingest

In **App settings → Secrets**:

```toml
INGEST_ON_STARTUP = "true"
INGEST_SOURCE_DIR = "corpora/rust-book/src"
PHOENIX_ENABLED = "false"
```

On first boot with an empty `data/chroma`, the app builds Chroma + BM25 from `corpora/rust-book` (can take several minutes). Leave `INGEST_ON_STARTUP` off locally unless you want the same behavior.
