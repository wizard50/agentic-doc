# agentic-doc-rag

RAG retrieval layer for **agentic-doc** — ingest, document parsers, chunking, embeddings, hybrid search, reranking, vector store, and evaluation.

## Layout

```
src/agentic_doc_rag/
  config.py        # RagSettings (vector store, search, embeddings, ingest, rerank)
  ingest/          # IngestSettings, run_ingestion()
  parsers/         # DocumentParser registry (Markdown, PDF)
  chunk/           # Shared chunking helpers (markdown + make_chunk_id)
  embeddings/      # Embeddings protocol (Chroma default, sentence-transformers)
  sparse/          # BM25 keyword index
  vectorstore/     # VectorStore protocol + Chroma
  retrieval/       # PipelineRetriever (retrieve → filter → rerank → top-k)
  evaluation/      # Golden-set eval, metrics, Phoenix LLM relevance
  observability/   # OpenTelemetry / Phoenix tracing helpers
```

## Capabilities (M1)

- **Ingest:** recursive discovery of `.md` and `.pdf` under a source directory
- **PDF:** layout-aware extraction via pymupdf4llm → markdown, then header-aware chunking (OCR off by default); empty pages skipped
- **Retrieval:** semantic, BM25 keyword, hybrid (RRF); path/section metadata filters; optional cross-encoder rerank
- **Storage:** Chroma (local) + BM25 sparse index
- **Eval:** hit@k / MRR / recall + optional LLM document relevance

## Public API

Stable entry points are re-exported from `agentic_doc_rag`:

```python
from agentic_doc_rag import (
    create_retriever,
    create_sparse_index,
    create_vector_store,
    get_rag_settings,
    register_tracing,
    resolve_ingest_settings,
    run_ingestion,
    RetrievalRequest,
    SearchMode,
)

settings = get_rag_settings()
register_tracing(...)  # optional Phoenix

run_ingestion(
    create_vector_store(settings),
    create_sparse_index(settings),
    resolve_ingest_settings(settings),
)

retriever = create_retriever(settings)
hits = retriever.retrieve(
    RetrievalRequest(query="What is ownership?", mode=SearchMode.HYBRID, top_k=5)
)
```

Submodules remain available for advanced use (custom stages, parsers, eval helpers).

App entry points live in `apps/explorer` (`uv run explorer ingest|ui|eval`). See the workspace root [README](../../README.md) and [`.env.example`](../../.env.example).
