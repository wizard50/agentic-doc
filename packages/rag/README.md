# agentic-doc-rag

RAG retrieval layer for **agentic-doc** — ingest, multi-format parsers, chunking, embeddings, hybrid search, reranking, vector store, and evaluation.

## Layout

```
src/agentic_doc_rag/
  config.py        # RagSettings (vector store, search, embeddings, ingest, rerank)
  ingest/          # IngestSettings, run_ingestion()
  parsers/         # DocumentParser registry (Markdown, PDF)
  chunk/           # Markdown-oriented chunking helpers
  embeddings/      # Embeddings protocol (Chroma default, sentence-transformers)
  sparse/          # BM25 keyword index
  vectorstore/     # VectorStore protocol + Chroma
  retrieval/       # PipelineRetriever (retrieve → filter → rerank → top-k)
  evaluation/      # Golden-set eval, metrics, Phoenix LLM relevance
  observability/   # OpenTelemetry / Phoenix tracing helpers
```

## Capabilities (M1)

- **Ingest:** recursive discovery of `.md` and `.pdf` under a source directory
- **PDF:** text-layer extraction via PyMuPDF (no OCR); empty pages skipped
- **Retrieval:** semantic, BM25 keyword, hybrid (RRF); metadata filters; optional cross-encoder rerank
- **Storage:** Chroma (local) + BM25 sparse index
- **Eval:** hit@k / MRR / recall + optional LLM document relevance

App entry points live in `apps/explorer` (`uv run explorer ingest|ui|eval`). See the workspace root [README](../../README.md) and [`.env.example`](../../.env.example).
