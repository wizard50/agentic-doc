from pathlib import Path

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span
from agentic_doc_rag.sparse.protocols import SparseIndex
from agentic_doc_rag.vectorstore.base import VectorStore


def run_ingestion(
    source_dir: Path,
    vectorstore: VectorStore,
    sparse_index: SparseIndex,
    skip_files: frozenset[str] | None = None,
) -> None:
    with get_tracer(__name__).start_as_current_span("ingest.run") as span:
        mark_chain_span(span)
        span.set_attribute("source_dir", str(source_dir))
        print(f"Running ingestion from {source_dir}...")
        chunks: list[DocumentChunk] = chunk_markdown_dir(root=source_dir, skip_files=skip_files)
        vectorstore.upsert(chunks)
        sparse_index.build(chunks)
        span.set_attribute("chunk_count", len(chunks))
