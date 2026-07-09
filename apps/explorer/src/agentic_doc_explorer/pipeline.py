from pathlib import Path

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.vectorstore.base import VectorStore


def run_ingestion(
    source_dir: Path,
    vectorstore: VectorStore,
    skip_files: frozenset[str] | None = None,
) -> None:
    print(f"Running ingestion from {source_dir}...")
    chunks: list[DocumentChunk] = chunk_markdown_dir(root=source_dir, skip_files=skip_files)
    vectorstore.upsert(chunks)