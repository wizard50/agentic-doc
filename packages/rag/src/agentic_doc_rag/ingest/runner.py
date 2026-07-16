from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.ingest.errors import IngestSourceNotFoundError
from agentic_doc_rag.ingest.models import IngestResult, IngestSettings
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span
from agentic_doc_rag.sparse.protocols import SparseIndex
from agentic_doc_rag.vectorstore.base import VectorStore


def run_ingestion(
    vectorstore: VectorStore,
    sparse_index: SparseIndex,
    settings: IngestSettings,
) -> IngestResult:
    source_dir = settings.source_dir
    if not source_dir.exists():
        msg = f"Ingest source directory does not exist: {source_dir.resolve()}"
        raise IngestSourceNotFoundError(msg)

    with get_tracer(__name__).start_as_current_span("ingest.run") as span:
        mark_chain_span(span)
        span.set_attribute("source_dir", str(source_dir.resolve()))

        chunks = chunk_markdown_dir(
            root=source_dir,
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            header_levels=set(settings.header_levels),
            skip_files=settings.skip_files,
        )
        vectorstore.upsert(chunks)
        sparse_index.build(chunks)

        file_count = len({chunk.metadata.get("source", "") for chunk in chunks})
        span.set_attribute("file_count", file_count)
        span.set_attribute("chunk_count", len(chunks))

        return IngestResult(
            source_dir=source_dir,
            file_count=file_count,
            chunk_count=len(chunks),
        )