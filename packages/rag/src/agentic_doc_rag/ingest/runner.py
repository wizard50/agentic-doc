from agentic_doc_rag.ingest.errors import IngestEmptyCorpusError, IngestSourceNotFoundError
from agentic_doc_rag.ingest.models import IngestResult, IngestSettings
from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span
from agentic_doc_rag.parsers.discovery import discover_files
from agentic_doc_rag.parsers.registry import default_parsers, parser_for_path, supported_extensions
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

        parsers = default_parsers()
        files = discover_files(
            source_dir,
            skip_files=settings.skip_files,
            extensions=supported_extensions(parsers),
        )
        chunks: list[DocumentChunk] = []
        for path in files:
            parser = parser_for_path(path, parsers)
            if parser is None:
                continue
            chunks.extend(parser.parse(path, settings))

        if not chunks:
            msg = (
                f"No indexable files under {source_dir.resolve()}. "
                "Check --source, INGEST_SOURCE_DIR, and --skip."
            )
            raise IngestEmptyCorpusError(msg)

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