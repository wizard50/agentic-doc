from pathlib import Path

from agentic_doc_rag.chunk.chunker import chunk_markdown_file
from agentic_doc_rag.ingest.models import IngestSettings
from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span


class MarkdownParser:
    """Parse Markdown files into document chunks."""

    extensions: frozenset[str] = frozenset({".md"})

    def can_parse(self, path: Path) -> bool:
        return path.suffix.casefold() in self.extensions

    def parse(self, path: Path, settings: IngestSettings) -> list[DocumentChunk]:
        with get_tracer(__name__).start_as_current_span("parser.markdown") as span:
            mark_chain_span(span)
            span.set_attribute("source", str(path))
            chunks = chunk_markdown_file(
                path,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                header_levels=set(settings.header_levels),
            )
            span.set_attribute("chunk_count", len(chunks))
            return chunks
