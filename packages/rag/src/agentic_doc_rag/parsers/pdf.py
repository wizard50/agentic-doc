from pathlib import Path

import pymupdf

from agentic_doc_rag.chunk.chunker import _chunk_id, split_with_overlap
from agentic_doc_rag.ingest.models import IngestSettings
from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return (1-based page number, plain text) pairs from a PDF text layer."""
    document = pymupdf.open(path)
    try:
        pages: list[tuple[int, str]] = []
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            # get_text is overloaded; "text" always returns a string at runtime.
            text = page.get_text("text")
            pages.append((page_index + 1, text if isinstance(text, str) else str(text)))
        return pages
    finally:
        document.close()


class PdfParser:
    """Parse PDF files via the embedded text layer (no OCR)."""

    extensions: frozenset[str] = frozenset({".pdf"})

    def can_parse(self, path: Path) -> bool:
        return path.suffix.casefold() in self.extensions

    def parse(self, path: Path, settings: IngestSettings) -> list[DocumentChunk]:
        with get_tracer(__name__).start_as_current_span("parser.pdf") as span:
            mark_chain_span(span)
            span.set_attribute("source", str(path))

            pages = extract_pdf_pages(path)
            empty_page_count = 0
            chunks: list[DocumentChunk] = []

            for page_number, text in pages:
                if not text.strip():
                    empty_page_count += 1
                    continue

                section_path = f"Page {page_number}"
                parts = split_with_overlap(
                    text,
                    settings.chunk_size,
                    settings.chunk_overlap,
                )
                for index, part in enumerate(parts):
                    chunks.append(
                        DocumentChunk(
                            id=_chunk_id(path, section_path, index),
                            text=part,
                            metadata={
                                "source": str(path),
                                "page": page_number,
                                "section_path": section_path,
                                "file_type": "pdf",
                            },
                        )
                    )

            span.set_attribute("page_count", len(pages))
            span.set_attribute("empty_page_count", empty_page_count)
            span.set_attribute("chunk_count", len(chunks))
            return chunks
