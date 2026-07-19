from pathlib import Path
from typing import Any

import pymupdf4llm

from agentic_doc_rag.chunk.chunker import chunk_markdown_text, make_chunk_id
from agentic_doc_rag.ingest.models import IngestSettings
from agentic_doc_rag.models import DocumentChunk
from agentic_doc_rag.observability.tracing import get_tracer, mark_chain_span


def extract_pdf_pages(path: Path) -> list[tuple[int, str]]:
    """Return (1-based page number, markdown text) pairs via pymupdf4llm.

    Uses layout-aware conversion without OCR (text layer only).
    """
    # Stubs type this as str | list[dict]; page_chunks=True always returns a list.
    raw = pymupdf4llm.to_markdown(
        str(path),
        page_chunks=True,
        use_ocr=False,
    )
    if isinstance(raw, str):
        return [(1, raw)] if raw.strip() else []

    pages: list[tuple[int, str]] = []
    for page in raw:
        if not isinstance(page, dict):
            continue
        metadata: dict[str, Any] = page.get("metadata") or {}
        if not isinstance(metadata, dict):
            metadata = {}
        page_number = int(metadata.get("page_number", len(pages) + 1))
        text = page.get("text") or ""
        if not isinstance(text, str):
            text = str(text)
        pages.append((page_number, text))
    return pages


class PdfParser:
    """Parse PDF files to markdown with pymupdf4llm, then chunk like Markdown."""

    extensions: frozenset[str] = frozenset({".pdf"})

    def can_parse(self, path: Path) -> bool:
        return path.suffix.casefold() in self.extensions

    def parse(self, path: Path, settings: IngestSettings) -> list[DocumentChunk]:
        with get_tracer(__name__).start_as_current_span("parser.pdf") as span:
            mark_chain_span(span)
            span.set_attribute("source", str(path))
            span.set_attribute("extractor", "pymupdf4llm")

            pages = extract_pdf_pages(path)
            empty_page_count = 0
            chunks: list[DocumentChunk] = []

            for page_number, text in pages:
                if not text.strip():
                    empty_page_count += 1
                    continue

                page_chunks = chunk_markdown_text(
                    text,
                    path,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                    header_levels=set(settings.header_levels),
                )
                for index, chunk in enumerate(page_chunks):
                    base_section = str(chunk.metadata.get("section_path") or "").strip()
                    section_path = (
                        f"Page {page_number} > {base_section}"
                        if base_section
                        else f"Page {page_number}"
                    )
                    chunks.append(
                        DocumentChunk(
                            id=make_chunk_id(path, section_path, index),
                            text=chunk.text,
                            metadata={
                                **chunk.metadata,
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
