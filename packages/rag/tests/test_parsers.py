from pathlib import Path

from support.paths import EMPTY_PAGE_PDF_PATH, SAMPLE_PDF_PATH

from agentic_doc_rag.ingest.models import IngestSettings
from agentic_doc_rag.parsers import (
    MarkdownParser,
    PdfParser,
    default_parsers,
    discover_files,
    extract_pdf_pages,
    parser_for_path,
    supported_extensions,
)


def test_supported_extensions_includes_markdown_and_pdf() -> None:
    extensions = supported_extensions(default_parsers())
    assert ".md" in extensions
    assert ".pdf" in extensions
    assert ".rs" not in extensions
    assert ".py" not in extensions


def test_discover_files_respects_extensions_and_skip(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("## Keep\n\nx\n", encoding="utf-8")
    (tmp_path / "skip.md").write_text("## Skip\n\nx\n", encoding="utf-8")
    (tmp_path / "notes.txt").write_text("ignore\n", encoding="utf-8")
    nested = tmp_path / "sub"
    nested.mkdir()
    (nested / "nested.md").write_text("## Nested\n\ny\n", encoding="utf-8")

    files = discover_files(
        tmp_path,
        skip_files=frozenset({"skip.md"}),
        extensions=frozenset({".md"}),
    )

    names = {path.name for path in files}
    assert names == {"keep.md", "nested.md"}


def test_markdown_parser_can_parse_and_chunk(tmp_path: Path) -> None:
    path = tmp_path / "doc.md"
    path.write_text("## Section\n\nHello world.\n", encoding="utf-8")
    parser = MarkdownParser()
    settings = IngestSettings(source_dir=tmp_path)

    assert parser.can_parse(path)
    assert not parser.can_parse(tmp_path / "doc.pdf")

    chunks = parser.parse(path, settings)

    assert chunks
    assert "Hello world" in chunks[0].text
    assert str(path) in chunks[0].metadata["source"]
    assert chunks[0].metadata["file_type"] == "markdown"


def test_pdf_parser_extracts_page_text_and_metadata() -> None:
    parser = PdfParser()
    settings = IngestSettings(source_dir=SAMPLE_PDF_PATH.parent)

    assert parser.can_parse(SAMPLE_PDF_PATH)
    pages = extract_pdf_pages(SAMPLE_PDF_PATH)
    assert len(pages) == 2
    assert "Ownership" in pages[0][1]

    chunks = parser.parse(SAMPLE_PDF_PATH, settings)

    assert len(chunks) >= 2
    texts = " ".join(chunk.text for chunk in chunks)
    assert "Ownership in Rust" in texts
    assert "Borrowing lets you" in texts
    assert all(chunk.metadata["file_type"] == "pdf" for chunk in chunks)
    assert {chunk.metadata["page"] for chunk in chunks} == {1, 2}
    assert chunks[0].metadata["section_path"] == "Page 1"


def test_pdf_parser_skips_empty_pages() -> None:
    chunks = PdfParser().parse(
        EMPTY_PAGE_PDF_PATH,
        IngestSettings(source_dir=EMPTY_PAGE_PDF_PATH.parent),
    )

    assert len(chunks) == 1
    assert chunks[0].metadata["page"] == 2
    assert "Only page with text" in chunks[0].text


def test_parser_for_path_returns_markdown_or_pdf_parser() -> None:
    parsers = default_parsers()

    assert isinstance(parser_for_path(Path("chapter.md"), parsers), MarkdownParser)
    assert isinstance(parser_for_path(Path("chapter.pdf"), parsers), PdfParser)
    assert parser_for_path(Path("lib.rs"), parsers) is None
    assert parser_for_path(Path("chapter.txt"), parsers) is None
