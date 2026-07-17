from pathlib import Path

from agentic_doc_rag.ingest.models import IngestSettings
from agentic_doc_rag.parsers import (
    MarkdownParser,
    default_parsers,
    discover_files,
    parser_for_path,
    supported_extensions,
)


def test_supported_extensions_includes_markdown() -> None:
    assert ".md" in supported_extensions(default_parsers())


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


def test_parser_for_path_returns_markdown_parser() -> None:
    parsers = default_parsers()
    path = Path("chapter.md")

    assert isinstance(parser_for_path(path, parsers), MarkdownParser)
    assert parser_for_path(Path("chapter.pdf"), parsers) is None