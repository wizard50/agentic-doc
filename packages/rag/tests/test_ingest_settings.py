from pathlib import Path

from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.ingest import ingest_settings_from_rag, parse_skip_files


def test_parse_skip_files_splits_comma_separated_names() -> None:
    assert parse_skip_files("SUMMARY.md, title-page.md") == frozenset(
        {"SUMMARY.md", "title-page.md"}
    )


def test_parse_skip_files_returns_empty_for_blank_value() -> None:
    assert parse_skip_files("") == frozenset()
    assert parse_skip_files("   ") == frozenset()


def test_ingest_settings_from_rag_uses_defaults() -> None:
    settings = ingest_settings_from_rag(RagSettings())

    assert settings.source_dir == Path("data/download/rust-book/src")
    assert settings.skip_files == frozenset({"SUMMARY.md", "title-page.md"})
    assert settings.chunk_size == 1500
    assert settings.chunk_overlap == 200
    assert settings.header_levels == frozenset({2, 3})