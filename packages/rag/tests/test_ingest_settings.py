from pathlib import Path

from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.ingest import (
    ingest_settings_from_rag,
    parse_skip_files,
    resolve_ingest_settings,
)


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


def test_resolve_ingest_settings_overrides_source() -> None:
    custom_source = Path("custom/corpus")

    settings = resolve_ingest_settings(RagSettings(), source_dir=custom_source)

    assert settings.source_dir == custom_source
    assert settings.skip_files == frozenset({"SUMMARY.md", "title-page.md"})


def test_resolve_ingest_settings_overrides_skip_files() -> None:
    settings = resolve_ingest_settings(
        RagSettings(),
        skip_files=frozenset({"notes.md"}),
    )

    assert settings.skip_files == frozenset({"notes.md"})


def test_resolve_ingest_settings_leaves_defaults_when_no_overrides() -> None:
    settings = resolve_ingest_settings(RagSettings())

    assert settings == ingest_settings_from_rag(RagSettings())


def test_resolve_ingest_settings_empty_skip_list_indexes_all_files() -> None:
    settings = resolve_ingest_settings(RagSettings(), skip_files=frozenset())

    assert settings.skip_files == frozenset()