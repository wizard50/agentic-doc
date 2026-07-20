from pathlib import Path

from agentic_doc_explorer.constants import EXAMPLE_QUERIES, PHOENIX_UI_URL
from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.ingest import ingest_settings_from_rag


def test_example_queries_defined() -> None:
    assert len(EXAMPLE_QUERIES) >= 3


def test_default_ingest_settings_use_rust_book_paths() -> None:
    settings = ingest_settings_from_rag(RagSettings())

    assert settings.source_dir == Path("corpora/rust-book/src")
    assert "SUMMARY.md" in settings.skip_files


def test_phoenix_ui_url() -> None:
    assert PHOENIX_UI_URL == "http://localhost:6006"