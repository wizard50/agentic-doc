from pathlib import Path

import pytest

from agentic_doc_explorer.startup_ingest import maybe_run_startup_ingest
from agentic_doc_rag.config import RagSettings


def test_maybe_run_startup_ingest_skips_when_flag_disabled(tmp_path: Path) -> None:
    settings = RagSettings(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="test",
        bm25_persist_dir=tmp_path / "bm25",
        ingest_source_dir=tmp_path / "missing",
        ingest_on_startup=False,
    )
    assert maybe_run_startup_ingest(settings) is None


def test_maybe_run_startup_ingest_skips_when_store_nonempty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = RagSettings(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="test",
        bm25_persist_dir=tmp_path / "bm25",
        ingest_source_dir=tmp_path / "docs",
        ingest_on_startup=True,
    )

    class _Store:
        def count(self) -> int:
            return 3

    monkeypatch.setattr(
        "agentic_doc_explorer.startup_ingest.create_vector_store",
        lambda _settings: _Store(),
    )
    assert maybe_run_startup_ingest(settings) is None


def test_maybe_run_startup_ingest_indexes_when_empty(tmp_path: Path) -> None:
    source = tmp_path / "docs"
    source.mkdir()
    (source / "intro.md").write_text("# Intro\n\nHello ownership.\n", encoding="utf-8")

    settings = RagSettings(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="startup",
        bm25_persist_dir=tmp_path / "bm25",
        ingest_source_dir=source,
        ingest_on_startup=True,
        ingest_skip_files="",
    )
    result = maybe_run_startup_ingest(settings)
    assert result is not None
    assert result.chunk_count > 0
    assert result.file_count == 1

    assert maybe_run_startup_ingest(settings) is None


def test_maybe_run_startup_ingest_raises_when_source_missing(tmp_path: Path) -> None:
    from agentic_doc_rag.ingest import IngestSourceNotFoundError

    settings = RagSettings(
        chroma_persist_dir=tmp_path / "chroma",
        chroma_collection_name="startup",
        bm25_persist_dir=tmp_path / "bm25",
        ingest_source_dir=tmp_path / "nope",
        ingest_on_startup=True,
    )
    with pytest.raises(IngestSourceNotFoundError):
        maybe_run_startup_ingest(settings)
