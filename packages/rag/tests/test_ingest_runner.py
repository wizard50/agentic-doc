from pathlib import Path
from shutil import copy2

import pytest
from support.paths import CORPUS_DIR, SAMPLE_PDF_PATH, SAMPLE_RUST_PATH

from agentic_doc_rag.ingest import (
    IngestEmptyCorpusError,
    IngestSettings,
    IngestSourceNotFoundError,
    run_ingestion,
)
from agentic_doc_rag.models import DocumentChunk, SearchResult


class _RecordingVectorStore:
    def __init__(self) -> None:
        self.upserted: list[DocumentChunk] = []

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        self.upserted.extend(chunks)

    def search(self, query: str, k: int) -> list[SearchResult]:
        raise NotImplementedError

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        return len(self.upserted)


class _RecordingSparseIndex:
    def __init__(self) -> None:
        self.built: list[DocumentChunk] = []

    def build(self, chunks: list[DocumentChunk]) -> None:
        self.built = list(chunks)

    def search(self, query: str, k: int) -> list[SearchResult]:
        raise NotImplementedError

    def count(self) -> int:
        return len(self.built)


def test_run_ingestion_indexes_fixture_corpus() -> None:
    vectorstore = _RecordingVectorStore()
    sparse_index = _RecordingSparseIndex()
    settings = IngestSettings(source_dir=CORPUS_DIR)

    result = run_ingestion(vectorstore, sparse_index, settings)

    assert result.chunk_count > 0
    assert result.file_count > 0
    assert len(vectorstore.upserted) == result.chunk_count
    assert len(sparse_index.built) == result.chunk_count


def test_run_ingestion_raises_when_source_dir_is_missing(tmp_path: Path) -> None:
    settings = IngestSettings(source_dir=tmp_path / "missing")

    with pytest.raises(IngestSourceNotFoundError, match="does not exist"):
        run_ingestion(_RecordingVectorStore(), _RecordingSparseIndex(), settings)


def test_run_ingestion_raises_when_source_dir_has_no_indexable_files(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    (empty_dir / "notes.txt").write_text("not indexed yet\n", encoding="utf-8")
    settings = IngestSettings(source_dir=empty_dir)

    with pytest.raises(IngestEmptyCorpusError, match="No indexable files"):
        run_ingestion(_RecordingVectorStore(), _RecordingSparseIndex(), settings)


def test_run_ingestion_indexes_mixed_markdown_and_pdf(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("## Notes\n\nMarkdown ownership note.\n", encoding="utf-8")
    pdf_path = tmp_path / "manual.pdf"
    copy2(SAMPLE_PDF_PATH, pdf_path)

    vectorstore = _RecordingVectorStore()
    sparse_index = _RecordingSparseIndex()
    result = run_ingestion(vectorstore, sparse_index, IngestSettings(source_dir=tmp_path))

    sources = {chunk.metadata["source"] for chunk in vectorstore.upserted}
    texts = " ".join(chunk.text for chunk in vectorstore.upserted)

    assert result.file_count == 2
    assert str(tmp_path / "notes.md") in sources
    assert str(pdf_path) in sources
    assert "Markdown ownership" in texts
    assert "Ownership in Rust" in texts


def test_run_ingestion_indexes_mixed_markdown_pdf_and_code(tmp_path: Path) -> None:
    (tmp_path / "notes.md").write_text("## Notes\n\nMarkdown ownership note.\n", encoding="utf-8")
    copy2(SAMPLE_PDF_PATH, tmp_path / "manual.pdf")
    copy2(SAMPLE_RUST_PATH, tmp_path / "lib.rs")

    vectorstore = _RecordingVectorStore()
    result = run_ingestion(
        vectorstore,
        _RecordingSparseIndex(),
        IngestSettings(source_dir=tmp_path),
    )

    sources = {Path(chunk.metadata["source"]).name for chunk in vectorstore.upserted}
    languages = {
        chunk.metadata.get("language")
        for chunk in vectorstore.upserted
        if chunk.metadata.get("file_type") == "code"
    }

    assert result.file_count == 3
    assert sources == {"notes.md", "manual.pdf", "lib.rs"}
    assert "rust" in languages
    assert any("take_ownership" in chunk.text for chunk in vectorstore.upserted)
