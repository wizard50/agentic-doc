from pathlib import Path

import pytest
from support.paths import CORPUS_DIR

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


def test_run_ingestion_raises_when_source_dir_has_no_markdown(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    settings = IngestSettings(source_dir=empty_dir)

    with pytest.raises(IngestEmptyCorpusError, match="No markdown files indexed"):
        run_ingestion(_RecordingVectorStore(), _RecordingSparseIndex(), settings)