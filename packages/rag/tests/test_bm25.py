from pathlib import Path

import pytest
from support.paths import CORPUS_DIR

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.sparse.bm25 import Bm25Index


def test_bm25_search_ranks_borrowing_chunk_first(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    index = Bm25Index(tmp_path / "bm25")
    index.build(chunks)

    results = index.search("borrowing", k=2)

    assert results
    assert "Borrowing" in results[0].chunk.metadata.get("section_path", "")
    assert results[0].score >= results[1].score


def test_bm25_persist_and_reload_preserves_search_results(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    persist_dir = tmp_path / "bm25"

    built = Bm25Index(persist_dir)
    built.build(chunks)
    expected = built.search("ownership", k=1)

    loaded = Bm25Index(persist_dir)
    actual = loaded.search("ownership", k=1)

    assert loaded.count() == built.count() == len(chunks)
    assert actual[0].chunk.id == expected[0].chunk.id
    assert actual[0].score == pytest.approx(expected[0].score)


def test_bm25_empty_query_returns_no_results(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    index = Bm25Index(tmp_path / "bm25")
    index.build(chunks)

    assert index.search("!!!", k=5) == []


def test_bm25_unbuilt_index_returns_no_results(tmp_path: Path) -> None:
    index = Bm25Index(tmp_path / "bm25")

    assert index.count() == 0
    assert index.search("borrowing", k=3) == []
