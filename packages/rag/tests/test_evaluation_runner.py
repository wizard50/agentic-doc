from pathlib import Path

import pytest

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.evaluation.dataset import load_eval_dataset
from agentic_doc_rag.evaluation.models import EvalQuery
from agentic_doc_rag.evaluation.reporting import format_eval_summary
from agentic_doc_rag.evaluation.runner import EmptyVectorStoreError, run_retrieval_eval
from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore

FIXTURES_DIR = Path(__file__).parent / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"
EVAL_DATASET_PATH = FIXTURES_DIR / "eval_dataset.jsonl"


class _StubVectorStore:
    def __init__(self, responses: dict[str, list[SearchResult]], *, count: int) -> None:
        self._responses = responses
        self._count = count

    def upsert(self, chunks: list[DocumentChunk]) -> None:
        raise NotImplementedError

    def search(self, query: str, k: int) -> list[SearchResult]:
        del k
        for eval_query, results in self._responses.items():
            if eval_query in query or query in eval_query:
                return results
        return next(iter(self._responses.values()), [])

    def delete(self, ids: list[str]) -> None:
        raise NotImplementedError

    def count(self) -> int:
        return self._count


def _search_result(chunk_id: str, source: str) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(id=chunk_id, text="example", metadata={"source": source}),
        score=0.1,
    )


def test_run_retrieval_eval_raises_when_collection_is_empty() -> None:
    store = _StubVectorStore({}, count=0)
    queries = [
        EvalQuery(
            id="q1",
            query="ownership",
            expected_sources=["ownership.md"],
        )
    ]

    with pytest.raises(EmptyVectorStoreError, match="empty"):
        run_retrieval_eval(store, queries, top_k=3, dataset_name="test")


def test_run_retrieval_eval_computes_metrics_from_vector_store() -> None:
    queries = [
        EvalQuery(
            id="ownership-fixture",
            query="What is ownership in Rust?",
            expected_sources=["ownership.md"],
            tags=["ownership"],
        ),
        EvalQuery(
            id="borrowing-fixture",
            query="What is borrowing in Rust?",
            expected_sources=["borrowing.md"],
            tags=["borrowing"],
        ),
    ]
    store = _StubVectorStore(
        {
            "What is ownership in Rust?": [_search_result("1", "ownership.md")],
            "What is borrowing in Rust?": [_search_result("2", "borrowing.md")],
        },
        count=2,
    )

    report = run_retrieval_eval(store, queries, top_k=3, dataset_name="fixture")

    assert report.query_count == 2
    assert report.hit_at_k == 1.0
    assert report.mrr == 1.0


def test_run_retrieval_eval_against_indexed_fixture_corpus(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    store = ChromaVectorStore(tmp_path / "chroma", "eval-fixture")
    store.upsert(chunks)

    queries = load_eval_dataset(EVAL_DATASET_PATH)
    report = run_retrieval_eval(store, queries, top_k=3, dataset_name="eval_dataset.jsonl")

    assert report.query_count == 2
    assert report.hit_at_k == 1.0


def test_format_eval_summary_includes_tag_breakdown() -> None:
    queries = [
        EvalQuery(id="q1", query="ownership", expected_sources=["ownership.md"], tags=["ownership"]),
        EvalQuery(id="q2", query="traits", expected_sources=["traits.md"], tags=["traits"]),
    ]
    store = _StubVectorStore(
        {
            "ownership": [_search_result("1", "ownership.md")],
            "traits": [_search_result("2", "other.md")],
        },
        count=2,
    )
    report = run_retrieval_eval(store, queries, top_k=2, dataset_name="test.jsonl")

    summary = format_eval_summary(
        report,
        dataset_path="test.jsonl",
        collection_name="fixture",
        chunk_count=2,
    )

    assert "explorer eval" in summary
    assert "By tag:" in summary
    assert "ownership" in summary
    assert "traits" in summary
