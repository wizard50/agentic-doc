from pathlib import Path

import pytest
from support.builders import search_result
from support.fakes import StubVectorStore
from support.paths import CORPUS_DIR, EVAL_DATASET_PATH
from support.pipelines import indexed_pipeline_retriever, semantic_pipeline_retriever
from support.vectorstore import chroma_vector_store

from agentic_doc_rag.chunk.chunker import chunk_markdown_dir
from agentic_doc_rag.evaluation.dataset import load_eval_dataset
from agentic_doc_rag.evaluation.models import EvalQuery
from agentic_doc_rag.evaluation.reporting import format_eval_summary
from agentic_doc_rag.evaluation.runner import EmptyVectorStoreError, run_retrieval_eval
from agentic_doc_rag.sparse.bm25 import Bm25Index


def test_run_retrieval_eval_raises_when_collection_is_empty() -> None:
    store = StubVectorStore(count=0)
    queries = [
        EvalQuery(
            id="q1",
            query="ownership",
            expected_sources=["ownership.md"],
        )
    ]

    with pytest.raises(EmptyVectorStoreError, match="empty"):
        run_retrieval_eval(
            semantic_pipeline_retriever(store), queries, top_k=3, dataset_name="test"
        )


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
    store = StubVectorStore(
        {
            "What is ownership in Rust?": [search_result("1", "ownership.md")],
            "What is borrowing in Rust?": [search_result("2", "borrowing.md")],
        },
        count=2,
        match="contains",
    )

    eval_run = run_retrieval_eval(
        semantic_pipeline_retriever(store), queries, top_k=3, dataset_name="fixture"
    )

    assert eval_run.report.query_count == 2
    assert eval_run.report.hit_at_k == 1.0
    assert eval_run.report.mrr == 1.0


def test_run_retrieval_eval_against_indexed_fixture_corpus(tmp_path: Path) -> None:
    chunks = chunk_markdown_dir(CORPUS_DIR)
    store = chroma_vector_store(tmp_path / "chroma", "eval-fixture")
    store.upsert(chunks)
    sparse = Bm25Index(tmp_path / "bm25")
    sparse.build(chunks)
    retriever = indexed_pipeline_retriever(store, sparse)

    queries = load_eval_dataset(EVAL_DATASET_PATH)
    eval_run = run_retrieval_eval(retriever, queries, top_k=3, dataset_name="eval_dataset.jsonl")

    assert eval_run.report.query_count == 2
    assert eval_run.report.hit_at_k == 1.0


def test_format_eval_summary_includes_tag_breakdown() -> None:
    queries = [
        EvalQuery(
            id="q1", query="ownership", expected_sources=["ownership.md"], tags=["ownership"]
        ),
        EvalQuery(id="q2", query="traits", expected_sources=["traits.md"], tags=["traits"]),
    ]
    store = StubVectorStore(
        {
            "ownership": [search_result("1", "ownership.md")],
            "traits": [search_result("2", "other.md")],
        },
        count=2,
        match="contains",
    )
    eval_run = run_retrieval_eval(
        semantic_pipeline_retriever(store), queries, top_k=2, dataset_name="test.jsonl"
    )

    summary = format_eval_summary(
        eval_run.report,
        dataset_path="test.jsonl",
        collection_name="fixture",
        chunk_count=2,
    )

    assert "explorer eval" in summary
    assert "By tag:" in summary
    assert "ownership" in summary
    assert "traits" in summary
