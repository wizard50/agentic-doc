from agentic_doc_rag.evaluation.metrics import compute_report, evaluate_query
from agentic_doc_rag.evaluation.models import EvalQuery
from agentic_doc_rag.models import DocumentChunk, SearchResult


def _result(chunk_id: str, source: str, section_path: str = "") -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text="example",
            metadata={"source": source, "section_path": section_path},
        ),
        score=0.1,
    )


def test_evaluate_query_hit_at_first_rank() -> None:
    query = EvalQuery(
        id="ownership-001",
        query="What is ownership in Rust?",
        expected_sources=["ch04-01-what-is-ownership.md"],
        tags=["ownership"],
    )
    results = [
        _result(
            "1", "data/download/rust-book/src/ch04-01-what-is-ownership.md", "What Is Ownership"
        ),
        _result("2", "data/download/rust-book/src/ch04-02-references-and-borrowing.md"),
    ]

    outcome = evaluate_query(query, results, top_k=2)

    assert outcome.hit_at_k is True
    assert outcome.source_hit_at_k is True
    assert outcome.first_match_rank == 1
    assert outcome.recall_at_k == 1.0
    assert outcome.matched_sources == ["ch04-01-what-is-ownership.md"]


def test_evaluate_query_miss_when_expected_source_not_in_top_k() -> None:
    query = EvalQuery(
        id="traits-001",
        query="What are traits in Rust?",
        expected_sources=["ch10-02-traits.md"],
        tags=["traits"],
    )
    results = [
        _result("1", "data/download/rust-book/src/ch06-02-match.md"),
        _result("2", "data/download/rust-book/src/ch05-01-defining-structs.md"),
    ]

    outcome = evaluate_query(query, results, top_k=2)

    assert outcome.hit_at_k is False
    assert outcome.first_match_rank is None
    assert outcome.recall_at_k == 0.0


def test_evaluate_query_section_match_counts_as_hit() -> None:
    query = EvalQuery(
        id="match-001",
        query="How does match work in Rust?",
        expected_section_paths=["The match Control Flow Construct"],
        tags=["match"],
    )
    results = [
        _result(
            "1", "data/download/rust-book/src/ch06-02-match.md", "The match Control Flow Construct"
        ),
    ]

    outcome = evaluate_query(query, results, top_k=1)

    assert outcome.hit_at_k is True
    assert outcome.section_hit_at_k is True
    assert outcome.matched_sections == ["The match Control Flow Construct"]


def test_evaluate_query_recall_with_multiple_expected_sources() -> None:
    query = EvalQuery(
        id="ownership-multi",
        query="Explain ownership and borrowing",
        expected_sources=[
            "ch04-01-what-is-ownership.md",
            "ch04-02-references-and-borrowing.md",
        ],
        tags=["ownership", "borrowing"],
    )
    results = [
        _result("1", "data/download/rust-book/src/ch04-01-what-is-ownership.md"),
        _result("2", "data/download/rust-book/src/ch05-01-defining-structs.md"),
    ]

    outcome = evaluate_query(query, results, top_k=2)

    assert outcome.hit_at_k is True
    assert outcome.recall_at_k == 0.5
    assert outcome.matched_sources == ["ch04-01-what-is-ownership.md"]


def test_compute_report_aggregates_metrics_and_tags() -> None:
    queries = [
        EvalQuery(
            id="q1",
            query="ownership",
            expected_sources=["ch04-01-what-is-ownership.md"],
            tags=["ownership"],
        ),
        EvalQuery(
            id="q2",
            query="traits",
            expected_sources=["ch10-02-traits.md"],
            tags=["traits"],
        ),
        EvalQuery(
            id="q3",
            query="borrowing",
            expected_sources=["ch04-02-references-and-borrowing.md"],
            tags=["borrowing"],
        ),
    ]
    results_by_query_id = {
        "q1": [_result("1", "data/download/rust-book/src/ch04-01-what-is-ownership.md")],
        "q2": [_result("1", "data/download/rust-book/src/ch06-02-match.md")],
        "q3": [
            _result("1", "data/download/rust-book/src/ch05-01-defining-structs.md"),
            _result("2", "data/download/rust-book/src/ch04-02-references-and-borrowing.md"),
        ],
    }

    report = compute_report(
        queries,
        results_by_query_id,
        top_k=2,
        dataset_name="test_dataset",
    )

    assert report.dataset_name == "test_dataset"
    assert report.query_count == 3
    assert report.top_k == 2
    assert report.hit_at_k == 2 / 3
    assert report.mrr == (1.0 + 0.5) / 3
    assert report.source_match_at_k == 2 / 3
    assert {metric.tag: metric.hit_at_k for metric in report.by_tag} == {
        "borrowing": 1.0,
        "ownership": 1.0,
        "traits": 0.0,
    }
