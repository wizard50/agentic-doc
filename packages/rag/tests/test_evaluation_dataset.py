from pathlib import Path

import pytest

from agentic_doc_rag.evaluation.dataset import DEFAULT_DATASET_PATH, load_eval_dataset
from agentic_doc_rag.evaluation.models import EvalQuery


def test_default_dataset_loads_with_unique_ids() -> None:
    queries = load_eval_dataset()

    assert len(queries) >= 15
    assert len({query.id for query in queries}) == len(queries)
    assert all(isinstance(query, EvalQuery) for query in queries)


def test_load_eval_dataset_rejects_invalid_entry(tmp_path: Path) -> None:
    dataset_path = tmp_path / "invalid.jsonl"
    dataset_path.write_text(
        '{"id": "bad", "query": "missing ground truth"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Invalid dataset entry"):
        load_eval_dataset(dataset_path)


def test_load_eval_dataset_rejects_duplicate_ids(tmp_path: Path) -> None:
    dataset_path = tmp_path / "duplicate.jsonl"
    dataset_path.write_text(
        "\n".join(
            [
                '{"id": "dup", "query": "one", "expected_sources": ["a.md"]}',
                '{"id": "dup", "query": "two", "expected_sources": ["b.md"]}',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Duplicate query id"):
        load_eval_dataset(dataset_path)


def test_default_dataset_path_points_to_rust_book_jsonl() -> None:
    assert DEFAULT_DATASET_PATH.name == "rust_book.jsonl"
    assert DEFAULT_DATASET_PATH.exists()
