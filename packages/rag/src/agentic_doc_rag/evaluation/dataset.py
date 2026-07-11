import json
from pathlib import Path

from agentic_doc_rag.evaluation.models import EvalQuery

DEFAULT_DATASET_PATH = Path(__file__).parent / "datasets" / "rust_book.jsonl"


def load_eval_dataset(path: Path | str | None = None) -> list[EvalQuery]:
    """Load and validate a golden evaluation dataset from JSONL."""
    dataset_path = Path(path) if path is not None else DEFAULT_DATASET_PATH
    if not dataset_path.exists():
        msg = f"Evaluation dataset not found: {dataset_path}"
        raise FileNotFoundError(msg)

    queries: list[EvalQuery] = []
    seen_ids: set[str] = set()

    for line_number, line in enumerate(
        dataset_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue

        try:
            payload = json.loads(line)
            query = EvalQuery.model_validate(payload)
        except (json.JSONDecodeError, ValueError) as exc:
            msg = f"Invalid dataset entry at {dataset_path}:{line_number}: {exc}"
            raise ValueError(msg) from exc

        if query.id in seen_ids:
            msg = f"Duplicate query id '{query.id}' at {dataset_path}:{line_number}"
            raise ValueError(msg)

        seen_ids.add(query.id)
        queries.append(query)

    if not queries:
        msg = f"Evaluation dataset is empty: {dataset_path}"
        raise ValueError(msg)

    return queries
