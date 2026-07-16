import json
from pathlib import Path

from support.builders import search_result

from agentic_doc_rag.evaluation.metrics import compute_report
from agentic_doc_rag.evaluation.models import EvalQuery, LlmEvalReport
from agentic_doc_rag.evaluation.persistence import build_eval_payload, save_eval_report


def test_build_eval_payload_includes_metadata_and_optional_llm_report() -> None:
    queries = [EvalQuery(id="q1", query="ownership", expected_sources=["ownership.md"])]
    report = compute_report(
        queries,
        {"q1": [search_result("1", "ownership.md")]},
        top_k=1,
        dataset_name="test.jsonl",
    )
    llm_report = LlmEvalReport(model="gpt-4o-mini", precision_at_k=1.0, llm_hit_at_k=1.0)

    payload = build_eval_payload(
        report,
        llm_report=llm_report,
        metadata={"dataset": "test.jsonl"},
    )

    assert payload["metadata"]["dataset"] == "test.jsonl"
    assert payload["report"]["hit_at_k"] == 1.0
    assert payload["llm_report"]["model"] == "gpt-4o-mini"


def test_save_eval_report_writes_timestamped_json(tmp_path: Path) -> None:
    payload = {"metadata": {"dataset": "test.jsonl"}, "report": {"hit_at_k": 0.5}}

    report_path = save_eval_report(payload, tmp_path / "reports")

    assert report_path.parent.exists()
    assert report_path.name.startswith("eval_")
    assert report_path.name.endswith(".json")
    assert json.loads(report_path.read_text(encoding="utf-8")) == payload


def test_save_eval_report_uses_llm_suffix(tmp_path: Path) -> None:
    payload = {
        "metadata": {},
        "report": {"hit_at_k": 0.5},
        "llm_report": {"precision_at_k": 0.8},
    }

    report_path = save_eval_report(payload, tmp_path)

    assert report_path.name.endswith("_llm.json")
