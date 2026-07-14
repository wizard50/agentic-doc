import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from agentic_doc_rag.evaluation.models import EvalReport, LlmEvalReport


def build_eval_payload(
    report: EvalReport,
    *,
    llm_report: LlmEvalReport | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable eval report payload."""
    payload: dict[str, Any] = {
        "metadata": metadata or {},
        "report": report.model_dump(),
    }
    if llm_report is not None:
        payload["llm_report"] = llm_report.model_dump()
    return payload


def save_eval_report(payload: dict[str, Any], report_dir: Path) -> Path:
    """Persist an eval report payload and return the written file path."""
    report_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    suffix = "_llm" if "llm_report" in payload else ""
    report_path = report_dir / f"eval_{timestamp}{suffix}.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return report_path
