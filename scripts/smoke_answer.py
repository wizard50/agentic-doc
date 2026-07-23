#!/usr/bin/env python3
"""Live smoke for the M2 Answer workflow.

Requires (from the workspace root):
  - Indexed corpus: ``uv run explorer ingest``
  - LLM credentials in ``.env`` (``LLM_API_KEY``; optional ``LLM_BASE_URL`` / ``LLM_MODEL``)

Usage:
  uv run python scripts/smoke_answer.py
  uv run python scripts/smoke_answer.py --goal "What is borrowing?"
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _require_workspace_root() -> None:
    cwd = Path.cwd()
    if (cwd / "packages").is_dir() and (cwd / "apps").is_dir():
        return
    print(
        "smoke_answer must be run from the workspace root (agentic-doc/).\n\n"
        "  cd path/to/agentic-doc\n"
        "  uv run python scripts/smoke_answer.py\n",
        file=sys.stderr,
    )
    raise SystemExit(1)


def main() -> int:
    _require_workspace_root()

    parser = argparse.ArgumentParser(description="Smoke-test run_workflow(ANSWER).")
    parser.add_argument(
        "--goal",
        default="What is ownership in Rust?",
        help="User goal for the answer workflow",
    )
    args = parser.parse_args()

    from agentic_doc_agent import AgentRequest, AgentStatus, WorkflowId, run_workflow
    from agentic_doc_rag.config import get_rag_settings
    from agentic_doc_rag.retrieval import create_retriever

    settings = get_rag_settings()
    count = create_retriever(settings).count()
    if count == 0:
        print(
            "Index is empty (retriever.count() == 0).\n"
            "Index the demo corpus first:\n\n"
            "  uv run explorer ingest\n",
            file=sys.stderr,
        )
        return 1

    print(f"index_chunks: {count}")
    print(f"goal: {args.goal}")

    result = run_workflow(
        AgentRequest(workflow=WorkflowId.ANSWER, goal=args.goal),
    )

    answer = result.answer or ""
    step_names = [step.name for step in result.steps]
    print(f"status: {result.status.value}")
    print(f"error: {result.error}")
    print(f"retrieved: {len(result.retrieved)}")
    print(f"citations: {len(result.citations)}")
    print(f"steps: {step_names}")
    print(f"duration_ms: {result.metrics.duration_ms}")
    print(f"faithfulness: {result.metrics.faithfulness}")
    print(f"answer_chars: {len(answer)}")
    print("--- answer ---")
    print(answer if answer else "(empty)")
    print("--- end ---")

    if result.status is not AgentStatus.SUCCEEDED:
        print("SMOKE FAIL: workflow did not succeed", file=sys.stderr)
        return 1
    if not result.retrieved:
        print(
            "SMOKE FAIL: retrieved 0 passages (index may be empty or query mismatch)",
            file=sys.stderr,
        )
        return 1
    if not answer.strip():
        print("SMOKE FAIL: empty answer", file=sys.stderr)
        return 1
    allowed_steps = (
        ["retrieve", "generate"],
        ["retrieve", "generate", "evaluate"],
    )
    if step_names not in allowed_steps:
        print(
            f"SMOKE FAIL: unexpected steps {step_names}",
            file=sys.stderr,
        )
        return 1
    if step_names == ["retrieve", "generate", "evaluate"] and result.metrics.faithfulness is None:
        print(
            "SMOKE WARN: evaluate ran but faithfulness is None (judge may have failed)",
            file=sys.stderr,
        )

    print("SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
