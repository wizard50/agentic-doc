"""Unit tests for smoke step-sequence validation (imported from the smoke script)."""

import importlib.util
from pathlib import Path

_SMOKE_PATH = Path(__file__).resolve().parents[3] / "scripts" / "smoke_answer.py"
_spec = importlib.util.spec_from_file_location("smoke_answer", _SMOKE_PATH)
assert _spec is not None and _spec.loader is not None
_smoke = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_smoke)
_steps_look_valid = _smoke._steps_look_valid


def test_steps_valid_plan_retrieve_generate_evaluate() -> None:
    assert _steps_look_valid(["plan", "retrieve", "generate", "evaluate"])


def test_steps_valid_without_plan() -> None:
    assert _steps_look_valid(["retrieve", "generate"])
    assert _steps_look_valid(["retrieve", "generate", "evaluate"])


def test_steps_valid_re_retrieve_loop() -> None:
    assert _steps_look_valid(["plan", "retrieve", "generate", "retrieve", "generate", "evaluate"])
    assert _steps_look_valid(["retrieve", "generate", "retrieve", "generate"])


def test_steps_invalid_sequences() -> None:
    assert not _steps_look_valid([])
    assert not _steps_look_valid(["plan"])
    assert not _steps_look_valid(["generate", "retrieve"])
    assert not _steps_look_valid(["retrieve", "evaluate"])
    assert not _steps_look_valid(["plan", "retrieve", "generate", "evaluate", "extra"])
