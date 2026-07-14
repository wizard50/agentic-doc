from pathlib import Path

import pytest

from agentic_doc_rag.evaluation.config import EvalSettings, get_eval_settings


def test_eval_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("EVAL_TOP_K", raising=False)
    monkeypatch.delenv("EVAL_DATASET_PATH", raising=False)
    monkeypatch.delenv("EVAL_FAIL_UNDER_HIT_AT_K", raising=False)

    settings = EvalSettings()

    assert settings.top_k == 5
    assert settings.dataset_path.name == "rust_book.jsonl"
    assert settings.fail_under_hit_at_k is None
    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_concurrency == 5


def test_eval_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("EVAL_TOP_K", "3")
    monkeypatch.setenv("EVAL_DATASET_PATH", "custom.jsonl")
    monkeypatch.setenv("EVAL_FAIL_UNDER_HIT_AT_K", "0.8")

    settings = EvalSettings()

    assert settings.top_k == 3
    assert settings.dataset_path == Path("custom.jsonl")
    assert settings.fail_under_hit_at_k == 0.8


def test_get_eval_settings_returns_instance() -> None:
    assert isinstance(get_eval_settings(), EvalSettings)