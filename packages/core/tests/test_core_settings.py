from pathlib import Path

import pytest

from agentic_doc_core.config import (
    CoreSettings,
    PhoenixSettings,
    get_core_settings,
    get_phoenix_settings,
)


def test_settings_loads() -> None:
    settings = get_core_settings()
    assert isinstance(settings, CoreSettings)


def test_phoenix_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("PHOENIX_ENABLED", raising=False)
    monkeypatch.delenv("PHOENIX_COLLECTOR_ENDPOINT", raising=False)
    monkeypatch.delenv("PHOENIX_PROJECT_NAME", raising=False)

    settings = PhoenixSettings()

    assert settings.enabled is False
    assert settings.collector_endpoint == "http://localhost:4317"
    assert settings.project_name == "agentic-doc"


def test_phoenix_loads_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PHOENIX_ENABLED", "true")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:4317")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "test-project")

    settings = PhoenixSettings()

    assert settings.enabled is True
    assert settings.collector_endpoint == "http://phoenix:4317"
    assert settings.project_name == "test-project"


def test_core_settings_exposes_phoenix(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("PHOENIX_ENABLED", "true")
    monkeypatch.setenv("PHOENIX_COLLECTOR_ENDPOINT", "http://phoenix:4317")
    monkeypatch.setenv("PHOENIX_PROJECT_NAME", "test-project")

    core = CoreSettings()

    assert core.phoenix == get_phoenix_settings()
    assert core.phoenix.enabled is True
    assert core.phoenix.collector_endpoint == "http://phoenix:4317"
    assert core.phoenix.project_name == "test-project"
