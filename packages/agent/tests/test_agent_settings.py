from pathlib import Path

import pytest

from agentic_doc_agent.config import AgentSettings, get_agent_settings


def test_settings_loads() -> None:
    settings = get_agent_settings()
    assert isinstance(settings, AgentSettings)


def test_agent_settings_defaults(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_MODEL", raising=False)
    monkeypatch.delenv("LLM_TEMPERATURE", raising=False)
    monkeypatch.delenv("MAX_TOOL_ROUNDS", raising=False)
    monkeypatch.delenv("DEFAULT_TOP_K", raising=False)
    monkeypatch.delenv("FAITHFULNESS_ENABLED", raising=False)

    settings = AgentSettings()

    assert settings.llm_model == "gpt-4o-mini"
    assert settings.llm_temperature == 0.0
    assert settings.max_tool_rounds == 5
    assert settings.default_top_k == 5
    assert settings.faithfulness_enabled is True


def test_agent_settings_loads_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.2")
    monkeypatch.setenv("MAX_TOOL_ROUNDS", "3")
    monkeypatch.setenv("DEFAULT_TOP_K", "8")
    monkeypatch.setenv("FAITHFULNESS_ENABLED", "false")

    settings = AgentSettings()

    assert settings.llm_model == "openai/gpt-4o-mini"
    assert settings.llm_temperature == 0.2
    assert settings.max_tool_rounds == 3
    assert settings.default_top_k == 8
    assert settings.faithfulness_enabled is False


def test_agent_settings_reads_llm_credentials_from_core(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")

    settings = AgentSettings()

    assert settings.llm_api_key == "test-key"
    assert settings.llm_base_url == "https://openrouter.ai/api/v1"
