"""ApiSettings unit tests."""

from collections.abc import Iterator

import pytest

from agentic_doc_api.config import ApiSettings, get_api_settings
from agentic_doc_core.config import Environment


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    get_api_settings.cache_clear()
    yield
    get_api_settings.cache_clear()


def test_settings_defaults() -> None:
    settings = ApiSettings()
    assert settings.app_name == "Agentic Doc API"
    assert settings.app_version == "0.1.0"
    assert settings.environment == Environment.DEV
    assert settings.log_level == "INFO"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.allowed_origins == ["http://localhost:5173"]
    assert settings.docs_enabled is True


def test_allowed_origins_from_comma_string(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(
        "ALLOWED_ORIGINS",
        "http://localhost:5173, https://example.com ",
    )
    settings = ApiSettings()
    assert settings.allowed_origins == [
        "http://localhost:5173",
        "https://example.com",
    ]


def test_log_level_normalized(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = ApiSettings()
    assert settings.log_level == "DEBUG"


def test_get_api_settings_cached() -> None:
    first = get_api_settings()
    second = get_api_settings()
    assert first is second
