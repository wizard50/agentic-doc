"""Shared fixtures for the API test suite."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from agentic_doc_api.config import get_api_settings
from agentic_doc_api.main import create_app


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """HTTP client against a fresh app with default settings cache cleared."""
    get_api_settings.cache_clear()
    monkeypatch.delenv("ALLOWED_ORIGINS", raising=False)
    monkeypatch.delenv("API_HOST", raising=False)
    monkeypatch.delenv("API_PORT", raising=False)
    monkeypatch.delenv("API_DOCS_ENABLED", raising=False)
    monkeypatch.delenv("LOG_LEVEL", raising=False)
    monkeypatch.delenv("ENVIRONMENT", raising=False)
    get_api_settings.cache_clear()
    return TestClient(create_app())
