from unittest.mock import MagicMock

import pytest

from agentic_doc_core.config import PhoenixSettings
from agentic_doc_rag.observability.setup import _reset_tracing_state, register_tracing


@pytest.fixture(autouse=True)
def reset_tracing_state() -> None:
    _reset_tracing_state()


def test_register_tracing_skipped_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_register = MagicMock()
    monkeypatch.setattr("agentic_doc_rag.observability.setup.register", mock_register)

    register_tracing(PhoenixSettings(enabled=False))

    mock_register.assert_not_called()


def test_register_tracing_registers_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_register = MagicMock()
    monkeypatch.setattr("agentic_doc_rag.observability.setup.register", mock_register)

    register_tracing(
        PhoenixSettings(
            enabled=True,
            collector_endpoint="http://localhost:4317",
            project_name="test-project",
        )
    )

    mock_register.assert_called_once_with(
        endpoint="http://localhost:4317",
        project_name="test-project",
        verbose=False,
    )


def test_register_tracing_is_idempotent(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_register = MagicMock()
    monkeypatch.setattr("agentic_doc_rag.observability.setup.register", mock_register)

    settings = PhoenixSettings(enabled=True)

    register_tracing(settings)
    register_tracing(settings)

    mock_register.assert_called_once()
