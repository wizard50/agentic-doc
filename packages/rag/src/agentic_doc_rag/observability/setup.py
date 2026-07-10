from phoenix.otel import register

from agentic_doc_core.config import PhoenixSettings

_tracing_registered = False


def register_tracing(settings: PhoenixSettings) -> None:
    """Register OpenTelemetry tracing with Phoenix when enabled."""
    global _tracing_registered

    if not settings.enabled or _tracing_registered:
        return

    register(
        endpoint=settings.collector_endpoint,
        project_name=settings.project_name,
        verbose=False,
    )
    _tracing_registered = True


def _reset_tracing_state() -> None:
    """Reset registration state for tests."""
    global _tracing_registered
    _tracing_registered = False
