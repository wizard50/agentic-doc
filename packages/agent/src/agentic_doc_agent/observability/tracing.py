"""OpenTelemetry / OpenInference helpers for agent workflow spans."""

from __future__ import annotations

from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

DEFAULT_SPAN_ATTR_MAX_CHARS = 2000


def get_tracer(name: str) -> Tracer:
    return trace.get_tracer(name)


def truncate_for_span(
    text: str | None,
    *,
    max_chars: int = DEFAULT_SPAN_ATTR_MAX_CHARS,
) -> str | None:
    """Truncate long attribute values for Phoenix-friendly payloads."""
    if text is None:
        return None
    if max_chars < 1:
        raise ValueError("max_chars must be >= 1")
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3] + "..."


def mark_agent_span(span: Span) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.AGENT.value,
    )


def mark_tool_span(span: Span, *, name: str) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.TOOL.value,
    )
    span.set_attribute(SpanAttributes.TOOL_NAME, name)


def mark_llm_span(span: Span) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.LLM.value,
    )


def mark_evaluator_span(span: Span) -> None:
    span.set_attribute(
        SpanAttributes.OPENINFERENCE_SPAN_KIND,
        OpenInferenceSpanKindValues.EVALUATOR.value,
    )


def set_input_value(
    span: Span, value: str | None, *, max_chars: int = DEFAULT_SPAN_ATTR_MAX_CHARS
) -> None:
    truncated = truncate_for_span(value, max_chars=max_chars)
    if truncated is not None:
        span.set_attribute(SpanAttributes.INPUT_VALUE, truncated)


def set_output_value(
    span: Span,
    value: str | None,
    *,
    max_chars: int = DEFAULT_SPAN_ATTR_MAX_CHARS,
) -> None:
    truncated = truncate_for_span(value, max_chars=max_chars)
    if truncated is not None:
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, truncated)


def set_span_error(span: Span, message: str) -> None:
    """Mark a span as failed without re-raising."""
    span.set_attribute("error.message", message)
    span.set_status(Status(StatusCode.ERROR, message))


def record_exception(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
