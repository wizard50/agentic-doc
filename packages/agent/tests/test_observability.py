from collections.abc import Iterator
from typing import TypeVar

import pytest
from openinference.semconv.trace import OpenInferenceSpanKindValues, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from pydantic import BaseModel

from agentic_doc_agent import (
    AgentRequest,
    AgentSettings,
    AgentStatus,
    ChatMessage,
    ChatResult,
    WorkflowId,
    run_workflow,
)
from agentic_doc_agent.evaluation import FaithfulnessVerdict
from agentic_doc_agent.graphs.answer_models import AnswerDraft
from agentic_doc_agent.observability.tracing import (
    get_tracer,
    mark_agent_span,
    mark_evaluator_span,
    mark_llm_span,
    mark_tool_span,
    truncate_for_span,
)
from agentic_doc_agent.tools import RetrieveTool
from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_rag.retrieval import RetrievalRequest

T = TypeVar("T", bound=BaseModel)


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    """Install an in-memory tracer provider (same pattern as RAG tracing tests)."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    # Prefer private assign: set_tracer_provider rejects overrides once a provider exists.
    previous = getattr(trace, "_TRACER_PROVIDER", None)
    trace._TRACER_PROVIDER = provider
    yield exporter
    exporter.clear()
    trace._TRACER_PROVIDER = previous


class FakeRetriever:
    def __init__(
        self,
        results: list[SearchResult] | None = None,
        *,
        error: Exception | None = None,
    ) -> None:
        self._results = results or []
        self._error = error

    def retrieve(self, request: RetrievalRequest) -> list[SearchResult]:
        if self._error is not None:
            raise self._error
        return list(self._results)

    def count(self) -> int:
        return len(self._results)


class FakeLlm:
    def __init__(
        self,
        draft: AnswerDraft,
        *,
        verdict: FaithfulnessVerdict | None = None,
    ) -> None:
        self._draft = draft
        self._verdict = verdict or FaithfulnessVerdict(
            score=0.9,
            explanation="Grounded.",
        )

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def complete_structured(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> T:
        if schema is FaithfulnessVerdict:
            return schema.model_validate(self._verdict.model_dump())
        return schema.model_validate(self._draft.model_dump())


def _hit(chunk_id: str = "c1") -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(
            id=chunk_id,
            text="ownership rules",
            metadata={"source": f"{chunk_id}.md"},
        ),
        score=0.9,
    )


def test_truncate_for_span() -> None:
    assert truncate_for_span(None) is None
    assert truncate_for_span("short") == "short"
    assert truncate_for_span("abcdefghij", max_chars=6) == "abc..."
    with pytest.raises(ValueError, match="max_chars"):
        truncate_for_span("x", max_chars=0)


def test_mark_helpers_set_openinference_kinds(span_exporter: InMemorySpanExporter) -> None:
    tracer = get_tracer(__name__)
    with tracer.start_as_current_span("agent") as span:
        mark_agent_span(span)
    with tracer.start_as_current_span("tool") as span:
        mark_tool_span(span, name="retrieve")
    with tracer.start_as_current_span("llm") as span:
        mark_llm_span(span)
    with tracer.start_as_current_span("eval") as span:
        mark_evaluator_span(span)

    spans = {s.name: s for s in span_exporter.get_finished_spans()}
    kind = SpanAttributes.OPENINFERENCE_SPAN_KIND
    agent_attrs = spans["agent"].attributes
    tool_attrs = spans["tool"].attributes
    llm_attrs = spans["llm"].attributes
    eval_attrs = spans["eval"].attributes
    assert agent_attrs is not None
    assert tool_attrs is not None
    assert llm_attrs is not None
    assert eval_attrs is not None
    assert agent_attrs[kind] == OpenInferenceSpanKindValues.AGENT.value
    assert tool_attrs[kind] == OpenInferenceSpanKindValues.TOOL.value
    assert tool_attrs[SpanAttributes.TOOL_NAME] == "retrieve"
    assert llm_attrs[kind] == OpenInferenceSpanKindValues.LLM.value
    assert eval_attrs[kind] == OpenInferenceSpanKindValues.EVALUATOR.value


def test_run_workflow_emits_parent_and_child_spans(
    span_exporter: InMemorySpanExporter,
) -> None:
    draft = AnswerDraft(answer="Each value has one owner.", citation_chunk_ids=["c1"])
    result = run_workflow(
        AgentRequest(goal="What is ownership?"),
        retrieve_tool=RetrieveTool(FakeRetriever([_hit()])),
        llm=FakeLlm(draft),
        settings=AgentSettings(faithfulness_enabled=True),
    )

    assert result.status is AgentStatus.SUCCEEDED
    names = [span.name for span in span_exporter.get_finished_spans()]
    assert "agent.run_workflow" in names
    assert "agent.tool.retrieve" in names
    assert "agent.generate" in names
    assert "agent.evaluate" in names

    parent = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.run_workflow")
    attrs = parent.attributes or {}
    assert attrs[SpanAttributes.OPENINFERENCE_SPAN_KIND] == OpenInferenceSpanKindValues.AGENT.value
    assert attrs[SpanAttributes.INPUT_VALUE] == "What is ownership?"
    assert attrs["workflow"] == WorkflowId.ANSWER.value
    assert attrs["agent.status"] == AgentStatus.SUCCEEDED.value
    assert attrs["agent.tool_calls"] == 1
    assert attrs["agent.retrieved_count"] == 1
    assert attrs["agent.citation_count"] == 1
    assert attrs["agent.faithfulness"] == 0.9
    assert attrs[SpanAttributes.OUTPUT_VALUE] == draft.answer


def test_run_workflow_skips_evaluate_span_when_disabled(
    span_exporter: InMemorySpanExporter,
) -> None:
    draft = AnswerDraft(answer="Ownership rules apply.", citation_chunk_ids=["c1"])
    result = run_workflow(
        AgentRequest(goal="ownership?"),
        retrieve_tool=RetrieveTool(FakeRetriever([_hit()])),
        llm=FakeLlm(draft),
        settings=AgentSettings(faithfulness_enabled=False),
    )

    assert result.status is AgentStatus.SUCCEEDED
    names = [span.name for span in span_exporter.get_finished_spans()]
    assert "agent.run_workflow" in names
    assert "agent.tool.retrieve" in names
    assert "agent.generate" in names
    assert "agent.evaluate" not in names


def test_run_workflow_failed_retrieve_still_has_parent(
    span_exporter: InMemorySpanExporter,
) -> None:
    result = run_workflow(
        AgentRequest(goal="x"),
        retrieve_tool=RetrieveTool(FakeRetriever(error=RuntimeError("down"))),
        llm=FakeLlm(AnswerDraft(answer="unused", citation_chunk_ids=[])),
        settings=AgentSettings(faithfulness_enabled=True),
    )

    assert result.status is AgentStatus.FAILED
    names = [span.name for span in span_exporter.get_finished_spans()]
    assert "agent.run_workflow" in names
    assert "agent.tool.retrieve" in names
    assert "agent.generate" not in names
    assert "agent.evaluate" not in names

    parent = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.run_workflow")
    attrs = parent.attributes or {}
    assert attrs["agent.status"] == AgentStatus.FAILED.value
    assert "retrieve failed" in str(attrs.get("error.message", ""))


def test_truncate_long_output_on_parent_span(span_exporter: InMemorySpanExporter) -> None:
    long_answer = "A" * 3000
    draft = AnswerDraft(answer=long_answer, citation_chunk_ids=[])
    run_workflow(
        AgentRequest(goal="q"),
        retrieve_tool=RetrieveTool(FakeRetriever([_hit()])),
        llm=FakeLlm(draft),
        settings=AgentSettings(faithfulness_enabled=False),
    )

    parent = next(s for s in span_exporter.get_finished_spans() if s.name == "agent.run_workflow")
    output = (parent.attributes or {})[SpanAttributes.OUTPUT_VALUE]
    assert isinstance(output, str)
    assert len(output) == 2000
    assert output.endswith("...")
