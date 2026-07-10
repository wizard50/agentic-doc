from collections.abc import Iterator
from pathlib import Path

import pytest
from openinference.semconv.trace import DocumentAttributes, SpanAttributes
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from phoenix.trace.attributes import unflatten

from agentic_doc_rag.vectorstore.chroma import ChromaVectorStore


@pytest.fixture
def span_exporter() -> Iterator[InMemorySpanExporter]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    trace._TRACER_PROVIDER = provider
    yield exporter
    exporter.clear()


def test_chunk_markdown_dir_emits_span(span_exporter: InMemorySpanExporter, tmp_path: Path) -> None:
    from agentic_doc_rag.chunk.chunker import chunk_markdown_dir

    (tmp_path / "doc.md").write_text("## Section\n\nHello", encoding="utf-8")

    chunks = chunk_markdown_dir(tmp_path)

    spans = span_exporter.get_finished_spans()
    assert len(spans) == 1
    attributes = spans[0].attributes
    assert spans[0].name == "chunk.markdown_dir"
    assert attributes is not None
    assert attributes["file_count"] == 1
    assert attributes["chunk_count"] == len(chunks)


def test_chroma_upsert_and_search_emit_spans(
    span_exporter: InMemorySpanExporter, tmp_path: Path
) -> None:
    from agentic_doc_rag.models import DocumentChunk

    store = ChromaVectorStore(tmp_path / "chroma", "test")
    chunks = [
        DocumentChunk(id="1", text="ownership in Rust", metadata={"source": "book.md"}),
        DocumentChunk(id="2", text="borrowing rules", metadata={"source": "book.md"}),
    ]

    store.upsert(chunks)
    store.search("ownership", k=1)

    spans = span_exporter.get_finished_spans()
    assert [span.name for span in spans] == ["vectorstore.upsert", "vectorstore.search"]
    upsert_attributes = spans[0].attributes
    search_attributes = spans[1].attributes
    assert upsert_attributes is not None
    assert search_attributes is not None
    assert upsert_attributes["chunk_count"] == 2
    assert search_attributes["result_count"] == 1
    assert search_attributes["input.value"] == "ownership"
    assert (
        search_attributes[
            f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.0.{DocumentAttributes.DOCUMENT_CONTENT}"
        ]
        == "ownership in Rust"
    )

    nested = unflatten(search_attributes.items())
    documents = nested["retrieval"]["documents"]
    assert isinstance(documents, list)
    assert documents[0]["document"]["content"] == "ownership in Rust"
