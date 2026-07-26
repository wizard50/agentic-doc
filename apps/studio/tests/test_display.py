from agentic_doc_agent import StepEvent, StepKind
from agentic_doc_rag.models import DocumentChunk, SearchResult
from agentic_doc_studio.display import (
    faithfulness_caption,
    format_faithfulness,
    retrieved_by_chunk_id,
    step_title,
)


def test_format_faithfulness() -> None:
    assert format_faithfulness(None) == "—"
    assert format_faithfulness(0.875) == "0.88"
    assert format_faithfulness(0.0) == "0.00"


def test_faithfulness_caption() -> None:
    assert faithfulness_caption(None) == "Disabled or judge skipped"
    assert "Strong" in faithfulness_caption(0.9)
    assert "Moderate" in faithfulness_caption(0.6)
    assert "Weak" in faithfulness_caption(0.2)


def test_retrieved_by_chunk_id() -> None:
    hits = [
        SearchResult(
            chunk=DocumentChunk(id="a", text="A", metadata={}),
            score=0.1,
        ),
        SearchResult(
            chunk=DocumentChunk(id="b", text="B", metadata={}),
            score=0.2,
        ),
        SearchResult(
            chunk=DocumentChunk(id="a", text="A2", metadata={}),
            score=0.3,
        ),
    ]
    by_id = retrieved_by_chunk_id(hits)
    assert set(by_id) == {"a", "b"}
    assert by_id["a"].chunk.text == "A2"
    assert by_id["b"].score == 0.2


def test_step_title() -> None:
    step = StepEvent(
        kind=StepKind.PLAN,
        name="plan",
        detail="Planned search query: ownership",
        payload={"search_query": "ownership"},
    )
    title = step_title(1, step)
    assert title.startswith("**1.**")
    assert "`plan`" in title
    assert "ownership" in title

    bare = StepEvent(kind=StepKind.TOOL, name="retrieve")
    assert step_title(2, bare) == "**2.** `tool` · retrieve"
