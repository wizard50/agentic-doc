from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest
from phoenix.evals import Score

from agentic_doc_rag.evaluation.config import EvalSettings
from agentic_doc_rag.evaluation.metrics import compute_report
from agentic_doc_rag.evaluation.models import EvalQuery
from agentic_doc_rag.evaluation.phoenix import (
    LlmEvalError,
    _fetch_retriever_span_ids,
    build_document_eval_dataframe,
    compute_llm_retrieval_metrics,
    run_document_relevance_eval,
    run_llm_relevance_eval,
    upload_relevance_annotations,
)
from agentic_doc_rag.evaluation.runner import RetrievalEvalRun
from agentic_doc_rag.models import DocumentChunk, SearchResult


def _eval_run(
    queries: list[EvalQuery], results_by_query_id: dict[str, list[SearchResult]], top_k: int
):
    report = compute_report(queries, results_by_query_id, top_k=top_k, dataset_name="test")
    return RetrievalEvalRun(report=report, results_by_query_id=results_by_query_id)


def _search_result(chunk_id: str, text: str, source: str) -> SearchResult:
    return SearchResult(
        chunk=DocumentChunk(id=chunk_id, text=text, metadata={"source": source}),
        score=0.1,
    )


def test_build_document_eval_dataframe_expands_retrieved_documents() -> None:
    queries = [
        EvalQuery(id="q1", query="ownership", expected_sources=["ownership.md"]),
    ]
    results = {
        "q1": [
            _search_result("1", "ownership rules", "ownership.md"),
            _search_result("2", "borrowing rules", "borrowing.md"),
        ]
    }

    dataframe = build_document_eval_dataframe(queries, results)

    assert len(dataframe) == 2
    assert set(dataframe.columns) == {"query_id", "input", "document_text", "document_position"}
    assert dataframe.iloc[0]["document_position"] == 0


def test_compute_llm_retrieval_metrics_aggregates_precision_and_hit() -> None:
    scored_df = pd.DataFrame(
        {
            "query_id": ["q1", "q1", "q2", "q2"],
            "input": ["ownership", "ownership", "traits", "traits"],
            "document_position": [0, 1, 0, 1],
            "document_relevance_score": [
                Score(name="document_relevance", score=1.0, label="relevant"),
                Score(name="document_relevance", score=0.0, label="unrelated"),
                Score(name="document_relevance", score=0.0, label="unrelated"),
                Score(name="document_relevance", score=0.0, label="unrelated"),
            ],
        }
    )

    report = compute_llm_retrieval_metrics(scored_df, top_k=2, model="gpt-4o-mini")

    assert report.model == "gpt-4o-mini"
    assert report.precision_at_k == 0.25
    assert report.llm_hit_at_k == 0.5
    assert len(report.scores) == 4


def test_run_document_relevance_eval_requires_llm_api_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    queries = [EvalQuery(id="q1", query="ownership", expected_sources=["ownership.md"])]
    eval_run = _eval_run(
        queries,
        {"q1": [_search_result("1", "ownership rules", "ownership.md")]},
        top_k=1,
    )

    with pytest.raises(LlmEvalError, match="LLM_API_KEY"):
        run_document_relevance_eval(eval_run, queries, EvalSettings())


@patch("agentic_doc_rag.evaluation.phoenix.LLM")
@patch(
    "agentic_doc_rag.evaluation.phoenix.async_evaluate_dataframe", new_callable=lambda: MagicMock()
)
def test_run_document_relevance_eval_uses_phoenix_evaluator(
    mock_evaluate: MagicMock,
    mock_llm_cls: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.delenv("EVAL_LLM_MODEL", raising=False)
    queries = [EvalQuery(id="q1", query="ownership", expected_sources=["ownership.md"])]
    eval_run = _eval_run(
        queries,
        {"q1": [_search_result("1", "ownership rules", "ownership.md")]},
        top_k=1,
    )

    async def fake_async_evaluate_dataframe(**kwargs: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "query_id": ["q1"],
                "input": ["ownership"],
                "document_position": [0],
                "document_relevance_score": [
                    Score(name="document_relevance", score=1.0, label="relevant", explanation="yes")
                ],
            }
        )

    mock_evaluate.side_effect = fake_async_evaluate_dataframe

    _, llm_report = run_document_relevance_eval(eval_run, queries, EvalSettings())

    assert llm_report.precision_at_k == 1.0
    assert llm_report.llm_hit_at_k == 1.0
    mock_evaluate.assert_called_once()
    assert mock_evaluate.call_args.kwargs["concurrency"] == 5
    mock_llm_cls.assert_called_once_with(
        provider="openai",
        model="gpt-4o-mini",
        api_key="test-key",
        base_url="https://openrouter.ai/api/v1",
    )


@patch("phoenix.client.Client")
def test_fetch_retriever_span_ids_uses_project_and_input_value_column(
    mock_client_cls: MagicMock,
) -> None:
    valid_documents = [{"document": {"content": "ownership rules"}}]
    flat_documents = [{"document.id": "chunk-1", "document.content": "legacy format"}]
    spans_df = pd.DataFrame(
        {
            "input.value": [
                "What is ownership in Rust?",
                "What is ownership in Rust?",
                "What is ownership in Rust?",
            ],
            "retrieval.documents": [valid_documents, flat_documents, valid_documents],
        },
        index=["span-new", "span-legacy", "span-old"],
    )
    mock_client_cls.return_value.spans.get_spans_dataframe.return_value = spans_df

    span_ids = _fetch_retriever_span_ids(
        client_url="http://localhost:6006",
        query_texts=["What is ownership in Rust?", "missing query"],
        project_identifier="agentic-doc",
    )

    assert span_ids == {"What is ownership in Rust?": "span-new"}
    call_kwargs = mock_client_cls.return_value.spans.get_spans_dataframe.call_args.kwargs
    assert call_kwargs["project_identifier"] == "agentic-doc"


@patch("agentic_doc_rag.evaluation.phoenix.upload_relevance_annotations", return_value=True)
@patch("agentic_doc_rag.evaluation.phoenix.run_document_relevance_eval")
@patch(
    "agentic_doc_rag.evaluation.phoenix._safe_fetch_retriever_span_ids",
    return_value={"ownership": "span-1"},
)
def test_run_llm_relevance_eval_uploads_annotations_when_enabled(
    mock_fetch_spans: MagicMock,
    mock_run_document_eval: MagicMock,
    mock_upload: MagicMock,
) -> None:
    queries = [EvalQuery(id="q1", query="ownership", expected_sources=["ownership.md"])]
    eval_run = _eval_run(
        queries,
        {"q1": [_search_result("1", "ownership rules", "ownership.md")]},
        top_k=1,
    )
    scored_df = pd.DataFrame(
        {
            "query_id": ["q1"],
            "input": ["ownership"],
            "document_position": [0],
            "document_relevance_score": [
                Score(name="document_relevance", score=1.0, label="relevant")
            ],
        }
    )
    mock_run_document_eval.return_value = (
        scored_df,
        compute_llm_retrieval_metrics(scored_df, top_k=1, model="gpt-4o-mini"),
    )

    llm_report = run_llm_relevance_eval(
        eval_run,
        queries,
        EvalSettings(),
        upload_annotations=True,
    )

    mock_upload.assert_called_once()
    mock_fetch_spans.assert_called_once()
    assert llm_report.annotations_uploaded is True
    assert llm_report.scores[0].span_id == "span-1"


@patch("agentic_doc_rag.evaluation.phoenix._safe_fetch_retriever_span_ids")
@patch("agentic_doc_rag.evaluation.phoenix.run_document_relevance_eval")
def test_run_llm_relevance_eval_skips_phoenix_when_upload_disabled(
    mock_run_document_eval: MagicMock,
    mock_safe_fetch_spans: MagicMock,
) -> None:
    queries = [EvalQuery(id="q1", query="ownership", expected_sources=["ownership.md"])]
    eval_run = _eval_run(
        queries,
        {"q1": [_search_result("1", "ownership rules", "ownership.md")]},
        top_k=1,
    )
    scored_df = pd.DataFrame(
        {
            "query_id": ["q1"],
            "input": ["ownership"],
            "document_position": [0],
            "document_relevance_score": [
                Score(name="document_relevance", score=1.0, label="relevant")
            ],
        }
    )
    mock_run_document_eval.return_value = (
        scored_df,
        compute_llm_retrieval_metrics(scored_df, top_k=1, model="gpt-4o-mini"),
    )

    llm_report = run_llm_relevance_eval(
        eval_run,
        queries,
        EvalSettings(),
        upload_annotations=False,
    )

    mock_safe_fetch_spans.assert_not_called()
    assert llm_report.annotations_uploaded is False


@patch("agentic_doc_rag.evaluation.phoenix.upload_relevance_annotations", return_value=False)
@patch("agentic_doc_rag.evaluation.phoenix.run_document_relevance_eval")
@patch(
    "agentic_doc_rag.evaluation.phoenix._fetch_retriever_span_ids",
    side_effect=httpx.ConnectError("refused"),
)
def test_run_llm_relevance_eval_survives_unreachable_phoenix(
    mock_fetch_spans: MagicMock,
    mock_run_document_eval: MagicMock,
    mock_upload: MagicMock,
) -> None:
    queries = [EvalQuery(id="q1", query="ownership", expected_sources=["ownership.md"])]
    eval_run = _eval_run(
        queries,
        {"q1": [_search_result("1", "ownership rules", "ownership.md")]},
        top_k=1,
    )
    scored_df = pd.DataFrame(
        {
            "query_id": ["q1"],
            "input": ["ownership"],
            "document_position": [0],
            "document_relevance_score": [
                Score(name="document_relevance", score=1.0, label="relevant")
            ],
        }
    )
    mock_run_document_eval.return_value = (
        scored_df,
        compute_llm_retrieval_metrics(scored_df, top_k=1, model="gpt-4o-mini"),
    )

    llm_report = run_llm_relevance_eval(
        eval_run,
        queries,
        EvalSettings(),
        upload_annotations=True,
    )

    assert llm_report.precision_at_k == 1.0
    assert llm_report.annotations_uploaded is False
    mock_upload.assert_called_once()


@patch("phoenix.client.Client")
def test_upload_relevance_annotations_returns_false_on_http_error(
    mock_client_cls: MagicMock,
) -> None:
    scored_df = pd.DataFrame(
        {
            "input": ["ownership"],
            "document_position": [0],
            "document_relevance_score": [
                Score(name="document_relevance", score=1.0, label="relevant")
            ],
        }
    )
    mock_client_cls.return_value.spans.log_document_annotations_dataframe.side_effect = (
        httpx.HTTPStatusError(
            "422",
            request=httpx.Request("POST", "http://localhost:6006/v1/document_annotations"),
            response=httpx.Response(422),
        )
    )

    uploaded = upload_relevance_annotations(
        scored_df,
        settings=EvalSettings(),
        span_ids_by_query={"ownership": "span-123"},
    )

    assert uploaded is False


@patch("phoenix.client.Client")
def test_upload_relevance_annotations_logs_document_scores(mock_client_cls: MagicMock) -> None:
    scored_df = pd.DataFrame(
        {
            "input": ["ownership"],
            "document_position": [0],
            "document_relevance_score": [
                Score(name="document_relevance", score=1.0, label="relevant", explanation="match")
            ],
        }
    )
    mock_client = mock_client_cls.return_value

    uploaded = upload_relevance_annotations(
        scored_df,
        settings=EvalSettings(),
        span_ids_by_query={"ownership": "span-123"},
    )

    assert uploaded is True
    mock_client.spans.log_document_annotations_dataframe.assert_called_once()
    call_kwargs = mock_client.spans.log_document_annotations_dataframe.call_args.kwargs
    assert call_kwargs["annotation_name"] == "relevance"
    assert call_kwargs["annotator_kind"] == "LLM"
