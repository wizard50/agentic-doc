import asyncio
import json
from typing import Any

import httpx
import pandas as pd
from phoenix.evals import LLM, Score, async_evaluate_dataframe
from phoenix.evals.metrics import DocumentRelevanceEvaluator

from agentic_doc_core.config import get_core_settings
from agentic_doc_rag.evaluation.config import EvalSettings
from agentic_doc_rag.evaluation.models import EvalQuery, LlmEvalReport, LlmRelevanceScore
from agentic_doc_rag.evaluation.runner import RetrievalEvalRun
from agentic_doc_rag.evaluation.utils import mean
from agentic_doc_rag.models import SearchResult

DOCUMENT_RELEVANCE_SCORE_COLUMN = "document_relevance_score"


class LlmEvalError(Exception):
    """Raised when LLM-based evaluation cannot run."""


def _require_llm_api_key() -> str:
    api_key = get_core_settings().llm_api_key
    if not api_key:
        msg = "LLM_API_KEY is required for --llm eval mode"
        raise LlmEvalError(msg)
    return api_key


def _create_llm(settings: EvalSettings) -> LLM:
    core_settings = get_core_settings()
    llm_kwargs: dict[str, Any] = {
        "provider": settings.llm_provider,
        "model": settings.llm_model,
        "api_key": _require_llm_api_key(),
    }
    if core_settings.llm_base_url:
        llm_kwargs["base_url"] = core_settings.llm_base_url
    return LLM(**llm_kwargs)


def build_document_eval_dataframe(
    queries: list[EvalQuery],
    results_by_query_id: dict[str, list[SearchResult]],
) -> pd.DataFrame:
    """Build one row per retrieved document for Phoenix relevance evaluation."""
    rows: list[dict[str, Any]] = []
    for query in queries:
        for position, result in enumerate(results_by_query_id.get(query.id, [])):
            rows.append(
                {
                    "query_id": query.id,
                    "input": query.query,
                    "document_text": result.chunk.text,
                    "document_position": position,
                }
            )
    return pd.DataFrame(rows)


def _parse_score(value: Any) -> Score:
    if isinstance(value, Score):
        return value
    if isinstance(value, dict):
        return Score(**value)
    if isinstance(value, str):
        payload = json.loads(value)
        if isinstance(payload, dict):
            return Score(**payload)
    msg = f"Unsupported score value: {value!r}"
    raise ValueError(msg)


def _score_value(score: Score) -> float:
    raw = score.score
    if isinstance(raw, (int, float)):
        return float(raw)
    return 0.0


def _as_scalar_float(value: object) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    item_method = getattr(value, "item", None)
    if callable(item_method):
        item = item_method()
        if isinstance(item, (bool, int, float)):
            return float(item)
    msg = f"Expected numeric scalar, got {type(value)!r}"
    raise TypeError(msg)


def compute_llm_retrieval_metrics(
    scored_df: pd.DataFrame,
    *,
    top_k: int,
    model: str,
) -> LlmEvalReport:
    """Aggregate precision@k and llm_hit@k from document relevance scores."""
    if scored_df.empty:
        return LlmEvalReport(model=model, precision_at_k=0.0, llm_hit_at_k=0.0)

    working_df = scored_df.copy()
    working_df["parsed_score"] = working_df[DOCUMENT_RELEVANCE_SCORE_COLUMN].map(_parse_score)
    working_df["relevance_score"] = working_df["parsed_score"].map(_score_value)

    per_query_precision: list[float] = []
    per_query_hit: list[float] = []

    for _, group in working_df.groupby("query_id"):
        top_group = group.sort_values("document_position").head(top_k)
        per_query_precision.append(_as_scalar_float(top_group["relevance_score"].mean()))
        per_query_hit.append(_as_scalar_float((top_group["relevance_score"] > 0).any()))

    scores = [
        LlmRelevanceScore(
            query_id=row["query_id"],
            query=row["input"],
            document_position=int(row["document_position"]),
            score=_score_value(parsed),
            label=str(parsed.label or ""),
            explanation=parsed.explanation,
        )
        for row, parsed in zip(
            working_df.to_dict(orient="records"),
            working_df["parsed_score"],
            strict=True,
        )
    ]

    return LlmEvalReport(
        model=model,
        precision_at_k=mean(per_query_precision),
        llm_hit_at_k=mean(per_query_hit),
        scores=scores,
    )


def run_document_relevance_eval(
    eval_run: RetrievalEvalRun,
    queries: list[EvalQuery],
    settings: EvalSettings,
) -> tuple[pd.DataFrame, LlmEvalReport]:
    """Score retrieved documents with Phoenix DocumentRelevanceEvaluator."""
    document_df = build_document_eval_dataframe(queries, eval_run.results_by_query_id)
    if document_df.empty:
        return document_df, LlmEvalReport(
            model=settings.llm_model,
            precision_at_k=0.0,
            llm_hit_at_k=0.0,
        )

    llm = _create_llm(settings)
    relevance_evaluator = DocumentRelevanceEvaluator(llm=llm)
    scored_df = asyncio.run(
        async_evaluate_dataframe(
            dataframe=document_df,
            evaluators=[relevance_evaluator],
            concurrency=settings.llm_concurrency,
            hide_tqdm_bar=True,
        )
    )
    llm_report = compute_llm_retrieval_metrics(
        scored_df,
        top_k=eval_run.report.top_k,
        model=settings.llm_model,
    )
    return scored_df, llm_report


def _input_value_column(spans_df: pd.DataFrame) -> str | None:
    for column in ("input.value", "attributes.input.value"):
        if column in spans_df.columns:
            return column
    return None


def _span_has_annotatable_documents(row: pd.Series) -> bool:
    documents = row.get("retrieval.documents")
    if not isinstance(documents, list) or not documents:
        return False
    first = documents[0]
    return isinstance(first, dict) and "document" in first


def _fetch_retriever_span_ids(
    *,
    client_url: str,
    query_texts: list[str],
    project_identifier: str,
) -> dict[str, str]:
    from phoenix.client import Client
    from phoenix.client.types.spans import SpanQuery

    client = Client(base_url=client_url)
    spans_df = client.spans.get_spans_dataframe(
        query=SpanQuery()
        .where("span_kind == 'RETRIEVER'")
        .where("name == 'vectorstore.search'")
        .select("input.value", "retrieval.documents"),
        project_identifier=project_identifier,
    )
    if spans_df.empty:
        return {}

    input_column = _input_value_column(spans_df)
    if input_column is None:
        return {}

    span_ids_by_query: dict[str, str] = {}
    for span_id, row in spans_df.iterrows():
        query_text = row.get(input_column)
        if not isinstance(query_text, str) or query_text in span_ids_by_query:
            continue
        if not _span_has_annotatable_documents(row):
            continue
        span_ids_by_query[query_text] = str(span_id)
    return {query: span_ids_by_query[query] for query in query_texts if query in span_ids_by_query}


def _safe_fetch_retriever_span_ids(
    *,
    client_url: str,
    query_texts: list[str],
    project_identifier: str,
) -> dict[str, str]:
    """Fetch retriever span IDs, returning an empty map when Phoenix is unreachable."""
    try:
        return _fetch_retriever_span_ids(
            client_url=client_url,
            query_texts=query_texts,
            project_identifier=project_identifier,
        )
    except httpx.HTTPError:
        return {}


def upload_relevance_annotations(
    scored_df: pd.DataFrame,
    *,
    settings: EvalSettings,
    span_ids_by_query: dict[str, str],
) -> bool:
    """Upload LLM relevance scores to Phoenix as document annotations."""
    if scored_df.empty or DOCUMENT_RELEVANCE_SCORE_COLUMN not in scored_df.columns:
        return False

    from phoenix.client import Client

    if not span_ids_by_query:
        return False

    annotation_rows: list[dict[str, Any]] = []
    for row in scored_df.to_dict(orient="records"):
        query_text = row["input"]
        span_id = span_ids_by_query.get(query_text)
        if span_id is None:
            continue

        parsed = _parse_score(row[DOCUMENT_RELEVANCE_SCORE_COLUMN])
        annotation_rows.append(
            {
                "span_id": span_id,
                "document_position": int(row["document_position"]),
                "label": parsed.label,
                "score": _score_value(parsed),
                "explanation": parsed.explanation,
            }
        )

    if not annotation_rows:
        return False

    client = Client(base_url=settings.phoenix_client_url)
    try:
        client.spans.log_document_annotations_dataframe(
            dataframe=pd.DataFrame(annotation_rows),
            annotation_name="relevance",
            annotator_kind="LLM",
            sync=True,
        )
    except httpx.HTTPError:
        return False
    return True


def run_llm_relevance_eval(
    eval_run: RetrievalEvalRun,
    queries: list[EvalQuery],
    settings: EvalSettings,
    *,
    upload_annotations: bool,
) -> LlmEvalReport:
    """Run LLM relevance scoring and optionally upload annotations to Phoenix."""
    scored_df, llm_report = run_document_relevance_eval(eval_run, queries, settings)

    if not upload_annotations:
        return llm_report

    phoenix_project = get_core_settings().phoenix.project_name
    span_ids_by_query = _safe_fetch_retriever_span_ids(
        client_url=settings.phoenix_client_url,
        query_texts=[query.query for query in queries],
        project_identifier=phoenix_project,
    )
    if span_ids_by_query:
        updated_scores = [
            score.model_copy(update={"span_id": span_ids_by_query.get(score.query)})
            for score in llm_report.scores
        ]
        llm_report = llm_report.model_copy(update={"scores": updated_scores})

    uploaded = upload_relevance_annotations(
        scored_df,
        settings=settings,
        span_ids_by_query=span_ids_by_query,
    )
    return llm_report.model_copy(update={"annotations_uploaded": uploaded})
