from agentic_doc_rag.evaluation.models import EvalReport, LlmEvalReport


def _tag_hit_count(report: EvalReport, tag: str) -> int:
    return sum(1 for result in report.results if tag in result.tags and result.hit_at_k)


def format_eval_summary(
    report: EvalReport,
    *,
    dataset_path: str,
    collection_name: str,
    chunk_count: int,
) -> str:
    """Format a human-readable retrieval evaluation summary."""
    lines = [
        "explorer eval — retrieval benchmark",
        "─" * 40,
        f"  {'Dataset':<14}  {dataset_path} ({report.query_count} queries)",
        f"  {'Collection':<14}  {collection_name}",
        f"  {'Chunks':<14}  {chunk_count}",
        f"  {'top_k':<14}  {report.top_k}",
        "",
        f"  {'hit@' + str(report.top_k):<14}  {report.hit_at_k:.4f}",
        f"  {'mrr':<14}  {report.mrr:.4f}",
        f"  {'recall@' + str(report.top_k):<14}  {report.recall_at_k:.4f}",
        f"  {'source@' + str(report.top_k):<14}  {report.source_match_at_k:.4f}",
        f"  {'section@' + str(report.top_k):<14}  {report.section_match_at_k:.4f}",
    ]

    if report.by_tag:
        lines.append("")
        lines.append("  By tag:")
        for metric in report.by_tag:
            hits = _tag_hit_count(report, metric.tag)
            lines.append(
                f"    {metric.tag:<16}  hit@{report.top_k}={metric.hit_at_k:.2f}  ({hits}/{metric.query_count})"
            )

    return "\n".join(lines)


def format_llm_eval_summary(
    llm_report: LlmEvalReport,
    *,
    top_k: int,
    upload_attempted: bool = False,
) -> str:
    """Format LLM relevance evaluation metrics."""
    lines = [
        "",
        "  LLM relevance (DocumentRelevanceEvaluator)",
        f"  {'model':<14}  {llm_report.model}",
        f"  {'precision@' + str(top_k):<14}  {llm_report.precision_at_k:.4f}",
        f"  {'llm_hit@' + str(top_k):<14}  {llm_report.llm_hit_at_k:.4f}",
    ]
    if llm_report.annotations_uploaded:
        lines.append("  annotations     uploaded to Phoenix (relevance)")
    elif upload_attempted:
        lines.append("  annotations     not uploaded (see stderr)")
    return "\n".join(lines)
