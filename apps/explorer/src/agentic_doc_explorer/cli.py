import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from agentic_doc_core.config import get_phoenix_settings
from agentic_doc_explorer.workspace import require_workspace_root
from agentic_doc_rag.config import get_rag_settings
from agentic_doc_rag.evaluation import (
    EmptyVectorStoreError,
    LlmEvalError,
    build_eval_payload,
    format_eval_summary,
    format_llm_eval_summary,
    get_eval_settings,
    load_eval_dataset,
    run_llm_relevance_eval,
    run_retrieval_eval,
    save_eval_report,
)
from agentic_doc_rag.ingest import (
    IngestSourceNotFoundError,
    resolve_ingest_settings,
    run_ingestion,
)
from agentic_doc_rag.observability import register_tracing
from agentic_doc_rag.retrieval import SearchMode, create_retriever
from agentic_doc_rag.sparse import create_sparse_index
from agentic_doc_rag.vectorstore.factory import create_vector_store


def _run_ingest(args: argparse.Namespace) -> None:
    require_workspace_root("ingest")
    rag_settings = get_rag_settings()
    vectorstore = create_vector_store(rag_settings)
    sparse_index = create_sparse_index(rag_settings)
    ingest_settings = resolve_ingest_settings(
        rag_settings,
        source_dir=args.source,
        skip_files=frozenset(args.skip) if args.skip is not None else None,
    )
    try:
        result = run_ingestion(vectorstore, sparse_index, ingest_settings)
    except IngestSourceNotFoundError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    document_count = vectorstore.count()
    sparse_count = sparse_index.count()
    details = {
        "Vector store": rag_settings.vector_store_type.value,
        "Collection": rag_settings.chroma_collection_name,
        "Persist dir": str(rag_settings.chroma_persist_dir),
        "BM25 dir": str(rag_settings.bm25_persist_dir),
        "Source": str(ingest_settings.source_dir),
        "Chunks (vector)": str(document_count),
        "Chunks (BM25)": str(sparse_count),
        "Files indexed": str(result.file_count),
    }
    label_width = max(len(label) for label in details)

    print("explorer ingest — Agentic Doc indexing")
    print("─" * 40)
    for label, value in details.items():
        print(f"  {label:<{label_width}}  {value}")

    if document_count == 0:
        print(
            f"\n  No documents indexed. Check that the source directory exists:\n  {ingest_settings.source_dir.resolve()}"
        )


def _run_ui() -> None:
    require_workspace_root("explorer")
    app_path = Path(__file__).parent / "app.py"
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", str(app_path)],
        check=True,
    )


def _run_eval(args: argparse.Namespace) -> None:
    require_workspace_root("eval")
    rag_settings = get_rag_settings()
    eval_settings = get_eval_settings()
    dataset_path = Path(args.dataset) if args.dataset else eval_settings.dataset_path
    top_k = args.top_k if args.top_k is not None else eval_settings.top_k
    fail_under = (
        args.fail_under if args.fail_under is not None else eval_settings.fail_under_hit_at_k
    )
    report_dir = Path(args.report_dir) if args.report_dir else eval_settings.report_dir
    save_report = not args.no_save
    search_mode = (
        SearchMode(args.search_mode) if args.search_mode is not None else rag_settings.search_mode
    )
    candidate_k = rag_settings.candidate_k
    rerank = True if args.rerank else None

    queries = load_eval_dataset(dataset_path)
    retriever = create_retriever(rag_settings)

    try:
        eval_run = run_retrieval_eval(
            retriever,
            queries,
            top_k=top_k,
            dataset_name=dataset_path.name,
            search_mode=search_mode,
            candidate_k=candidate_k,
            rerank=rerank,
        )
    except EmptyVectorStoreError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    report = eval_run.report
    llm_report = None
    phoenix_settings = get_phoenix_settings()
    if args.llm:
        try:
            llm_report = run_llm_relevance_eval(
                eval_run,
                queries,
                eval_settings,
                upload_annotations=phoenix_settings.enabled,
            )
        except LlmEvalError as exc:
            print(exc, file=sys.stderr)
            sys.exit(1)
        if (
            phoenix_settings.enabled
            and llm_report is not None
            and not llm_report.annotations_uploaded
        ):
            print(
                "Phoenix relevance annotations were not uploaded "
                "(no matching retriever spans or upload failed).",
                file=sys.stderr,
            )

    payload = build_eval_payload(
        report,
        llm_report=llm_report,
        metadata={
            "run_at": datetime.now(UTC).isoformat(),
            "dataset": str(dataset_path),
            "collection": rag_settings.chroma_collection_name,
            "chunk_count": retriever.count(),
            "top_k": top_k,
            "candidate_k": candidate_k,
            "search_mode": search_mode.value,
            "rerank": rerank if rerank is not None else rag_settings.rerank_enabled,
            "llm_enabled": llm_report is not None,
        },
    )

    saved_report_path: Path | None = None
    if save_report:
        saved_report_path = save_eval_report(payload, report_dir)

    if args.output == "json":
        print(json.dumps(payload, indent=2))
    else:
        summary = format_eval_summary(
            report,
            dataset_path=str(dataset_path),
            collection_name=rag_settings.chroma_collection_name,
            chunk_count=retriever.count(),
        )
        if llm_report is not None:
            summary += format_llm_eval_summary(
                llm_report,
                top_k=top_k,
                upload_attempted=phoenix_settings.enabled,
            )
        if saved_report_path is not None:
            summary += f"\n\n  Report saved to  {saved_report_path}"
        print(summary)

    if fail_under is not None and report.hit_at_k < fail_under:
        print(
            f"\n  hit@{top_k} {report.hit_at_k:.4f} is below threshold {fail_under:.4f}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    register_tracing(get_phoenix_settings())

    parser = argparse.ArgumentParser(
        description="Agentic Doc explorer — search UI and ingestion CLI."
    )
    subparsers = parser.add_subparsers(dest="command")

    ingest_parser = subparsers.add_parser("ingest", help="Index markdown files into the vector store")
    ingest_parser.add_argument(
        "--source",
        type=Path,
        help="Markdown root directory (default: INGEST_SOURCE_DIR)",
    )
    ingest_parser.add_argument(
        "--skip",
        action="append",
        default=None,
        metavar="FILE",
        help="Skip filename (repeatable; replaces INGEST_SKIP_FILES when set)",
    )

    eval_parser = subparsers.add_parser("eval", help="Run retrieval evaluation against the index")
    eval_parser.add_argument("--dataset", help="Path to golden dataset JSONL")
    eval_parser.add_argument("--top-k", type=int, help="Number of results to retrieve per query")
    eval_parser.add_argument(
        "--search-mode",
        choices=[mode.value for mode in SearchMode],
        help="Retrieval mode (default: SEARCH_MODE from settings)",
    )
    eval_parser.add_argument(
        "--rerank",
        action="store_true",
        help="Enable cross-encoder reranking for this eval run",
    )
    eval_parser.add_argument(
        "--output",
        choices=["text", "json"],
        default="text",
        help="Report format (default: text)",
    )
    eval_parser.add_argument(
        "--fail-under",
        type=float,
        help="Exit with code 1 when hit@k is below this threshold",
    )
    eval_parser.add_argument(
        "--llm",
        action="store_true",
        help="Run Phoenix DocumentRelevanceEvaluator (requires LLM_API_KEY)",
    )
    eval_parser.add_argument(
        "--report-dir",
        help="Directory to save the eval JSON report (default: EVAL_REPORT_DIR)",
    )
    eval_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip writing the eval report to disk",
    )

    subparsers.add_parser("ui", help="Launch the Streamlit search UI")

    args = parser.parse_args()
    match args.command:
        case "ingest":
            _run_ingest(args)
        case "eval":
            _run_eval(args)
        case "ui" | None:
            _run_ui()
        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
