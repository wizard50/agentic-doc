import argparse
import subprocess
import sys
from pathlib import Path

from agentic_doc_core.config import get_phoenix_settings
from agentic_doc_explorer.constants import RUST_BOOK_SKIP, RUST_BOOK_SRC
from agentic_doc_explorer.pipeline import run_ingestion
from agentic_doc_explorer.workspace import require_workspace_root
from agentic_doc_rag.config import get_rag_settings
from agentic_doc_rag.evaluation import (
    EmptyVectorStoreError,
    format_eval_summary,
    get_eval_settings,
    load_eval_dataset,
    run_retrieval_eval,
)
from agentic_doc_rag.observability import register_tracing
from agentic_doc_rag.vectorstore.factory import create_vector_store


def _run_ingest() -> None:
    require_workspace_root("ingest")
    rag_settings = get_rag_settings()
    vectorstore = create_vector_store(rag_settings)
    run_ingestion(RUST_BOOK_SRC, vectorstore, RUST_BOOK_SKIP)

    document_count = vectorstore.count()
    details = {
        "Vector store": rag_settings.vector_store_type.value,
        "Collection": rag_settings.chroma_collection_name,
        "Persist dir": str(rag_settings.chroma_persist_dir),
        "Documents": str(document_count),
    }
    label_width = max(len(label) for label in details)

    print("explorer ingest — Agentic Doc indexing")
    print("─" * 40)
    for label, value in details.items():
        print(f"  {label:<{label_width}}  {value}")

    if document_count == 0:
        print(
            f"\n  No documents indexed. Check that the source directory exists:\n  {RUST_BOOK_SRC.resolve()}"
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

    queries = load_eval_dataset(dataset_path)
    vectorstore = create_vector_store(rag_settings)

    try:
        report = run_retrieval_eval(
            vectorstore,
            queries,
            top_k=top_k,
            dataset_name=dataset_path.name,
        )
    except EmptyVectorStoreError as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)

    if args.output == "json":
        print(report.model_dump_json(indent=2))
    else:
        print(
            format_eval_summary(
                report,
                dataset_path=str(dataset_path),
                collection_name=rag_settings.chroma_collection_name,
                chunk_count=vectorstore.count(),
            )
        )

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

    subparsers.add_parser("ingest", help="Index markdown files into the vector store")

    eval_parser = subparsers.add_parser("eval", help="Run retrieval evaluation against the index")
    eval_parser.add_argument("--dataset", help="Path to golden dataset JSONL")
    eval_parser.add_argument("--top-k", type=int, help="Number of results to retrieve per query")
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

    subparsers.add_parser("ui", help="Launch the Streamlit search UI")

    args = parser.parse_args()
    match args.command:
        case "ingest":
            _run_ingest()
        case "eval":
            _run_eval(args)
        case "ui" | None:
            _run_ui()
        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
