import argparse
import subprocess
import sys
from pathlib import Path

from agentic_doc_core.config import get_phoenix_settings
from agentic_doc_explorer.constants import RUST_BOOK_SKIP, RUST_BOOK_SRC
from agentic_doc_explorer.pipeline import run_ingestion
from agentic_doc_explorer.workspace import require_workspace_root
from agentic_doc_rag.config import get_rag_settings
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


def main() -> None:
    register_tracing(get_phoenix_settings())

    parser = argparse.ArgumentParser(
        description="Agentic Doc explorer — search UI and ingestion CLI."
    )
    subparsers = parser.add_subparsers(dest="command")

    subparsers.add_parser("ingest", help="Index markdown files into the vector store")
    subparsers.add_parser("ui", help="Launch the Streamlit search UI")

    args = parser.parse_args()
    match args.command:
        case "ingest":
            _run_ingest()
        case "ui" | None:
            _run_ui()
        case _:
            parser.print_help()
            sys.exit(1)


if __name__ == "__main__":
    main()
