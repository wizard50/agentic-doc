from agentic_doc_rag.config import RagSettings
from agentic_doc_rag.ingest import ingest_settings_from_rag

_DEFAULT_INGEST = ingest_settings_from_rag(RagSettings())

RUST_BOOK_SRC = _DEFAULT_INGEST.source_dir
RUST_BOOK_SKIP = _DEFAULT_INGEST.skip_files

DEFAULT_TOP_K = 5
PREVIEW_LENGTH = 400
PHOENIX_UI_URL = "http://localhost:6006"

EXAMPLE_QUERIES = (
    "What is ownership in Rust?",
    "How do structs work?",
    "What is borrowing?",
    "How does match work?",
    "What are traits?",
)