from pathlib import Path

RUST_BOOK_SRC = Path("data/download/rust-book/src")
RUST_BOOK_SKIP = frozenset({"SUMMARY.md", "title-page.md"})

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
