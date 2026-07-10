from agentic_doc_explorer.constants import EXAMPLE_QUERIES, PHOENIX_UI_URL, RUST_BOOK_SKIP


def test_example_queries_defined() -> None:
    assert len(EXAMPLE_QUERIES) >= 3


def test_rust_book_skip_files() -> None:
    assert "SUMMARY.md" in RUST_BOOK_SKIP


def test_phoenix_ui_url() -> None:
    assert PHOENIX_UI_URL == "http://localhost:6006"
