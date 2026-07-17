from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"
PDF_FIXTURES_DIR = FIXTURES_DIR / "pdf"
SAMPLE_PDF_PATH = PDF_FIXTURES_DIR / "sample.pdf"
EMPTY_PAGE_PDF_PATH = PDF_FIXTURES_DIR / "with_empty_page.pdf"
CODE_FIXTURES_DIR = FIXTURES_DIR / "code"
SAMPLE_RUST_PATH = CODE_FIXTURES_DIR / "sample.rs"
SAMPLE_PYTHON_PATH = CODE_FIXTURES_DIR / "sample.py"
EVAL_DATASET_PATH = FIXTURES_DIR / "eval_dataset.jsonl"