from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent.parent
FIXTURES_DIR = TESTS_DIR / "fixtures"
CORPUS_DIR = FIXTURES_DIR / "corpus"
EVAL_DATASET_PATH = FIXTURES_DIR / "eval_dataset.jsonl"