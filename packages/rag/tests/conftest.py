from pathlib import Path

import pytest
from support.fakes import PassthroughReranker
from support.paths import CORPUS_DIR, EVAL_DATASET_PATH, FIXTURES_DIR


@pytest.fixture
def fixtures_dir() -> Path:
    return FIXTURES_DIR


@pytest.fixture
def corpus_dir() -> Path:
    return CORPUS_DIR


@pytest.fixture
def eval_dataset_path() -> Path:
    return EVAL_DATASET_PATH


@pytest.fixture
def passthrough_reranker() -> PassthroughReranker:
    return PassthroughReranker()
