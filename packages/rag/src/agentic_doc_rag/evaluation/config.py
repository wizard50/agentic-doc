from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_doc_rag.evaluation.dataset import DEFAULT_DATASET_PATH


class EvalSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="EVAL_",
    )

    top_k: int = Field(default=5, description="Number of retrieval results to evaluate")
    dataset_path: Path = Field(
        default=DEFAULT_DATASET_PATH,
        description="Path to the golden evaluation dataset (JSONL)",
    )
    fail_under_hit_at_k: float | None = Field(
        default=None,
        description="Exit with failure when hit@k falls below this threshold",
    )


def get_eval_settings() -> EvalSettings:
    return EvalSettings()