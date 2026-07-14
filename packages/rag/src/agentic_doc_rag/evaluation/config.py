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
    llm_provider: str = Field(default="openai", description="LLM provider for --llm eval mode")
    llm_model: str = Field(default="gpt-4o-mini", description="LLM model for --llm eval mode")
    llm_concurrency: int = Field(default=5, description="Concurrent LLM eval requests")
    phoenix_client_url: str = Field(
        default="http://localhost:6006",
        description="Phoenix HTTP API used for annotation upload",
    )
    report_dir: Path = Field(
        default=Path("data/eval/reports"),
        description="Directory where eval JSON reports are saved",
    )


def get_eval_settings() -> EvalSettings:
    return EvalSettings()
