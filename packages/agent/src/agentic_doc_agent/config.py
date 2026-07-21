"""Settings for the agentic intelligence layer."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from agentic_doc_core.config import get_core_settings


class AgentSettings(BaseSettings):
    """Environment-backed configuration for agent workflows and LLM calls."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Default chat model id (provider-dependent; OpenRouter-style names ok)",
    )
    llm_temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Sampling temperature for generation",
    )
    max_tool_rounds: int = Field(
        default=5,
        ge=1,
        description="Maximum tool/LLM rounds per workflow run",
    )
    default_top_k: int = Field(
        default=5,
        ge=1,
        description="Default retrieval top-k when a request does not override it",
    )
    faithfulness_enabled: bool = Field(
        default=True,
        description="Score groundedness of the final answer against retrieved context",
    )

    @property
    def llm_api_key(self) -> str | None:
        return get_core_settings().llm_api_key

    @property
    def llm_base_url(self) -> str | None:
        return get_core_settings().llm_base_url


def get_agent_settings() -> AgentSettings:
    return AgentSettings()
