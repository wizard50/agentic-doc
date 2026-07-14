from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEV = "dev"
    PROD = "prod"


class PhoenixSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="PHOENIX_",
    )

    enabled: bool = Field(default=False, description="Enable Phoenix tracing")
    collector_endpoint: str = Field(
        default="http://localhost:4317",
        description="Phoenix OTLP collector endpoint",
    )
    project_name: str = Field(
        default="agentic-doc",
        description="Phoenix project name",
    )


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Environment = Field(default=Environment.DEV, description="Environment to run in")
    log_level: str = Field(default="INFO", description="Log level")
    llm_api_key: str | None = Field(
        default=None,
        description="API key for LLM providers (OpenAI, OpenRouter, etc.)",
    )
    llm_base_url: str | None = Field(
        default=None,
        description="OpenAI-compatible API base URL (e.g. OpenRouter, Azure, local proxy)",
    )

    @property
    def phoenix(self) -> PhoenixSettings:
        return get_phoenix_settings()


def get_phoenix_settings() -> PhoenixSettings:
    return PhoenixSettings()


def get_core_settings() -> CoreSettings:
    return CoreSettings()
