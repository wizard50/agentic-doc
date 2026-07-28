"""API process settings (pydantic-settings)."""

from functools import lru_cache
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from agentic_doc_core.config import Environment


class ApiSettings(BaseSettings):
    """Environment-backed configuration for the FastAPI process."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(
        default="Agentic Doc API",
        description="OpenAPI title and process name",
    )
    app_version: str = Field(
        default="0.1.0",
        description="OpenAPI / service version",
    )
    environment: Environment = Field(
        default=Environment.DEV,
        description="Runtime environment (dev | prod)",
    )
    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR)",
    )
    host: str = Field(
        default="0.0.0.0",
        validation_alias="API_HOST",
        description="Bind host for the uvicorn process",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        validation_alias="API_PORT",
        description="Bind port for the uvicorn process",
    )
    # NoDecode: allow comma-separated ALLOWED_ORIGINS instead of JSON lists.
    allowed_origins: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["http://localhost:5173"],
        description="CORS allowed origins (comma-separated in env)",
    )
    docs_enabled: bool = Field(
        default=True,
        validation_alias="API_DOCS_ENABLED",
        description="Expose /docs and /redoc",
    )

    @field_validator("allowed_origins", mode="before")
    @classmethod
    def _parse_origins(cls, value: object) -> object:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",")]
            return [part for part in parts if part]
        return value

    @field_validator("log_level")
    @classmethod
    def _normalize_log_level(cls, value: str) -> str:
        return value.upper()


@lru_cache
def get_api_settings() -> ApiSettings:
    """Return cached API settings (process-wide)."""
    return ApiSettings()
