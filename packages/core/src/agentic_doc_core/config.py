from enum import StrEnum

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    DEV = "dev"
    PROD = "prod"


class CoreSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    environment: Environment = Field(default=Environment.DEV, description="Environment to run in")
    log_level: str = Field(default="INFO", description="Log level")


def get_core_settings() -> CoreSettings:
    return CoreSettings()
