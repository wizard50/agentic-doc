"""LLM message types, results, and errors."""

from enum import StrEnum

from pydantic import BaseModel, Field


class ChatRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


class ChatMessage(BaseModel):
    """One chat message sent to or returned from the model."""

    role: ChatRole
    content: str = Field(..., min_length=1)


class TokenUsage(BaseModel):
    """Token counts when the provider reports them."""

    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None


class ChatResult(BaseModel):
    """Successful chat completion."""

    content: str
    model: str
    usage: TokenUsage | None = None


class LlmError(Exception):
    """Base error for LLM client failures."""


class LlmConfigError(LlmError):
    """Missing or invalid client configuration (e.g. no API key)."""


class LlmRequestError(LlmError):
    """Provider/HTTP failure while calling the model."""


class LlmResponseError(LlmError):
    """Provider returned an unusable completion (empty body, bad shape)."""
