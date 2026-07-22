"""OpenAI-compatible chat client and factory."""

from __future__ import annotations

from typing import Any, TypeVar, cast

from openai import APIError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, ValidationError

from agentic_doc_agent.config import AgentSettings, get_agent_settings
from agentic_doc_agent.llm.models import (
    ChatMessage,
    ChatResult,
    LlmConfigError,
    LlmRequestError,
    LlmResponseError,
    TokenUsage,
)

T = TypeVar("T", bound=BaseModel)


class OpenAICompatibleClient:
    """Chat completions via the OpenAI SDK (OpenRouter and other compatible APIs)."""

    def __init__(
        self,
        *,
        api_key: str,
        default_model: str,
        default_temperature: float = 0.0,
        base_url: str | None = None,
        client: OpenAI | None = None,
    ) -> None:
        if not api_key:
            raise LlmConfigError("api_key must be a non-empty string")
        if not default_model:
            raise LlmConfigError("default_model must be a non-empty string")
        self._default_model = default_model
        self._default_temperature = default_temperature
        self._client = client or OpenAI(api_key=api_key, base_url=base_url)

    def complete(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> ChatResult:
        response, resolved_model = self._create_completion(
            messages,
            model=model,
            temperature=temperature,
        )
        content = _assistant_content_from_response(response)
        usage = _token_usage_from_response(response)
        response_model = getattr(response, "model", None) or resolved_model
        return ChatResult(content=content, model=str(response_model), usage=usage)

    def complete_structured(
        self,
        messages: list[ChatMessage],
        schema: type[T],
        *,
        model: str | None = None,
        temperature: float | None = None,
    ) -> T:
        """Run a chat completion and parse the assistant content as ``schema``."""
        if not issubclass(schema, BaseModel):
            raise LlmConfigError("schema must be a pydantic BaseModel subclass")

        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": _schema_name(schema),
                "schema": schema.model_json_schema(),
                "strict": False,
            },
        }
        response, _resolved_model = self._create_completion(
            messages,
            model=model,
            temperature=temperature,
            response_format=response_format,
        )
        content = _assistant_content_from_response(response)
        try:
            return schema.model_validate_json(content)
        except ValidationError as exc:
            raise LlmResponseError(
                f"LLM response failed schema validation for {schema.__name__}: {exc}"
            ) from exc

    def _create_completion(
        self,
        messages: list[ChatMessage],
        *,
        model: str | None,
        temperature: float | None,
        response_format: dict[str, Any] | None = None,
    ) -> tuple[Any, str]:
        if not messages:
            raise LlmConfigError("messages must be a non-empty list")

        resolved_model = model if model is not None else self._default_model
        resolved_temperature = (
            temperature if temperature is not None else self._default_temperature
        )
        payload = _messages_payload(messages)
        create_kwargs: dict[str, Any] = {
            "model": resolved_model,
            "messages": payload,
            "temperature": resolved_temperature,
        }
        if response_format is not None:
            create_kwargs["response_format"] = response_format

        try:
            response = self._client.chat.completions.create(**create_kwargs)
        except APIError as exc:
            raise LlmRequestError(f"LLM request failed: {exc}") from exc
        except Exception as exc:
            raise LlmRequestError(f"LLM request failed: {exc}") from exc

        return response, resolved_model


def create_llm_client(
    settings: AgentSettings | None = None,
    *,
    client: OpenAI | None = None,
) -> OpenAICompatibleClient:
    """Build an LLM client from agent settings (and core credentials)."""
    resolved = settings or get_agent_settings()
    api_key = resolved.llm_api_key
    if not api_key:
        raise LlmConfigError(
            "LLM_API_KEY is required to create an LLM client. "
            "Set it in the environment or .env file."
        )
    return OpenAICompatibleClient(
        api_key=api_key,
        base_url=resolved.llm_base_url,
        default_model=resolved.llm_model,
        default_temperature=resolved.llm_temperature,
        client=client,
    )


def _messages_payload(messages: list[ChatMessage]) -> list[ChatCompletionMessageParam]:
    return [
        cast(
            ChatCompletionMessageParam,
            {"role": message.role.value, "content": message.content},
        )
        for message in messages
    ]


def _schema_name(schema: type[BaseModel]) -> str:
    name = schema.__name__
    # OpenAI json_schema names: a-z, A-Z, 0-9, underscores, dashes; max 64.
    cleaned = "".join(ch if ch.isalnum() or ch in "_-" else "_" for ch in name)
    return (cleaned or "response")[:64]


def _assistant_content_from_response(response: Any) -> str:
    choices = getattr(response, "choices", None) or []
    if not choices:
        raise LlmResponseError("LLM response contained no choices")

    message = getattr(choices[0], "message", None)
    content = getattr(message, "content", None) if message is not None else None
    if content is None or not str(content).strip():
        raise LlmResponseError("LLM response contained empty assistant content")
    return str(content)


def _token_usage_from_response(response: Any) -> TokenUsage | None:
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    return TokenUsage(
        prompt_tokens=getattr(usage, "prompt_tokens", None),
        completion_tokens=getattr(usage, "completion_tokens", None),
        total_tokens=getattr(usage, "total_tokens", None),
    )
