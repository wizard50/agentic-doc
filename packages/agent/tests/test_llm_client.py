from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from openai import APIError
from pydantic import BaseModel, Field, ValidationError

from agentic_doc_agent.config import AgentSettings
from agentic_doc_agent.llm import (
    ChatMessage,
    ChatResult,
    ChatRole,
    LlmClient,
    LlmConfigError,
    LlmRequestError,
    LlmResponseError,
    OpenAICompatibleClient,
    create_llm_client,
)


class _Joke(BaseModel):
    setup: str
    punchline: str = Field(min_length=1)


def _messages() -> list[ChatMessage]:
    return [
        ChatMessage(role=ChatRole.SYSTEM, content="You are helpful."),
        ChatMessage(role=ChatRole.USER, content="What is ownership?"),
    ]


def _fake_response(
    *,
    content: str | None = "Ownership is a set of rules…",
    model: str = "gpt-4o-mini",
    usage: bool = True,
) -> SimpleNamespace:
    message = SimpleNamespace(content=content)
    choice = SimpleNamespace(message=message)
    usage_obj = (
        SimpleNamespace(prompt_tokens=10, completion_tokens=20, total_tokens=30) if usage else None
    )
    return SimpleNamespace(choices=[choice], model=model, usage=usage_obj)


def _client_with_mock(
    create_return: Any = None, *, side_effect: Exception | None = None
) -> tuple[OpenAICompatibleClient, MagicMock]:
    mock_openai = MagicMock()
    if side_effect is not None:
        mock_openai.chat.completions.create.side_effect = side_effect
    else:
        mock_openai.chat.completions.create.return_value = create_return or _fake_response()
    client = OpenAICompatibleClient(
        api_key="test-key",
        default_model="gpt-4o-mini",
        default_temperature=0.0,
        client=mock_openai,
    )
    return client, mock_openai


def test_openai_compatible_client_is_llm_client() -> None:
    client, _ = _client_with_mock()
    assert isinstance(client, LlmClient)


def test_complete_happy_path_sends_messages_and_defaults() -> None:
    client, mock_openai = _client_with_mock()

    result = client.complete(_messages())

    assert isinstance(result, ChatResult)
    assert result.content.startswith("Ownership")
    assert result.model == "gpt-4o-mini"
    assert result.usage is not None
    assert result.usage.total_tokens == 30

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.0
    assert kwargs["messages"] == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "What is ownership?"},
    ]


def test_complete_overrides_model_and_temperature() -> None:
    client, mock_openai = _client_with_mock()

    client.complete(_messages(), model="openai/gpt-4o", temperature=0.3)

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4o"
    assert kwargs["temperature"] == 0.3


def test_complete_rejects_empty_messages() -> None:
    client, _ = _client_with_mock()
    with pytest.raises(LlmConfigError, match="non-empty"):
        client.complete([])


def test_complete_maps_api_error() -> None:
    api_error = APIError(
        message="boom",
        request=MagicMock(),
        body=None,
    )
    client, _ = _client_with_mock(side_effect=api_error)

    with pytest.raises(LlmRequestError, match="LLM request failed"):
        client.complete(_messages())


def test_complete_empty_content_raises() -> None:
    client, _ = _client_with_mock(create_return=_fake_response(content="  "))

    with pytest.raises(LlmResponseError, match="empty assistant content"):
        client.complete(_messages())


def test_complete_no_choices_raises() -> None:
    client, _ = _client_with_mock(create_return=SimpleNamespace(choices=[], model="m", usage=None))

    with pytest.raises(LlmResponseError, match="no choices"):
        client.complete(_messages())


def test_create_llm_client_requires_api_key(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.delenv("LLM_BASE_URL", raising=False)

    with pytest.raises(LlmConfigError, match="LLM_API_KEY"):
        create_llm_client(AgentSettings())


def test_create_llm_client_uses_settings(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("LLM_API_KEY", "secret")
    monkeypatch.setenv("LLM_BASE_URL", "https://openrouter.ai/api/v1")
    monkeypatch.setenv("LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.1")

    mock_openai = MagicMock()
    mock_openai.chat.completions.create.return_value = _fake_response()

    client = create_llm_client(AgentSettings(), client=mock_openai)
    result = client.complete(_messages())

    assert result.content
    # defaults came from settings when not overridden
    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4o-mini"
    assert kwargs["temperature"] == 0.1


def test_chat_message_rejects_empty_content() -> None:
    with pytest.raises(ValidationError):
        ChatMessage(role=ChatRole.USER, content="")


def test_complete_structured_happy_path() -> None:
    payload = (
        '{"setup": "Why did the chicken cross the road?", "punchline": "To get to the other side."}'
    )
    client, mock_openai = _client_with_mock(create_return=_fake_response(content=payload))

    joke = client.complete_structured(_messages(), _Joke)

    assert isinstance(joke, _Joke)
    assert joke.setup.startswith("Why did")
    assert "other side" in joke.punchline

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "gpt-4o-mini"
    assert kwargs["temperature"] == 0.0
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["name"] == "_Joke"
    assert kwargs["response_format"]["json_schema"]["schema"]["title"] == "_Joke"
    assert kwargs["response_format"]["json_schema"]["strict"] is False


def test_complete_structured_overrides_model_and_temperature() -> None:
    payload = '{"setup": "Q", "punchline": "A"}'
    client, mock_openai = _client_with_mock(create_return=_fake_response(content=payload))

    client.complete_structured(
        _messages(),
        _Joke,
        model="openai/gpt-4o",
        temperature=0.2,
    )

    kwargs = mock_openai.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "openai/gpt-4o"
    assert kwargs["temperature"] == 0.2
    assert "response_format" in kwargs


def test_complete_structured_invalid_json_raises() -> None:
    client, _ = _client_with_mock(create_return=_fake_response(content="not-json"))

    with pytest.raises(LlmResponseError, match="schema validation"):
        client.complete_structured(_messages(), _Joke)


def test_complete_structured_wrong_shape_raises() -> None:
    client, _ = _client_with_mock(create_return=_fake_response(content='{"setup": "only"}'))

    with pytest.raises(LlmResponseError, match="schema validation"):
        client.complete_structured(_messages(), _Joke)


def test_complete_structured_empty_content_raises() -> None:
    client, _ = _client_with_mock(create_return=_fake_response(content=" "))

    with pytest.raises(LlmResponseError, match="empty assistant content"):
        client.complete_structured(_messages(), _Joke)


def test_complete_structured_rejects_empty_messages() -> None:
    client, _ = _client_with_mock()
    with pytest.raises(LlmConfigError, match="non-empty"):
        client.complete_structured([], _Joke)


def test_complete_does_not_send_response_format() -> None:
    client, mock_openai = _client_with_mock()
    client.complete(_messages())
    assert "response_format" not in mock_openai.chat.completions.create.call_args.kwargs
