from unittest.mock import MagicMock, patch

import pytest

from autobacktest.llm.base import AgentContext, LLMError
from autobacktest.llm.litellm_provider import AgentEditResponse, LiteLLMProvider


def test_litellm_provider_properties() -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o", temperature=0.5, max_tokens=100)
    assert provider.provider_name == "litellm"
    assert provider.model == "openai/gpt-4o"
    assert provider.temperature == 0.5
    assert provider.max_tokens == 100


@patch("litellm.completion")
def test_litellm_provider_success(mock_completion: MagicMock) -> None:
    mock_completion.return_value = _mock_response(_CLEAN_JSON)
    provider = LiteLLMProvider(model="gpt-4o")

    edit = provider.generate_edit(_make_context())

    mock_completion.assert_called_once()
    _, kwargs = mock_completion.call_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 4096
    assert edit.strategy_code == _EXPECTED_PAYLOAD["strategy_code"]
    assert edit.config_yaml == _EXPECTED_PAYLOAD["config_yaml"]
    assert edit.reasoning == _EXPECTED_PAYLOAD["reasoning"]
    assert edit.lessons_text == _EXPECTED_PAYLOAD["lessons_text"]
    assert "Conservative change" in edit.raw_response


def _make_context() -> AgentContext:
    return AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: []",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )


def _mock_response(content: str) -> MagicMock:
    mock_choice = MagicMock()
    mock_choice.message.content = content
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    return mock_response


_EXPECTED_PAYLOAD = {
    "strategy_code": "def generate_signals(): return None",
    "config_yaml": "universe: [SPY]",
    "reasoning": "Conservative change",
    "lessons_text": "Mock lessons",
}

_CLEAN_JSON = """{
    "strategy_code": "def generate_signals(): return None",
    "config_yaml": "universe: [SPY]",
    "reasoning": "Conservative change",
    "lessons_text": "Mock lessons"
}"""

_MISSING_LESSONS_JSON = """{
    "strategy_code": "def generate_signals(): return None",
    "config_yaml": "universe: [SPY]",
    "reasoning": "Conservative change"
}"""


@patch("litellm.completion")
def test_litellm_provider_allows_missing_lessons_text(
    mock_completion: MagicMock,
) -> None:
    mock_completion.return_value = _mock_response(_MISSING_LESSONS_JSON)
    provider = LiteLLMProvider(model="gpt-4o")

    edit = provider.generate_edit(_make_context())

    assert edit.strategy_code == _EXPECTED_PAYLOAD["strategy_code"]
    assert edit.config_yaml == _EXPECTED_PAYLOAD["config_yaml"]
    assert edit.reasoning == _EXPECTED_PAYLOAD["reasoning"]
    assert edit.lessons_text is None


@patch("litellm.completion")
def test_litellm_provider_json_fenced(mock_completion: MagicMock) -> None:
    mock_completion.return_value = _mock_response(f"```json\n{_CLEAN_JSON}\n```")
    provider = LiteLLMProvider(model="gpt-4o")
    edit = provider.generate_edit(_make_context())
    assert edit.strategy_code == _EXPECTED_PAYLOAD["strategy_code"]
    assert edit.config_yaml == _EXPECTED_PAYLOAD["config_yaml"]
    assert edit.reasoning == _EXPECTED_PAYLOAD["reasoning"]
    assert edit.lessons_text == _EXPECTED_PAYLOAD["lessons_text"]


@patch("litellm.completion")
def test_litellm_provider_plain_fenced(mock_completion: MagicMock) -> None:
    mock_completion.return_value = _mock_response(f"```\n{_CLEAN_JSON}\n```")
    provider = LiteLLMProvider(model="gpt-4o")
    edit = provider.generate_edit(_make_context())
    assert edit.strategy_code == _EXPECTED_PAYLOAD["strategy_code"]
    assert edit.config_yaml == _EXPECTED_PAYLOAD["config_yaml"]
    assert edit.reasoning == _EXPECTED_PAYLOAD["reasoning"]
    assert edit.lessons_text == _EXPECTED_PAYLOAD["lessons_text"]


@patch("litellm.completion")
def test_litellm_provider_prose_wrapped(mock_completion: MagicMock) -> None:
    prose = f"Here is the edit:\n{_CLEAN_JSON}\nLet me know if needed."
    mock_completion.return_value = _mock_response(prose)
    provider = LiteLLMProvider(model="gpt-4o")
    edit = provider.generate_edit(_make_context())
    assert edit.strategy_code == _EXPECTED_PAYLOAD["strategy_code"]
    assert edit.config_yaml == _EXPECTED_PAYLOAD["config_yaml"]
    assert edit.reasoning == _EXPECTED_PAYLOAD["reasoning"]
    assert edit.lessons_text == _EXPECTED_PAYLOAD["lessons_text"]


@patch("litellm.completion")
def test_litellm_provider_error_handling(mock_completion: MagicMock) -> None:
    mock_completion.side_effect = Exception("API connection failure")

    provider = LiteLLMProvider(model="gpt-4o")

    with pytest.raises(LLMError) as exc_info:
        provider.generate_edit(_make_context())

    assert exc_info.value.provider == "litellm"
    assert exc_info.value.model == "gpt-4o"
    assert "API connection failure" in exc_info.value.detail


@patch("litellm.completion")
@patch("litellm.supports_response_schema")
def test_litellm_provider_response_format_selection(
    mock_supports_schema: MagicMock,
    mock_completion: MagicMock,
) -> None:
    # 1. Test when model supports response schema
    mock_supports_schema.return_value = True
    mock_completion.return_value = _mock_response(_CLEAN_JSON)
    provider_capable = LiteLLMProvider(model="gpt-4o")

    provider_capable.generate_edit(_make_context())
    _, kwargs_capable = mock_completion.call_args
    assert kwargs_capable["response_format"] == AgentEditResponse

    # 2. Test when model does not support response schema
    mock_completion.reset_mock()
    mock_supports_schema.return_value = False
    provider_incapable = LiteLLMProvider(model="deepseek-v4-pro")

    provider_incapable.generate_edit(_make_context())
    _, kwargs_incapable = mock_completion.call_args
    assert kwargs_incapable["response_format"] == {"type": "json_object"}


@patch("litellm.completion")
def test_litellm_provider_error_classification(mock_completion: MagicMock) -> None:
    import litellm

    # 1. Non-retryable (BadRequestError)
    mock_completion.side_effect = litellm.BadRequestError(
        message="Bad request",
        model="gpt-4o",
        response=MagicMock(),
        llm_provider="openai",
    )
    provider = LiteLLMProvider(model="gpt-4o")
    with pytest.raises(LLMError) as exc_info:
        provider.generate_edit(_make_context())
    assert exc_info.value.retryable is False

    # 2. Retryable (generic Exception)
    mock_completion.side_effect = Exception("Generic error")
    with pytest.raises(LLMError) as exc_info:
        provider.generate_edit(_make_context())
    assert exc_info.value.retryable is True
