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
    mock_completion.side_effect = litellm.BadRequestError(  # type: ignore[attr-defined]
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


@patch("litellm.completion")
def test_litellm_provider_malformed_json_raises_error(mock_completion: MagicMock) -> None:
    # Trailing comma and missing closing brace
    malformed = """{
    "strategy_code": "def generate_signals(): return None",
    "config_yaml": "universe: [SPY]",
    "reasoning": "Conservative change",
    "lessons_text": "Mock lessons",
"""
    mock_completion.return_value = _mock_response(malformed)
    provider = LiteLLMProvider(model="gpt-4o")
    with pytest.raises(LLMError):
        provider.generate_edit(_make_context())

    # Garbage JSON still raises LLMError
    mock_completion.return_value = _mock_response("garbage stuff {not json")
    with pytest.raises(LLMError):
        provider.generate_edit(_make_context())


# --- _compute_run_max_tokens unit tests ---


@patch("litellm.get_max_tokens")
@patch("litellm.token_counter")
def test_compute_run_max_tokens_basic(
    mock_token_counter: MagicMock,
    mock_get_max_tokens: MagicMock,
) -> None:
    """Test basic token computation with mocked litellm functions."""
    from autobacktest.llm.litellm_provider import _compute_run_max_tokens

    mock_token_counter.return_value = 1000
    mock_get_max_tokens.return_value = 8192

    result = _compute_run_max_tokens("gpt-4o", 4096, None)

    # 8192 - 4096 (buffer) = 4096, capped by env_limit 4096
    assert result == 4096


@patch("litellm.get_max_tokens")
@patch("litellm.token_counter")
def test_compute_run_max_tokens_reasoning_model(
    mock_token_counter: MagicMock,
    mock_get_max_tokens: MagicMock,
) -> None:
    """Test with reasoning model (deepseek-v4-pro) - should NOT subtract prompt tokens."""
    from autobacktest.llm.litellm_provider import _compute_run_max_tokens

    mock_token_counter.return_value = 3637
    mock_get_max_tokens.return_value = 8192

    result = _compute_run_max_tokens("deepseek/deepseek-v4-pro", 4096, None)

    # 8192 - 4096 (buffer) = 4096, capped by env_limit 4096
    # Before fix: would be 8192 - 3637 - 4096 = 459 (too small)
    assert result == 4096


@patch("litellm.get_max_tokens")
@patch("litellm.token_counter")
def test_compute_run_max_tokens_instance_override(
    mock_token_counter: MagicMock,
    mock_get_max_tokens: MagicMock,
) -> None:
    """Test instance_max_tokens override when it differs from env_limit."""
    from autobacktest.llm.litellm_provider import _compute_run_max_tokens

    mock_token_counter.return_value = 1000
    mock_get_max_tokens.return_value = 8192

    # instance_max_tokens=2048 differs from env_limit=4096, so it overrides
    result = _compute_run_max_tokens("gpt-4o", 4096, 2048)

    # 8192 - 4096 (buffer) = 4096, capped by instance_max_tokens 2048
    assert result == 2048


@patch("litellm.get_max_tokens")
@patch("litellm.token_counter")
def test_compute_run_max_tokens_instance_matches_env(
    mock_token_counter: MagicMock,
    mock_get_max_tokens: MagicMock,
) -> None:
    """Test when instance_max_tokens equals env_limit (no override)."""
    from autobacktest.llm.litellm_provider import _compute_run_max_tokens

    mock_token_counter.return_value = 1000
    mock_get_max_tokens.return_value = 8192

    # instance_max_tokens=4096 equals env_limit=4096, no override
    result = _compute_run_max_tokens("gpt-4o", 4096, 4096)

    # 8192 - 4096 (buffer) = 4096, capped by env_limit 4096
    assert result == 4096


@patch("litellm.get_max_tokens")
@patch("litellm.token_counter")
def test_compute_run_max_tokens_fallback(
    mock_token_counter: MagicMock,
    mock_get_max_tokens: MagicMock,
) -> None:
    """Test fallback when litellm functions raise exceptions."""
    from autobacktest.llm.litellm_provider import _compute_run_max_tokens

    mock_token_counter.side_effect = Exception("token_counter failed")
    mock_get_max_tokens.side_effect = Exception("get_max_tokens failed")

    result = _compute_run_max_tokens("unknown-model", 4096, None)

    # Fallback: context_window = 128_000
    # 128_000 - 4096 (buffer) = 123_904, capped by env_limit 4096
    assert result == 4096


@patch("litellm.get_max_tokens")
@patch("litellm.token_counter")
def test_compute_run_max_tokens_small_context(
    mock_token_counter: MagicMock,
    mock_get_max_tokens: MagicMock,
) -> None:
    """Test with small context window model where dynamic_max < env_limit."""
    from autobacktest.llm.litellm_provider import _compute_run_max_tokens

    mock_token_counter.return_value = 500
    mock_get_max_tokens.return_value = 4096  # Small model

    result = _compute_run_max_tokens("gpt-3.5-turbo", 4096, None)

    # 4096 - 4096 (buffer) = 0, clamped to 1 by max(1, ...)
    assert result == 1
