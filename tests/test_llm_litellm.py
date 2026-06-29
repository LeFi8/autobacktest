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


def _mock_response_with_usage(content: str) -> MagicMock:
    mock_response = _mock_response(content)
    mock_response.usage.prompt_tokens = 123
    mock_response.usage.completion_tokens = 456
    mock_response.usage.total_tokens = 579
    mock_response.usage.prompt_tokens_details.cached_tokens = 17
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
def test_litellm_provider_salvages_trailing_comma_json(mock_completion: MagicMock) -> None:
    payload = """{
    "strategy_code": "def generate_signals(): return None",
    "config_yaml": "universe: [SPY]",
    "reasoning": "Conservative change",
    "lessons_text": "Mock lessons",
}"""
    mock_completion.return_value = _mock_response(payload)
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
@patch("litellm.completion_cost")
def test_litellm_provider_parse_error_carries_usage(
    mock_completion_cost: MagicMock,
    mock_completion: MagicMock,
) -> None:
    mock_completion.return_value = _mock_response_with_usage("not json")
    mock_completion_cost.return_value = 0.042
    provider = LiteLLMProvider(model="gpt-4o")

    with pytest.raises(LLMError) as exc_info:
        provider.generate_edit(_make_context())

    err = exc_info.value
    assert err.prompt_tokens == 123
    assert err.completion_tokens == 456
    assert err.total_tokens == 579
    assert err.cached_tokens == 17
    assert err.cost == 0.042


@patch("litellm.completion")
@patch("litellm.supports_response_schema")
@patch("autobacktest.llm.litellm_provider.settings")
def test_litellm_provider_response_format_selection(
    mock_settings: MagicMock,
    mock_supports_schema: MagicMock,
    mock_completion: MagicMock,
) -> None:
    # Override response_format_override to enable format selection
    mock_settings.response_format_override = None
    mock_settings.llm_max_tokens = 4096
    mock_settings.llm_prompt_cache = False
    mock_settings.llm_request_timeout = 600.0
    mock_settings.llm_num_retries = 2

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
@patch("litellm.supports_response_schema")
def test_litellm_provider_uses_response_format_by_default(
    mock_supports_schema: MagicMock,
    mock_completion: MagicMock,
) -> None:
    mock_supports_schema.return_value = True
    mock_completion.return_value = _mock_response(_CLEAN_JSON)
    provider = LiteLLMProvider(model="openai/gpt-4o")

    provider.generate_edit(_make_context())

    _, kwargs = mock_completion.call_args
    assert kwargs["response_format"] == AgentEditResponse


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

    # env_limit caps the model's output-token budget.
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

    # env_limit caps the model's output-token budget.
    assert result == 4096


@patch("litellm.get_max_tokens")
@patch("litellm.token_counter")
def test_compute_run_max_tokens_does_not_subtract_output_buffer(
    mock_token_counter: MagicMock,
    mock_get_max_tokens: MagicMock,
) -> None:
    from autobacktest.llm.litellm_provider import _compute_run_max_tokens

    mock_token_counter.return_value = 1000
    mock_get_max_tokens.return_value = 8192

    result = _compute_run_max_tokens("deepseek/deepseek-v4-pro", 64000, None)

    assert result == 8192


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

    # instance override caps the model's output-token budget.
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

    # env_limit caps the model's output-token budget.
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

    # Fallback output-token budget is capped by env_limit.
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

    assert result == 4096


@patch("litellm.completion")
@patch("litellm.supports_response_schema")
@patch("autobacktest.llm.litellm_provider.settings")
def test_litellm_provider_response_format_fallback_skips_retries(
    mock_settings: MagicMock,
    mock_supports_schema: MagicMock,
    mock_completion: MagicMock,
) -> None:
    """Test that response_format fallback uses num_retries=0 for format attempts.

    When a response_format is rejected (400 error), the provider should not retry
    the same failing format. Only the final None format attempt should use retries.
    """
    import litellm

    # Override response_format_override to enable format selection
    mock_settings.response_format_override = None
    mock_settings.llm_max_tokens = 4096
    mock_settings.llm_prompt_cache = False
    mock_settings.llm_request_timeout = 600.0
    mock_settings.llm_num_retries = 2

    # Mock model that claims schema support
    mock_supports_schema.return_value = True

    # Track all call args
    call_args_list = []

    def mock_completion_side_effect(**kwargs):
        call_args_list.append(kwargs.copy())
        resp_format = kwargs.get("response_format")

        # First two calls with response_format: return 400 error (response_format rejected)
        if resp_format in (AgentEditResponse, {"type": "json_object"}):
            raise litellm.BadRequestError(
                message="This response_format type is unavailable now",
                model="openai/gpt-4o",
                response=MagicMock(),
                llm_provider="openai",
            )
        # Third call with None: return success
        else:
            return _mock_response(_CLEAN_JSON)

    mock_completion.side_effect = mock_completion_side_effect

    # Use a model NOT in MODELS_WITHOUT_RESPONSE_FORMAT to test the fallback chain
    provider = LiteLLMProvider(model="openai/gpt-4o")
    edit = provider.generate_edit(_make_context())

    # Verify the result is successful
    assert edit.strategy_code == _EXPECTED_PAYLOAD["strategy_code"]
    assert edit.config_yaml == _EXPECTED_PAYLOAD["config_yaml"]

    # Verify exactly 3 calls were made (AgentEditResponse, json_object, None)
    assert len(call_args_list) == 3

    # Verify num_retries=0 for response_format attempts
    assert call_args_list[0]["num_retries"] == 0, "AgentEditResponse should use num_retries=0"
    assert call_args_list[1]["num_retries"] == 0, "json_object should use num_retries=0"

    # Verify num_retries=settings.llm_num_retries for None format
    assert call_args_list[2]["num_retries"] == mock_settings.llm_num_retries, (
        "None format should use num_retries=settings.llm_num_retries"
    )


@patch("litellm.completion")
@patch("litellm.supports_response_schema")
@patch("autobacktest.llm.litellm_provider.settings")
def test_litellm_provider_response_format_fallback_all_retries_skipped(
    mock_settings: MagicMock,
    mock_supports_schema: MagicMock,
    mock_completion: MagicMock,
) -> None:
    """Test that when all response_format attempts fail, only None format uses retries.

    This verifies the total API call count is 2 (not 4) when all formats are rejected.
    """
    import litellm

    # Override response_format_override to enable format selection
    mock_settings.response_format_override = None
    mock_settings.llm_max_tokens = 4096
    mock_settings.llm_prompt_cache = False
    mock_settings.llm_request_timeout = 600.0
    mock_settings.llm_num_retries = 2

    # Mock model that does NOT claim schema support (skip AgentEditResponse)
    mock_supports_schema.return_value = False

    call_count = 0

    def mock_completion_side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        resp_format = kwargs.get("response_format")

        # First call with json_object: return 400 error
        if resp_format == {"type": "json_object"}:
            raise litellm.BadRequestError(
                message="This response_format type is unavailable now",
                model="openai/gpt-4o",
                response=MagicMock(),
                llm_provider="openai",
            )
        # Second call with None: return success
        else:
            return _mock_response(_CLEAN_JSON)

    mock_completion.side_effect = mock_completion_side_effect

    # Use a model NOT in MODELS_WITHOUT_RESPONSE_FORMAT to test the fallback chain
    provider = LiteLLMProvider(model="openai/gpt-4o")
    edit = provider.generate_edit(_make_context())

    # Verify the result is successful
    assert edit.strategy_code == _EXPECTED_PAYLOAD["strategy_code"]

    # Verify exactly 2 calls were made (json_object, None) - not 4 (2 json_object retries + 2 None retries)
    assert call_count == 2, f"Expected 2 API calls, got {call_count}"
