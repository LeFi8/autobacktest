from unittest.mock import MagicMock, patch

import pytest

from autobacktest.llm.base import AgentContext, LLMError
from autobacktest.llm.litellm_provider import LiteLLMProvider


def test_litellm_provider_properties() -> None:
    provider = LiteLLMProvider(model="openai/gpt-4o", temperature=0.5, max_tokens=100)
    assert provider.provider_name == "litellm"
    assert provider.model == "openai/gpt-4o"
    assert provider.temperature == 0.5
    assert provider.max_tokens == 100


@patch("litellm.completion")
def test_litellm_provider_success(mock_completion: MagicMock) -> None:
    # Setup mock return value matching LiteLLM format
    mock_choice = MagicMock()
    mock_choice.message.content = """{
        "strategy_code": "def generate_signals(): return None",
        "config_yaml": "universe: [SPY]",
        "reasoning": "Conservative change"
    }"""
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_completion.return_value = mock_response

    provider = LiteLLMProvider(model="gpt-4o")
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: []",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )

    edit = provider.generate_edit(context)

    # Verify LiteLLM called correctly
    mock_completion.assert_called_once()
    _, kwargs = mock_completion.call_args
    assert kwargs["model"] == "gpt-4o"
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 4096

    # Verify structured parsing
    assert edit.strategy_code == "def generate_signals(): return None"
    assert edit.config_yaml == "universe: [SPY]"
    assert edit.reasoning == "Conservative change"
    assert "Conservative change" in edit.raw_response


@patch("litellm.completion")
def test_litellm_provider_error_handling(mock_completion: MagicMock) -> None:
    # Setup mock to raise API exception
    mock_completion.side_effect = Exception("API connection failure")

    provider = LiteLLMProvider(model="gpt-4o")
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: []",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )

    with pytest.raises(LLMError) as exc_info:
        provider.generate_edit(context)

    assert exc_info.value.provider == "litellm"
    assert exc_info.value.model == "gpt-4o"
    assert "API connection failure" in exc_info.value.detail
