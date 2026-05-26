from dataclasses import FrozenInstanceError

import pytest

from autobacktest.llm.base import AgentContext, AgentEdit, LLMError, LLMProvider


def test_agent_context_immutability() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )
    assert context.strategy_name == "haa"
    assert context.iteration == 1

    with pytest.raises(FrozenInstanceError):
        # type: ignore
        context.iteration = 2  # type: ignore


def test_agent_edit_immutability() -> None:
    edit = AgentEdit(
        strategy_code="def generate_signals(): return",
        config_yaml="universe: []",
        reasoning="none",
        raw_response="{}",
    )
    assert edit.reasoning == "none"

    with pytest.raises(FrozenInstanceError):
        # type: ignore
        edit.reasoning = "new reasoning"  # type: ignore


def test_llm_provider_abc_enforcement() -> None:
    with pytest.raises(TypeError):
        LLMProvider()  # type: ignore


def test_llm_error_fields() -> None:
    err = LLMError(provider="openai", model="gpt-4o", detail="Rate limit exceeded")
    assert err.provider == "openai"
    assert err.model == "gpt-4o"
    assert err.detail == "Rate limit exceeded"
    assert "openai" in str(err)
    assert "gpt-4o" in str(err)
    assert "Rate limit exceeded" in str(err)
