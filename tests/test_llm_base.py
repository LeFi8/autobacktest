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
    assert context.lessons_text == ""
    assert context.last_attempt is None

    with pytest.raises(FrozenInstanceError):
        context.iteration = 2  # type: ignore


def test_agent_context_last_attempt_default_is_none() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )
    assert context.last_attempt is None


def test_agent_context_last_attempt_populated() -> None:
    attempt = {
        "stage": "validation",
        "error_code": "lookahead_detected",
        "detail": "shift(-1) found",
        "candidate_strategy_code": "def generate_signals(): pass",
        "candidate_config_yaml": "universe: [SPY]",
    }
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=2,
        last_attempt=attempt,
    )
    assert context.last_attempt is not None
    assert context.last_attempt["stage"] == "validation"
    assert context.last_attempt["error_code"] == "lookahead_detected"


def test_agent_context_last_attempt_is_immutable() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=2,
        last_attempt={"stage": "gate"},
    )
    with pytest.raises(FrozenInstanceError):
        context.last_attempt = None  # type: ignore


def test_agent_edit_immutability() -> None:
    edit = AgentEdit(
        strategy_code="def generate_signals(): return",
        config_yaml="universe: []",
        reasoning="none",
        raw_response="{}",
    )
    assert edit.reasoning == "none"
    assert edit.lessons_text is None

    with pytest.raises(FrozenInstanceError):
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


def test_agent_context_new_fields_default() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )
    assert context.attempt_history is None
    assert context.mode == "explore"


def test_agent_context_attempt_history_populated() -> None:
    history = [
        {
            "iteration": 1,
            "accepted": True,
            "committed": True,
            "target_metric_value": 1.25,
            "observed_sharpe": 1.3,
            "deflated_sharpe": 1.1,
            "holdout_max_drawdown": 0.08,
            "holdout_turnover": 0.4,
            "regime_passed": True,
            "rejection_reason": None,
            "config_fingerprint": {"universe": ["SPY", "TIP"], "params": {"top_n": 3}},
        }
    ]
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=2,
        attempt_history=history,
    )
    assert context.attempt_history is not None
    assert len(context.attempt_history) == 1
    assert context.attempt_history[0]["iteration"] == 1


def test_agent_context_mode_exploit() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=5,
        mode="exploit",
    )
    assert context.mode == "exploit"
