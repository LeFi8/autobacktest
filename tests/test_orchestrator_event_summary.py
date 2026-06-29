from unittest.mock import MagicMock, patch

from autobacktest.config import settings
from autobacktest.llm.base import AgentContext, LLMError
from autobacktest.orchestrator import _OptimizationState


def test_build_candidates_summary_preserves_llm_error_detail() -> None:
    state = object.__new__(_OptimizationState)
    candidate_results = [
        {
            "edit": None,
            "directive": "lower drawdown",
            "llm_error": True,
            "detail": "LLMError(provider='litellm', model='deepseek', detail='finish_reason: length')",
            "finish_reason": "length",
            "retryable": True,
            "prompt_tokens": 123,
            "completion_tokens": 456,
            "total_tokens": 579,
            "cost": 0.042,
            "cached_tokens": 17,
        }
    ]

    summary = state._build_candidates_summary(candidate_results, winner=None)

    assert summary == [
        {
            "llm_error": True,
            "detail": "LLMError(provider='litellm', model='deepseek', detail='finish_reason: length')",
            "finish_reason": "length",
            "retryable": True,
            "directive": "lower drawdown",
            "prompt_tokens": 123,
            "completion_tokens": 456,
            "total_tokens": 579,
            "cost": 0.042,
            "cached_tokens": 17,
        }
    ]


def test_generate_and_pre_validate_candidates_counts_llm_error_usage() -> None:
    state = object.__new__(_OptimizationState)
    state.mode = "explore"
    state.consecutive_no_accept = 0
    state.iterations = 1
    state.total_prompt_tokens = 0
    state.total_completion_tokens = 0
    state.total_cost = 0.0
    state.provider = object()
    ctx = AgentContext(
        strategy_name="toy",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: []",
        program_text="improve",
        evaluation_report=None,
        iteration=1,
    )
    llm_error = LLMError(
        provider="litellm",
        model="deepseek",
        detail="finish_reason: length",
        finish_reason="length",
        prompt_tokens=123,
        completion_tokens=456,
        total_tokens=579,
        cost=0.042,
        cached_tokens=17,
    )

    with (
        patch.object(settings, "n_candidates", 1),
        patch("autobacktest.orchestrator._generate_candidates", return_value=[(None, llm_error)]),
        patch.object(_OptimizationState, "_validate_diversity_and_guards", return_value=None),
    ):
        candidates = state.generate_and_pre_validate_candidates(1, ctx, object(), MagicMock())

    assert state.total_prompt_tokens == 123
    assert state.total_completion_tokens == 456
    assert state.total_cost == 0.042
    assert candidates[0]["llm_error"] is True
    assert candidates[0]["finish_reason"] == "length"
    assert candidates[0]["prompt_tokens"] == 123
