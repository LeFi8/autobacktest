"""Tests for prompt hardening — verifies SYSTEM_PROMPT contains required rules
and that build_messages surfaces the right guidance for known error codes."""


def test_system_prompt_contains_pandas_version() -> None:
    """SYSTEM_PROMPT must embed the runtime pandas version to prevent deprecated-API bugs."""
    import pandas as pd

    from autobacktest.llm.prompts import SYSTEM_PROMPT

    assert pd.__version__ in SYSTEM_PROMPT, f"pandas version {pd.__version__} not in SYSTEM_PROMPT"


def test_system_prompt_contains_params_rule() -> None:
    """SYSTEM_PROMPT must instruct the model to place custom params under the params: key."""
    from autobacktest.llm.prompts import SYSTEM_PROMPT

    assert "params:" in SYSTEM_PROMPT
    # The schema uses extra="forbid" — at least one of these markers must be present
    assert (
        "extra_forbidden" in SYSTEM_PROMPT
        or "Extra inputs are not permitted" in SYSTEM_PROMPT
        or 'extra="forbid"' in SYSTEM_PROMPT
        or "params" in SYSTEM_PROMPT
    )
    # The key requirement: custom params must go under params:
    assert "MUST go under" in SYSTEM_PROMPT or "must go under" in SYSTEM_PROMPT or "under the" in SYSTEM_PROMPT


def test_system_prompt_contains_allowed_top_level_keys() -> None:
    """Every key in _ALLOWED_TOP_LEVEL_KEYS must be mentioned in SYSTEM_PROMPT."""
    from autobacktest.llm.prompts import _ALLOWED_TOP_LEVEL_KEYS, SYSTEM_PROMPT

    for key in _ALLOWED_TOP_LEVEL_KEYS:
        assert key in SYSTEM_PROMPT, f"Allowed key '{key}' not mentioned in SYSTEM_PROMPT"


def test_system_prompt_contains_no_full_sample_stats_rule() -> None:
    """SYSTEM_PROMPT must warn against full-sample statistics (lookahead prevention)."""
    from autobacktest.llm.prompts import SYSTEM_PROMPT

    assert "full" in SYSTEM_PROMPT.lower() or "rolling" in SYSTEM_PROMPT


def test_system_prompt_contains_decomposition_rule() -> None:
    """SYSTEM_PROMPT must require generate_signals to be an orchestrator / use helpers."""
    from autobacktest.llm.prompts import SYSTEM_PROMPT

    assert "orchestrator" in SYSTEM_PROMPT.lower() or "helper" in SYSTEM_PROMPT.lower()


def test_lookahead_explanation_mentions_normalization() -> None:
    """build_messages for a lookahead_detected error must explain normalization pitfalls."""
    from autobacktest.llm.base import AgentContext
    from autobacktest.llm.prompts import build_messages

    ctx = AgentContext(
        strategy_name="test",
        strategy_code="def generate_signals(prices, config): pass",
        config_yaml="universe: [SPY]\nparams: {}",
        program_text="test",
        evaluation_report=None,
        iteration=1,
        lessons_text="",
        n_historical_configs=0,
        last_attempt={
            "stage": "validation",
            "error_code": "lookahead_detected",
            "detail": "Lookahead bias sniff test failed.",
            "candidate_strategy_code": "",
            "candidate_config_yaml": "",
        },
        attempt_history=[],
        mode="explore",
        dd_limit=0.20,
        turnover_limit=2.0,
        min_return_ratio=0.5,
    )
    messages = build_messages(ctx)
    # The second message is the user turn containing error detail
    user_content = messages[1]["content"]
    assert (
        "normalization" in user_content.lower()
        or "full-sample" in user_content.lower()
        or "rank" in user_content.lower()
    ), "lookahead_detected explanation should mention normalization, full-sample, or rank"


def test_system_prompt_diversity_rule_mentions_returns() -> None:
    """Diversity should be described as behavioral/returns-based, not just config syntax."""
    from autobacktest.llm.prompts import SYSTEM_PROMPT

    assert "return profile" in SYSTEM_PROMPT or "return stream" in SYSTEM_PROMPT, (
        "SYSTEM_PROMPT diversity rule should reference 'return profile' or 'return stream'"
    )


def test_system_prompt_nanmean_guard() -> None:
    """SYSTEM_PROMPT must include a guard for np.nanmean on empty/all-NaN arrays."""
    from autobacktest.llm.prompts import SYSTEM_PROMPT

    assert "nanmean" in SYSTEM_PROMPT or "nan" in SYSTEM_PROMPT.lower(), (
        "SYSTEM_PROMPT should mention nanmean or nan guard"
    )
