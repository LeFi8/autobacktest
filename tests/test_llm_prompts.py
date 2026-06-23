from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.llm.base import AgentContext
from autobacktest.llm.prompts import SYSTEM_PROMPT, _diff_code, build_messages


def test_system_prompt_contents() -> None:
    assert "generate_signals" in SYSTEM_PROMPT
    assert "non-negative" in SYSTEM_PROMPT
    assert "pandas" in SYSTEM_PROMPT


def test_build_messages_structure_no_report() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )
    messages = build_messages(context)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    # program_text is now in the system message (stable cacheable prefix)
    sys_content = messages[0]["content"]
    assert isinstance(sys_content, str)
    assert SYSTEM_PROMPT in sys_content
    assert "make it conservative" in sys_content
    assert "## Objective" in sys_content

    user_msg = messages[1]["content"]
    assert "## Iteration" in user_msg
    # ## Objective moved to system; should not be duplicated in user message
    assert "## Objective" not in user_msg
    assert "## Current Strategy Code" in user_msg
    assert "## Current Config" in user_msg
    assert "## Latest Evaluation" in user_msg
    assert "First iteration" in user_msg
    assert "## Instructions" in user_msg


def test_build_messages_cache_blocks_structure() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="be aggressive",
        evaluation_report=None,
        iteration=1,
    )
    messages = build_messages(context, cache_supported=True)
    assert len(messages) == 2
    sys_content = messages[0]["content"]
    assert isinstance(sys_content, list)
    assert len(sys_content) == 2
    # First block: SYSTEM_PROMPT, no cache_control
    assert sys_content[0]["type"] == "text"
    assert SYSTEM_PROMPT in sys_content[0]["text"]
    assert "cache_control" not in sys_content[0]
    # Second block: program_text, with cache_control
    assert sys_content[1]["type"] == "text"
    assert "be aggressive" in sys_content[1]["text"]
    assert sys_content[1]["cache_control"] == {"type": "ephemeral"}
    # ## Objective should NOT be in user message
    assert "## Objective" not in messages[1]["content"]


def test_build_messages_stable_block_identity_across_candidates() -> None:
    """When cache_supported=True, the stable user block must be byte-identical
    across candidates within the same iteration — only the dynamic tail changes.
    """
    ctx_a = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="be aggressive",
        evaluation_report=None,
        iteration=1,
        directive="add momentum filter",
    )
    ctx_b = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="be aggressive",
        evaluation_report=None,
        iteration=1,
        directive="add mean reversion",
    )

    msgs_a = build_messages(ctx_a, cache_supported=True)
    msgs_b = build_messages(ctx_b, cache_supported=True)

    user_a = msgs_a[1]["content"]
    user_b = msgs_b[1]["content"]

    assert isinstance(user_a, list)
    assert isinstance(user_b, list)
    assert len(user_a) == 2
    assert len(user_b) == 2

    stable_a = user_a[0]
    stable_b = user_b[0]
    assert stable_a["type"] == "text"
    assert stable_b["type"] == "text"
    assert stable_a["text"] == stable_b["text"], "Stable block must be byte-identical across candidates"
    assert stable_a.get("cache_control") == {"type": "ephemeral"}
    assert stable_b.get("cache_control") == {"type": "ephemeral"}

    dynamic_a = user_a[1]
    dynamic_b = user_b[1]
    assert dynamic_a["type"] == "text"
    assert dynamic_b["type"] == "text"
    assert dynamic_a["text"] != dynamic_b["text"], "Dynamic tail must differ across candidates"
    assert "cache_control" not in dynamic_a
    assert "cache_control" not in dynamic_b


def test_build_messages_no_cache_control_without_flag() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="be aggressive",
        evaluation_report=None,
        iteration=1,
    )
    messages = build_messages(context, cache_supported=False)
    sys_content = messages[0]["content"]
    # Non-Anthropic path: plain string, no block list, no cache_control keys
    assert isinstance(sys_content, str)
    assert "cache_control" not in str(sys_content)


def test_diff_code_produces_unified_diff() -> None:
    """_diff_code must produce a valid unified diff with ---/+++ markers
    and must produce a diff shorter than full source for a single-line change
    in a realistically-sized strategy (~30 lines)."""
    old_code = """import pandas as pd
from typing import Any

def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    lookback = config.get("lookback", 12)
    top_n = config.get("top_n", 5)
    monthly = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly.index
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    for ticker in prices.columns:
        rolling = prices[ticker].pct_change(lookback).reindex(idx, method="ffill")
        weights.loc[idx, ticker] = rolling
    top = weights.stack().groupby(level=0).nlargest(top_n).unstack(fill_value=0.0)
    return top.div(top.sum(axis=1), axis=0).fillna(0.0)
"""
    new_code = old_code.replace('lookback = config.get("lookback", 12)', 'lookback = config.get("lookback", 6)')
    diff = _diff_code(old_code, new_code)
    assert "---" in diff
    assert "+++" in diff
    assert "incumbent" in diff
    assert "candidate" in diff
    assert "12" in diff or "6" in diff
    assert len(diff) < len(new_code), "Diff should be shorter than full source for a single-line change"


def test_diff_code_empty_input_returns_empty() -> None:
    """_diff_code must return empty string when either input is empty."""
    assert _diff_code("", "some code") == ""
    assert _diff_code("some code", "") == ""
    assert _diff_code("", "") == ""


def test_diff_code_identical_code_returns_empty() -> None:
    """_diff_code for identical code should return empty string (no changes)."""
    code = "def f():\n    pass\n"
    diff = _diff_code(code, code)
    assert diff == ""


def test_build_messages_with_report() -> None:
    mock_window = WindowReport(
        start_date="2020-01-01",
        end_date="2020-12-31",
        annualized_return=0.15,
        annualized_volatility=0.10,
        sharpe_ratio=1.5,
        sortino_ratio=2.0,
        max_drawdown=0.08,
        turnover=2.5,
        information_ratio=0.5,
    )
    mock_report = EvaluationReport(
        strategy_name="haa",
        dataset_hash="hash123",
        gates_passed={"sharpe": True, "drawdown": True},
        is_accepted=True,
        rejection_reason=None,
        holdout_metrics=mock_window,
        in_sample_metrics=mock_window,
        walk_forward_metrics=[mock_window],
        regime_drawdowns={"2008": 0.12},
        regime_passed=True,
        mc_sharpe_5th=1.1,
        mc_sharpe_50th=1.4,
        mc_sharpe_95th=1.8,
        observed_sharpe=1.5,
        effective_trials=10,
        deflated_sharpe=1.3,
    )

    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=mock_report,
        iteration=3,
    )

    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Iteration" in user_msg
    assert "Current Loop Iteration: 3" in user_msg
    assert "In-Sample Walk-Forward Aggregate" in user_msg
    assert "Sharpe:" in user_msg


def test_build_messages_lessons_warning() -> None:
    # 4096 tokens limit is approx 16384 characters. So let's make it >16385 chars.
    long_lessons = "a" * 17000
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def signals(): pass",
        config_yaml="universe: []",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
        lessons_text=long_lessons,
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "exceeds the cap of 4096 tokens" in user_msg
    assert "## Lessons" in user_msg
    assert long_lessons in user_msg


def _make_mock_report() -> EvaluationReport:
    mock_window = WindowReport(
        start_date="2020-01-01",
        end_date="2020-12-31",
        annualized_return=0.12,
        annualized_volatility=0.10,
        sharpe_ratio=1.2,
        sortino_ratio=1.8,
        max_drawdown=0.09,
        turnover=0.5,
        information_ratio=0.4,
    )
    return EvaluationReport(
        strategy_name="haa",
        dataset_hash="abc",
        gates_passed={"sharpe": True},
        is_accepted=True,
        rejection_reason=None,
        holdout_metrics=mock_window,
        in_sample_metrics=mock_window,
        walk_forward_metrics=[mock_window],
        regime_drawdowns={},
        regime_passed=True,
        mc_sharpe_5th=1.0,
        mc_sharpe_50th=1.2,
        mc_sharpe_95th=1.4,
        observed_sharpe=1.2,
        effective_trials=5,
        deflated_sharpe=1.1,
    )


def test_build_messages_no_last_attempt_omits_previous_attempt_section() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Previous Attempt Result" not in user_msg


def test_build_messages_last_attempt_none_omits_section() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=2,
        last_attempt=None,
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Previous Attempt Result" not in user_msg


def test_build_messages_validation_failure_renders_section() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=2,
        last_attempt={
            "stage": "validation",
            "error_code": "lookahead_detected",
            "detail": "Shift with negative index found at line 42.",
            "candidate_strategy_code": "def generate_signals(): return df.shift(-1)",
            "candidate_config_yaml": "universe: [SPY]",
        },
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Previous Attempt Result" in user_msg
    assert "validation" in user_msg
    assert "lookahead_detected" in user_msg
    assert "Shift with negative index found at line 42." in user_msg
    assert "lookahead_detected` means" in user_msg
    assert "df.shift(-1)" in user_msg
    assert "Diagnose the failure above" in user_msg


def test_build_messages_gate_failure_renders_section() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=_make_mock_report(),
        iteration=3,
        last_attempt={
            "stage": "gate",
            "rejection_reason": "Holdout max drawdown 0.2000 exceeds limit of 0.1500.",
            "failed_gate": "max_drawdown",
            "candidate_strategy_code": "def generate_signals(): pass",
            "candidate_config_yaml": "universe: [SPY, AGG]",
            "candidate_metrics": {
                "observed_sharpe": 1.1,
                "holdout_sharpe": 0.9,
                "in_sample_max_drawdown": 0.20,
                "in_sample_turnover": 0.6,
                "regime_passed": True,
            },
        },
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Previous Attempt Result" in user_msg
    assert "gate" in user_msg
    assert "max_drawdown" in user_msg
    assert "Holdout max drawdown 0.2000 exceeds limit of 0.1500." in user_msg
    assert "in_sample_max_drawdown" in user_msg
    assert "Diagnose the failure above" in user_msg


def test_build_messages_diversity_config_failure_renders_section() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=4,
        last_attempt={
            "stage": "diversity_config",
            "detail": (
                "Config similarity 0.970 exceeded threshold 0.950. Your config was too similar to a past attempt."
            ),
            "candidate_config_yaml": "universe: [SPY]\nmomentum_windows: [1, 3, 6, 12]",
        },
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Previous Attempt Result" in user_msg
    assert "diversity_config" in user_msg
    assert "Config similarity 0.970" in user_msg
    assert "Diagnose the failure above" in user_msg


def test_build_messages_performance_target_with_report() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=_make_mock_report(),
        iteration=2,
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Performance Target" in user_msg
    assert "1.2000" in user_msg
    assert "0.20" in user_msg
    assert "always enforced" in user_msg
    assert "regime stress tests" in user_msg


def test_build_messages_performance_target_no_report() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=1,
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Performance Target" in user_msg
    assert "drawdown <= 0.20" in user_msg
    assert "always enforced" in user_msg


def test_build_messages_previous_attempt_before_instructions() -> None:
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="make it conservative",
        evaluation_report=None,
        iteration=2,
        last_attempt={
            "stage": "eval_error",
            "detail": "KeyError: 'price'",
            "candidate_strategy_code": "def generate_signals(): return df['price']",
            "candidate_config_yaml": "universe: [SPY]",
        },
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    prev_pos = user_msg.index("## Previous Attempt Result")
    instructions_pos = user_msg.index("## Instructions")
    assert prev_pos < instructions_pos


# ── Attempt History tests ──────────────────────────────────────────────────────

_ATTEMPT_COMMITTED = {
    "iteration": 1,
    "accepted": True,
    "committed": True,
    "target_metric_value": 1.25,
    "observed_sharpe": 1.3,
    "deflated_sharpe": 1.1,
    "in_sample_max_drawdown": 0.08,
    "in_sample_turnover": 0.4,
    "regime_passed": True,
    "rejection_reason": None,
    "config_fingerprint": {"universe": ["SPY", "TIP"], "params": {"top_n": 3}},
}

_ATTEMPT_REJECTED = {
    "iteration": 2,
    "accepted": False,
    "committed": False,
    "target_metric_value": 0.95,
    "observed_sharpe": 1.0,
    "deflated_sharpe": 0.8,
    "in_sample_max_drawdown": 0.22,
    "in_sample_turnover": 1.5,
    "regime_passed": False,
    "rejection_reason": "Drawdown exceeds limit",
    "config_fingerprint": {"universe": ["SPY"], "params": {"top_n": 5}},
}


def _make_context(**kwargs) -> AgentContext:  # type: ignore[no-untyped-def]
    defaults = {
        "strategy_name": "haa",
        "strategy_code": "def generate_signals(): pass",
        "config_yaml": "universe: [SPY]",
        "program_text": "make it conservative",
        "evaluation_report": None,
        "iteration": 1,
    }
    defaults.update(kwargs)
    return AgentContext(**defaults)


def test_build_messages_attempt_history_omitted_when_none() -> None:
    context = _make_context(attempt_history=None)
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Attempt History" not in user_msg


def test_build_messages_attempt_history_omitted_when_empty() -> None:
    context = _make_context(attempt_history=[])
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Attempt History" not in user_msg


def test_build_messages_attempt_history_rendered() -> None:
    accepted_not_committed = {
        "iteration": 3,
        "accepted": True,
        "committed": False,
        "target_metric_value": 1.10,
        "observed_sharpe": 1.15,
        "deflated_sharpe": 0.95,
        "in_sample_max_drawdown": 0.10,
        "in_sample_turnover": 0.6,
        "regime_passed": True,
        "rejection_reason": None,
        "config_fingerprint": {"universe": ["AGG"], "params": {"top_n": 4}},
    }
    history = [_ATTEMPT_COMMITTED, _ATTEMPT_REJECTED, accepted_not_committed]
    context = _make_context(attempt_history=history)
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Attempt History" in user_msg
    assert "| iter | outcome |" in user_msg
    assert "✓ committed" in user_msg


def test_build_messages_attempt_history_truncation() -> None:
    # 30 non-committed rows — only 25 should be shown, triggering the omit note
    history = [
        {
            "iteration": i,
            "accepted": False,
            "committed": False,
            "target_metric_value": 0.9,
            "observed_sharpe": 0.9,
            "deflated_sharpe": 0.7,
            "in_sample_max_drawdown": 0.15,
            "in_sample_turnover": 1.0,
            "regime_passed": True,
            "rejection_reason": "Below target",
            "config_fingerprint": {"universe": ["SPY"], "params": {"top_n": i}},
        }
        for i in range(1, 31)
    ]
    context = _make_context(attempt_history=history)
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Attempt History" in user_msg
    assert "oldest non-committed omitted" in user_msg


# ── Mode section tests ─────────────────────────────────────────────────────────


def test_build_messages_mode_explore_section() -> None:
    context = _make_context()  # default mode is "explore"
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Mode" in user_msg
    assert "**EXPLORE**" in user_msg


def test_build_messages_mode_exploit_section() -> None:
    context = _make_context(mode="exploit")
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Mode" in user_msg
    assert "**EXPLOIT**" in user_msg


def test_system_prompt_forbidden_names() -> None:
    from autobacktest.strategy.constants import FORBIDDEN_NAMES

    for name in FORBIDDEN_NAMES:
        assert name in SYSTEM_PROMPT


def test_build_messages_with_repair_request() -> None:
    context = _make_context(
        repair_request={
            "failed_code": "def signals(): raise ValueError()",
            "failed_config_yaml": "universe: [SPY]",
            "error_code": "smoke_test_failed",
            "error_detail": "Subprocess crashed",
        }
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Repair Request" in user_msg
    assert "def signals(): raise ValueError()" in user_msg
    assert "smoke_test_failed" in user_msg
    assert "Fix ONLY this validation error" in user_msg


def test_build_messages_with_directive() -> None:
    context = _make_context(directive="do something different")
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Candidate Directive" in user_msg
    assert "This candidate MUST: do something different" in user_msg


def test_build_messages_with_explored_config_summary() -> None:
    context = _make_context(explored_config_summary="- **top_x**: [2, 3]")
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Explored Config Space" in user_msg
    assert "- **top_x**: [2, 3]" in user_msg
    assert "Any config with similarity >" in user_msg


def test_build_messages_with_last_iteration_failures() -> None:
    context = _make_context(
        last_iteration_failures=[
            {"stage": "validation", "error_code": "undefined_name", "detail": "Name 'x' is not defined"},
            {"stage": "diversity_config", "detail": "Too similar to past configs", "params": {"top_x": 3}},
        ]
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "## Previous Iteration — All Candidates" in user_msg
    assert "1. **Stage:** validation (undefined_name) | **Detail:** Name 'x' is not defined" in user_msg
    expected = "2. **Stage:** diversity_config | **Detail:** Too similar to past configs | Parameters: {'top_x': 3}"
    assert expected in user_msg


def test_build_messages_repair_hints() -> None:
    # 1. Test lookahead_detected hint
    context = _make_context(
        repair_request={
            "failed_code": "def signals(): pass",
            "failed_config_yaml": "universe: [SPY]",
            "error_code": "lookahead_detected",
            "error_detail": "Lookahead bias sniff test failed",
        }
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "Lookahead bias was detected via a shifted-rerun test" in user_msg
    assert "Calculating statistics on the full sample" in user_msg

    # 2. Test fn-length hint
    context = _make_context(
        repair_request={
            "failed_code": "def signals(): pass",
            "failed_config_yaml": "universe: [SPY]",
            "error_code": "ast_line_limit_exceeded",
            "error_detail": "Function signals has 150 lines",
        }
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "extract logic into top-level helper functions" in user_msg

    # 3. Test config schema invalid hint
    context = _make_context(
        repair_request={
            "failed_code": "def signals(): pass",
            "failed_config_yaml": "universe: [SPY]",
            "error_code": "config_schema_invalid",
            "error_detail": "validation error",
        }
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "Strategy-specific parameters must go under the 'params' dictionary" in user_msg

    # 4. Test smoke_test_failed hint
    context = _make_context(
        repair_request={
            "failed_code": "def signals(): pass",
            "failed_config_yaml": "universe: [SPY]",
            "error_code": "smoke_test_failed",
            "error_detail": "Exception raised",
        }
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "Smoke test execution failed" in user_msg
    assert "KeyError, IndexError, ValueError" in user_msg

    # 5. Test import_failed hint
    context = _make_context(
        repair_request={
            "failed_code": "def signals(): pass",
            "failed_config_yaml": "universe: [SPY]",
            "error_code": "import_failed",
            "error_detail": "SyntaxError",
        }
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "Import failed. The Python interpreter failed to load" in user_msg

    # 6. Test signature_mismatch hint
    context = _make_context(
        repair_request={
            "failed_code": "def signals(): pass",
            "failed_config_yaml": "universe: [SPY]",
            "error_code": "signature_mismatch",
            "error_detail": "signature error",
        }
    )
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "Signature mismatch. Your strategy must define" in user_msg


def test_build_messages_explored_config_instructions() -> None:
    context = _make_context(explored_config_summary="- **top_x**: [2, 3]")
    messages = build_messages(context)
    user_msg = messages[1]["content"]
    assert "pick numeric values ≥ ~25% away from every tried value" in user_msg


# ── Cache correctness tests ─────────────────────────────────────────────────────


def test_build_messages_no_empty_content_block_exploit_mode() -> None:
    """When cache_supported=True and dynamic_tail is empty (no directive,
    no repair), the user content must contain exactly one block — the
    stable body — not a second empty text block that Anthropic rejects."""
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="be aggressive",
        evaluation_report=None,
        iteration=1,
        mode="exploit",
        directive="",
        repair_request=None,
    )
    messages = build_messages(context, cache_supported=True)
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert len(user_content) == 1
    block = user_content[0]
    assert block["type"] == "text"
    assert len(block["text"]) > 0
    assert block.get("cache_control") == {"type": "ephemeral"}


def test_build_messages_dynamic_tail_appended_when_present() -> None:
    """When cache_supported=True and a directive is set, user content
    must have exactly two blocks: stable body + dynamic tail."""
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="be aggressive",
        evaluation_report=None,
        iteration=1,
        directive="add momentum filter",
    )
    messages = build_messages(context, cache_supported=True)
    user_content = messages[1]["content"]
    assert isinstance(user_content, list)
    assert len(user_content) == 2
    assert user_content[0]["text"].startswith("## Iteration")
    assert "add momentum filter" in user_content[1]["text"]


def test_build_messages_non_cache_starts_with_stable_body() -> None:
    """When cache_supported=False, the user message string must start
    with stable_body (## Iteration ...), not with the dynamic tail."""
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="be aggressive",
        evaluation_report=None,
        iteration=1,
        directive="add momentum filter",
    )
    messages = build_messages(context, cache_supported=False)
    user_msg = messages[1]["content"]
    assert isinstance(user_msg, str)
    assert user_msg.startswith("## Iteration")
    assert "add momentum filter" in user_msg


def test_build_messages_non_cache_without_dynamic_tail() -> None:
    """When cache_supported=False and dynamic_tail is empty, the user
    message should still start with stable_body."""
    context = AgentContext(
        strategy_name="haa",
        strategy_code="def generate_signals(): pass",
        config_yaml="universe: [SPY]",
        program_text="be aggressive",
        evaluation_report=None,
        iteration=1,
    )
    messages = build_messages(context, cache_supported=False)
    user_msg = messages[1]["content"]
    assert isinstance(user_msg, str)
    assert user_msg.startswith("## Iteration")
