from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.llm.base import AgentContext
from autobacktest.llm.prompts import SYSTEM_PROMPT, build_messages


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
    assert messages[0]["content"] == SYSTEM_PROMPT

    user_msg = messages[1]["content"]
    assert "## Iteration" in user_msg
    assert "## Objective" in user_msg
    assert "## Current Strategy Code" in user_msg
    assert "## Current Config" in user_msg
    assert "## Latest Evaluation" in user_msg
    assert "First iteration" in user_msg
    assert "## Instructions" in user_msg


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
    assert "hash123" in user_msg
    assert "observed_sharpe" in user_msg


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
                "holdout_max_drawdown": 0.20,
                "holdout_turnover": 0.6,
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
    assert "holdout_max_drawdown" in user_msg
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
                "Config similarity 0.970 exceeded threshold 0.950."
                " Your config was too similar to a past attempt."
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
    assert "overfitting insight only" in user_msg
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
    assert "overfitting insight only" in user_msg


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
