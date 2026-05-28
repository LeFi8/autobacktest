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
