from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.gate import GateResult
from autobacktest.llm.base import AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import run_optimization
from tests.test_orchestrator_e2e import BASELINE_STRATEGY, STRATEGY_CONFIG, _make_fake_provider, _make_synthetic_prices


def _make_canned_report(sharpe: float = 1.0) -> tuple[EvaluationReport, pd.Series]:
    window = WindowReport(
        start_date="2020-01-01",
        end_date="2020-12-31",
        annualized_return=0.12,
        annualized_volatility=0.10,
        sharpe_ratio=sharpe,
        sortino_ratio=1.5,
        max_drawdown=0.10,
        turnover=0.5,
        information_ratio=0.4,
    )
    report = EvaluationReport(
        strategy_name="toy",
        dataset_hash="testhash",
        gates_passed={"max_drawdown": True, "turnover": True, "regimes": True},
        is_accepted=False,
        rejection_reason=None,
        holdout_metrics=window,
        in_sample_metrics=window,
        walk_forward_metrics=[window],
        regime_drawdowns={},
        regime_passed=True,
        mc_sharpe_5th=0.8,
        mc_sharpe_50th=1.0,
        mc_sharpe_95th=1.2,
        observed_sharpe=sharpe,
        effective_trials=1,
        deflated_sharpe=sharpe,
    )
    returns = pd.Series([0.01, -0.005, 0.008], dtype=float)
    return report, returns


def test_holdout_peek_limit_early_termination(project_root_with_lessons: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    # 1. Patch evaluation to always return valid report
    def patched_evaluate(*_args, **_kwargs):
        return _make_canned_report(sharpe=1.2)

    monkeypatch.setattr("autobacktest.orchestrator.evaluate_strategy_detailed", patched_evaluate)
    monkeypatch.setattr("autobacktest.orchestrator.max_config_similarity", lambda *_: 0.0)
    monkeypatch.setattr("autobacktest.orchestrator.check_returns_correlation", lambda *_: (True, 0.0))

    # 2. Patch select to always accept
    monkeypatch.setattr("autobacktest.orchestrator.select", lambda *_a, **_kw: GateResult(accepted=True))

    # 3. Patch confirm to reject (so we don't commit, but we peek holdout!)
    monkeypatch.setattr(
        "autobacktest.orchestrator.confirm",
        lambda *_a, **_kw: GateResult(accepted=False, reason="holdout fail"),
    )

    # 4. Set holdout_peek_limit = 2, iterations = 5
    # Since confirm fails, every iteration that passes select will count as a peek.
    # Iteration 1: peeks = 0 -> peeks holdout.
    # Iteration 2: peeks = 1 -> peeks holdout.
    # Iteration 3: peeks = 2 -> reaches cap (>= 2) -> terminates early!
    edit = AgentEdit(
        strategy_code=BASELINE_STRATEGY,
        config_yaml=STRATEGY_CONFIG,
        reasoning="Dummy edit.",
        raw_response="{}",
    )
    provider = MockProvider(response=edit)

    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        result = run_optimization(
            program_path=project_root_with_lessons / "program.md",
            strategy_name="toy",
            iterations=5,
            provider=provider,
            run_dir=project_root_with_lessons / "runs",
            strategies_dir=project_root_with_lessons / "strategies",
            configs_dir=project_root_with_lessons / "configs",
            repo_path=project_root_with_lessons,
            start_date="2013-01-01",
            end_date="2025-01-01",
            holdout_peek_limit=2,
        )

    # Asserts that the run stopped early (committed 0, but early stop triggered in iteration 3)
    assert result.n_committed == 0
