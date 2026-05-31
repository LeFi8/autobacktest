"""Unit tests for the lexicographic improvement gate."""

from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.gate import TargetMetric, accept


def _create_mock_report(
    max_drawdown: float = 0.10,
    regime_passed: bool = True,
    turnover: float = 0.5,
    deflated_sharpe: float = 0.98,
    sharpe: float = 1.5,
    sortino: float = 2.0,
    information_ratio: float = 1.0,
) -> EvaluationReport:
    """Helper to mock EvaluationReport dataclass with parameter overrides."""
    window = WindowReport(
        start_date="2023-01-01",
        end_date="2025-12-31",
        annualized_return=0.15,
        annualized_volatility=0.10,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_drawdown,
        turnover=turnover,
        information_ratio=information_ratio,
    )
    return EvaluationReport(
        strategy_name="mock_strat",
        dataset_hash="abc",
        gates_passed={},
        is_accepted=True,
        rejection_reason=None,
        holdout_metrics=window,
        walk_forward_metrics=[window],
        regime_drawdowns={},
        regime_passed=regime_passed,
        mc_sharpe_5th=0.5,
        mc_sharpe_50th=1.2,
        mc_sharpe_95th=2.0,
        observed_sharpe=sharpe,
        effective_trials=1,
        deflated_sharpe=deflated_sharpe,
    )


def test_gate_accepts_good_candidate_no_baseline() -> None:
    """Verifies that a valid candidate is accepted when no baseline exists."""
    rep = _create_mock_report()
    res = accept(rep, baseline=None)
    assert res.accepted
    assert res.failed_gate is None


def test_gate_rejects_excessive_drawdown() -> None:
    """Verifies drawdown constraint is strictly checked."""
    rep = _create_mock_report(max_drawdown=0.20)
    res = accept(rep, baseline=None, dd_limit=0.15)
    assert not res.accepted
    assert res.failed_gate == "max_drawdown"


def test_gate_rejects_failed_regimes() -> None:
    """Verifies stress regimes check blocks progress."""
    rep = _create_mock_report(regime_passed=False)
    res = accept(rep, baseline=None)
    assert not res.accepted
    assert res.failed_gate == "regimes"


def test_gate_rejects_excessive_turnover() -> None:
    """Verifies turnover limit hard constraint."""
    rep = _create_mock_report(turnover=1.5)
    res = accept(rep, baseline=None, turnover_limit=1.0)
    assert not res.accepted
    assert res.failed_gate == "turnover"


def test_gate_dsr_not_a_hard_gate() -> None:
    """Verifies Deflated Sharpe Ratio is no longer a hard gate."""
    rep = _create_mock_report(deflated_sharpe=0.10)
    res = accept(rep, baseline=None)
    assert res.accepted


def test_gate_lexicographic_ordering() -> None:
    """Verifies that lexicographic check evaluates limits in priority sequence.

    Candidate violates Drawdown AND Turnover limits. Max drawdown is checked
    first, so it should be reported as the failure cause.
    """
    rep = _create_mock_report(max_drawdown=0.25, turnover=2.0)
    res = accept(rep, baseline=None, dd_limit=0.15, turnover_limit=1.0)
    assert not res.accepted
    assert res.failed_gate == "max_drawdown"


def test_gate_improvement_over_baseline() -> None:
    """Verifies candidate is rejected if target metric does not improve."""
    base = _create_mock_report(sharpe=1.2)

    # 1. Candidate is identical -> rejected (needs positive improvement)
    cand_same = _create_mock_report(sharpe=1.2)
    res_same = accept(cand_same, baseline=base, target_metric=TargetMetric.SHARPE)
    assert not res_same.accepted
    assert res_same.failed_gate == "target_metric_improvement"

    # 2. Candidate is worse -> rejected
    cand_worse = _create_mock_report(sharpe=1.1)
    res_worse = accept(cand_worse, baseline=base, target_metric=TargetMetric.SHARPE)
    assert not res_worse.accepted

    # 3. Candidate is better -> accepted
    cand_better = _create_mock_report(sharpe=1.3)
    res_better = accept(cand_better, baseline=base, target_metric=TargetMetric.SHARPE)
    assert res_better.accepted


def test_gate_different_target_metrics() -> None:
    """Verifies that the gate respects specified TargetMetric choices."""
    base = _create_mock_report(sharpe=1.5, sortino=2.0, information_ratio=1.0)

    # Better Sortino, worse Sharpe -> accepted if Sortino targeted
    cand = _create_mock_report(sharpe=1.4, sortino=2.2, information_ratio=0.9)

    res_sortino = accept(cand, baseline=base, target_metric=TargetMetric.SORTINO)
    assert res_sortino.accepted  # Sortino improved


def test_gate_rejects_nan_metrics() -> None:
    """Verifies that the gate rejects any candidates returning NaN metrics."""
    # 1. NaN Drawdown
    rep_nan_dd = _create_mock_report(max_drawdown=float("nan"))
    res = accept(rep_nan_dd, baseline=None)
    assert not res.accepted
    assert res.failed_gate == "max_drawdown"

    # 2. NaN Turnover
    rep_nan_turn = _create_mock_report(turnover=float("nan"))
    res = accept(rep_nan_turn, baseline=None)
    assert not res.accepted
    assert res.failed_gate == "turnover"

    # 3. NaN DSR — accepted since DSR is no longer a hard gate
    rep_nan_dsr = _create_mock_report(deflated_sharpe=float("nan"))
    res = accept(rep_nan_dsr, baseline=None)
    assert res.accepted

    # 4. NaN Target Metric
    base = _create_mock_report(sharpe=1.2)
    rep_nan_sharpe = _create_mock_report(sharpe=float("nan"))
    res = accept(rep_nan_sharpe, baseline=base, target_metric=TargetMetric.SHARPE)
    assert not res.accepted
    assert res.failed_gate == "target_metric_improvement"


def test_gate_resolves_limits_from_config() -> None:
    """Verifies dd_limit and turnover_limit resolution from config."""
    # 1. Config dict
    config_dict = {"max_drawdown_limit": 0.08, "turnover_limit": 0.3}
    rep = _create_mock_report(max_drawdown=0.09, turnover=0.25)

    # Passes with standard defaults (0.20 / 2.0)
    res_default = accept(rep, baseline=None)
    assert res_default.accepted

    # Fails with strict config dict limits
    res_config = accept(rep, baseline=None, config=config_dict)
    assert not res_config.accepted
    assert res_config.failed_gate == "max_drawdown"

    # 2. Config object
    class MockConfig:
        max_drawdown_limit = 0.12
        turnover_limit = 0.20

    res_obj = accept(rep, baseline=None, config=MockConfig())
    assert not res_obj.accepted
    assert res_obj.failed_gate == "turnover"


def test_gate_respects_min_improvement() -> None:
    """Verifies that candidate improves by at least min_improvement."""
    base = _create_mock_report(sharpe=1.20)

    # Candidate improves by 0.03 (1.23 vs 1.20)
    cand = _create_mock_report(sharpe=1.23)

    # Passes with 0.01 improvement threshold
    res_pass = accept(cand, baseline=base, min_improvement=0.01)
    assert res_pass.accepted

    # Fails with 0.05 improvement threshold
    res_fail = accept(cand, baseline=base, min_improvement=0.05)
    assert not res_fail.accepted
    assert res_fail.failed_gate == "target_metric_improvement"
