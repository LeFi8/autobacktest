"""Unit tests for the lexicographic improvement gate."""

from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.gate import TargetMetric, accept, confirm, select


def _create_mock_report(
    max_drawdown: float = 0.10,
    regime_passed: bool = True,
    turnover: float = 0.5,
    deflated_sharpe: float = 0.98,
    sharpe: float = 1.5,
    sortino: float = 2.0,
    information_ratio: float = 1.0,
    annualized_return: float = 0.15,
    pbo: float | None = None,
) -> EvaluationReport:
    """Helper to mock EvaluationReport dataclass with parameter overrides."""
    window = WindowReport(
        start_date="2023-01-01",
        end_date="2025-12-31",
        annualized_return=annualized_return,
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
        in_sample_metrics=window,
        walk_forward_metrics=[window],
        regime_drawdowns={},
        regime_passed=regime_passed,
        mc_sharpe_5th=0.5,
        mc_sharpe_50th=1.2,
        mc_sharpe_95th=2.0,
        observed_sharpe=sharpe,
        effective_trials=1,
        deflated_sharpe=deflated_sharpe,
        pbo=pbo,
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
    raw_config = {"select_compare_metric": "raw", "select_improvement_tol": 0.0}

    # 1. Candidate is identical -> rejected (needs positive improvement)
    cand_same = _create_mock_report(sharpe=1.2)
    res_same = accept(cand_same, baseline=base, target_metric=TargetMetric.SHARPE, config=raw_config)
    assert not res_same.accepted
    assert res_same.failed_gate == "target_metric_improvement"

    # 2. Candidate is worse -> rejected
    cand_worse = _create_mock_report(sharpe=1.1)
    res_worse = accept(cand_worse, baseline=base, target_metric=TargetMetric.SHARPE, config=raw_config)
    assert not res_worse.accepted

    # 3. Candidate is better -> accepted
    cand_better = _create_mock_report(sharpe=1.3)
    res_better = accept(cand_better, baseline=base, target_metric=TargetMetric.SHARPE, config=raw_config)
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

    # 4. NaN Target Metric (pin to raw comparison so NaN sharpe triggers the gate)
    base = _create_mock_report(sharpe=1.2)
    rep_nan_sharpe = _create_mock_report(sharpe=float("nan"))
    res = accept(
        rep_nan_sharpe,
        baseline=base,
        target_metric=TargetMetric.SHARPE,
        config={"select_compare_metric": "raw", "select_improvement_tol": 0.0},
    )
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


def test_gate_select_rejects_dsr_degradation() -> None:
    """Verifies select gate rejects DSR degradation (always-on by default)."""
    base = _create_mock_report(sharpe=1.0, deflated_sharpe=0.90)
    cand = _create_mock_report(sharpe=1.1, deflated_sharpe=0.50)
    # Use raw compare metric so improvement check passes, letting DSR gate fire
    res = accept(cand, baseline=base, config={"select_compare_metric": "raw", "select_improvement_tol": 0.0})
    assert not res.accepted
    assert res.failed_gate == "dsr_non_degradation"


def test_gate_dsr_non_degradation_accepts_when_dsr_improves() -> None:
    """Verifies DSR non-degradation gate accepts when candidate DSR is above baseline."""
    base = _create_mock_report(sharpe=1.0, deflated_sharpe=0.80)
    cand = _create_mock_report(sharpe=1.1, deflated_sharpe=0.85)
    res = accept(cand, baseline=base, require_dsr_non_degradation=True)
    assert res.accepted


def test_gate_dsr_non_degradation_rejects_when_dsr_degrades() -> None:
    """Verifies DSR non-degradation gate rejects when candidate DSR significantly degrades."""
    base = _create_mock_report(sharpe=1.0, deflated_sharpe=0.80)
    cand = _create_mock_report(sharpe=1.1, deflated_sharpe=0.50)
    # Use raw compare metric so improvement check passes, letting DSR gate fire
    res = accept(
        cand,
        baseline=base,
        require_dsr_non_degradation=True,
        config={"select_compare_metric": "raw", "select_improvement_tol": 0.0},
    )
    assert not res.accepted
    assert res.failed_gate == "dsr_non_degradation"


def test_gate_dsr_non_degradation_no_baseline_skipped() -> None:
    """Verifies DSR non-degradation gate is skipped when no baseline exists."""
    cand = _create_mock_report(deflated_sharpe=0.10)
    res = accept(cand, baseline=None, require_dsr_non_degradation=True)
    assert res.accepted


def test_gate_dsr_non_degradation_via_config_dict() -> None:
    """Verifies DSR non-degradation gate is resolved correctly from a config dict."""
    # Use raw compare metric so improvement check passes, letting DSR gate fire
    config = {"require_dsr_non_degradation": True, "select_compare_metric": "raw", "select_improvement_tol": 0.0}
    base = _create_mock_report(sharpe=1.0, deflated_sharpe=0.80)
    cand = _create_mock_report(sharpe=1.1, deflated_sharpe=0.50)
    res = accept(cand, baseline=base, config=config)
    assert not res.accepted
    assert res.failed_gate == "dsr_non_degradation"


# ---------------------------------------------------------------------------
# New two-phase gate: select + confirm
# ---------------------------------------------------------------------------


def _create_split_report(
    in_dd: float = 0.10,
    in_turnover: float = 0.5,
    in_sharpe: float = 1.5,
    in_dsr: float = 0.98,
    ho_dd: float = 0.10,
    ho_turnover: float = 0.5,
    ho_sharpe: float = 1.5,
    ho_dsr: float = 0.98,
    regime_passed: bool = True,
) -> EvaluationReport:
    """Helper with distinct in-sample and holdout windows."""
    in_window = WindowReport(
        start_date="2020-01-01",
        end_date="2024-12-31",
        annualized_return=0.12,
        annualized_volatility=0.10,
        sharpe_ratio=in_sharpe,
        sortino_ratio=in_sharpe * 1.3,
        max_drawdown=in_dd,
        turnover=in_turnover,
        information_ratio=in_sharpe * 0.6,
    )
    ho_window = WindowReport(
        start_date="2025-01-01",
        end_date="2025-12-31",
        annualized_return=0.10,
        annualized_volatility=0.12,
        sharpe_ratio=ho_sharpe,
        sortino_ratio=ho_sharpe * 1.3,
        max_drawdown=ho_dd,
        turnover=ho_turnover,
        information_ratio=ho_sharpe * 0.6,
    )
    return EvaluationReport(
        strategy_name="split_strat",
        dataset_hash="def",
        gates_passed={},
        is_accepted=True,
        rejection_reason=None,
        holdout_metrics=ho_window,
        in_sample_metrics=in_window,
        walk_forward_metrics=[in_window],
        regime_drawdowns={},
        regime_passed=regime_passed,
        mc_sharpe_5th=0.5,
        mc_sharpe_50th=1.2,
        mc_sharpe_95th=2.0,
        observed_sharpe=in_sharpe,
        effective_trials=1,
        deflated_sharpe=in_dsr,
        holdout_deflated_sharpe=ho_dsr,
    )


def test_select_rejects_weak_in_sample_strong_holdout() -> None:
    """select rejects a candidate with poor in-sample metrics but good holdout."""
    # Strong on holdout (good Sharpe, low DD/turn) but weak in-sample
    cand = _create_split_report(in_sharpe=0.5, in_dd=0.35, ho_sharpe=2.0, ho_dd=0.05)
    base = _create_split_report(in_sharpe=1.0, in_dd=0.10)
    res = select(cand, baseline=base)
    assert not res.accepted
    assert res.failed_gate in ("max_drawdown", "target_metric_improvement")


def test_confirm_rejects_holdout_drawdown_breach() -> None:
    """confirm rejects a select-passing candidate that fails holdout drawdown."""
    # Strong in-sample, but holdout DD too high
    cand = _create_split_report(in_sharpe=2.0, in_dd=0.05, ho_dd=0.35)
    base = _create_split_report(in_sharpe=1.0, in_dd=0.10, ho_dd=0.05)
    # select passes (strong in-sample)
    sel = select(cand, baseline=base)
    assert sel.accepted
    # confirm fails (holdout DD)
    cnf = confirm(cand, baseline=base)
    assert not cnf.accepted
    assert cnf.failed_gate == "max_drawdown"


def test_confirm_rejects_holdout_dsr_degradation() -> None:
    """confirm rejects a select-passing candidate whose holdout DSR degrades."""
    cand = _create_split_report(
        in_sharpe=2.0,
        in_dsr=0.90,
        ho_sharpe=1.8,
        ho_dsr=0.30,
    )
    base = _create_split_report(
        in_sharpe=1.0,
        in_dsr=0.80,
        ho_sharpe=1.5,
        ho_dsr=0.70,
    )
    sel = select(cand, baseline=base)
    assert sel.accepted
    cnf = confirm(cand, baseline=base)
    assert not cnf.accepted
    assert cnf.failed_gate == "holdout_dsr_non_degradation"


def test_select_confirm_full_accept() -> None:
    """select + confirm accept a candidate strong on both axes."""
    cand = _create_split_report(in_sharpe=2.0, in_dd=0.05, ho_sharpe=1.8, ho_dd=0.05)
    base = _create_split_report(in_sharpe=1.0, in_dd=0.10, ho_sharpe=1.2, ho_dd=0.10)
    sel = select(cand, baseline=base)
    assert sel.accepted
    cnf = confirm(cand, baseline=base)
    assert cnf.accepted


def test_select_always_on_dsr() -> None:
    """select always enforces DSR non-degradation (no config needed)."""
    base = _create_split_report(in_sharpe=1.0, in_dsr=0.80)
    cand = _create_split_report(in_sharpe=2.0, in_dsr=0.40)
    # Use raw compare metric so improvement check passes, letting DSR gate fire
    res = select(cand, baseline=base, config={"select_compare_metric": "raw", "select_improvement_tol": 0.0})
    assert not res.accepted
    assert res.failed_gate == "dsr_non_degradation"


# ---------------------------------------------------------------------------
# Return floor gate (min_return_ratio)
# ---------------------------------------------------------------------------


def test_select_rejects_low_return_despite_high_sharpe() -> None:
    """Verifies candidate with high Sharpe but near-zero return is rejected by return floor."""
    base = _create_mock_report(sharpe=1.0, annualized_return=0.15)
    cand = _create_mock_report(sharpe=2.0, annualized_return=0.01)
    res = select(cand, baseline=base, min_return_ratio=0.5)
    assert not res.accepted
    assert res.failed_gate == "min_return_ratio"


def test_select_accepts_adequate_return() -> None:
    """Verifies candidate with adequate return passes the return floor gate."""
    base = _create_mock_report(sharpe=1.0, annualized_return=0.15)
    cand = _create_mock_report(sharpe=1.1, annualized_return=0.10)
    res = select(cand, baseline=base, min_return_ratio=0.5)
    assert res.accepted


def test_select_skips_return_check_no_baseline() -> None:
    """Verifies return floor is skipped when no baseline exists."""
    cand = _create_mock_report(sharpe=1.5, annualized_return=0.01)
    res = select(cand, baseline=None, min_return_ratio=0.5)
    assert res.accepted


def test_select_skips_return_check_negative_baseline_return() -> None:
    """Verifies return floor is skipped when baseline annualized return is negative."""
    base = _create_mock_report(sharpe=1.0, annualized_return=-0.05)
    cand = _create_mock_report(sharpe=1.1, annualized_return=0.01)
    res = select(cand, baseline=base, min_return_ratio=0.5)
    assert res.accepted


def test_select_rejects_nan_return() -> None:
    """Verifies NaN annualized return is rejected by return floor gate."""
    base = _create_mock_report(sharpe=1.0, annualized_return=0.15)
    cand = _create_mock_report(sharpe=1.5, annualized_return=float("nan"))
    res = select(cand, baseline=base, min_return_ratio=0.5)
    assert not res.accepted
    assert res.failed_gate == "min_return_ratio"


def test_select_resolves_min_return_ratio_from_config() -> None:
    """Verifies min_return_ratio is resolved from config dict."""
    base = _create_mock_report(sharpe=1.0, annualized_return=0.15)

    # Config with strict 80% threshold — candidate at 10% return fails vs 15% baseline
    cand = _create_mock_report(sharpe=1.1, annualized_return=0.10)
    config = {"select_min_return_ratio": 0.8}
    res = select(cand, baseline=base, config=config)
    assert not res.accepted
    assert res.failed_gate == "min_return_ratio"

    # Config with loose 50% threshold — candidate at 10% return passes vs 15% baseline
    config2 = {"select_min_return_ratio": 0.5}
    res2 = select(cand, baseline=base, config=config2)
    assert res2.accepted


def test_gate_pbo_constraints() -> None:
    # 1. Passes when pbo is below limit
    rep = _create_mock_report(pbo=0.3)
    res = accept(rep, baseline=None, pbo_limit=0.5)
    assert res.accepted
    assert rep.gates_passed.get("pbo") is True

    # 2. Rejects when pbo exceeds limit
    rep_high = _create_mock_report(pbo=0.8)
    res_high = accept(rep_high, baseline=None, pbo_limit=0.5)
    assert not res_high.accepted
    assert res_high.failed_gate == "pbo"
    assert rep_high.gates_passed.get("pbo") is False

    # 3. Disabled gate (no-op pass) when pbo_limit is None or report.pbo is None
    rep_none_limit = _create_mock_report(pbo=0.9)
    res_none_limit = accept(rep_none_limit, baseline=None, pbo_limit=None)
    assert res_none_limit.accepted
    assert rep_none_limit.gates_passed.get("pbo") is True

    rep_none_pbo = _create_mock_report(pbo=None)
    res_none_pbo = accept(rep_none_pbo, baseline=None, pbo_limit=0.5)
    assert res_none_pbo.accepted
    assert rep_none_pbo.gates_passed.get("pbo") is True

    # 4. Reject NaN PBO
    rep_nan_pbo = _create_mock_report(pbo=float("nan"))
    res_nan_pbo = accept(rep_nan_pbo, baseline=None, pbo_limit=0.5)
    assert not res_nan_pbo.accepted
    assert res_nan_pbo.failed_gate == "pbo"
    assert rep_nan_pbo.gates_passed.get("pbo") is False

    # 5. Ordering test (turnover failure wins over pbo)
    rep_both_fail = _create_mock_report(turnover=2.5, pbo=0.8)
    res_ordering = accept(rep_both_fail, baseline=None, turnover_limit=1.0, pbo_limit=0.5)
    assert not res_ordering.accepted
    assert res_ordering.failed_gate == "turnover"  # turnover comes first lexicographically


def test_gate_hybrid_tradeoff() -> None:
    """Verifies that the gate allows target metric degradation in exchange for return."""
    base = _create_mock_report(sharpe=1.14, annualized_return=0.10)
    cand = _create_mock_report(sharpe=0.70, annualized_return=0.15)

    # 1. With no tradeoff configured -> Fails (standard strict check)
    config_none = {"metric_return_tradeoff": 0.0, "select_compare_metric": "raw", "select_improvement_tol": 0.0}
    res_none = accept(cand, baseline=base, config=config_none)
    assert not res_none.accepted
    assert res_none.failed_gate == "target_metric_improvement"

    # 2. With tradeoff configured -> Passes (hurdle is 1.14 - 0.1 * 5pp * 100 = 0.64 < 0.70)
    config_tradeoff = {"metric_return_tradeoff": 0.1, "select_compare_metric": "raw", "select_improvement_tol": 0.0}
    res_tradeoff = accept(cand, baseline=base, config=config_tradeoff)
    assert res_tradeoff.accepted


def test_gate_hybrid_floor() -> None:
    """Verifies that the hybrid gate enforces the absolute metric floor constraint."""
    base = _create_mock_report(sharpe=1.14, annualized_return=0.10)
    cand = _create_mock_report(sharpe=0.70, annualized_return=0.15)

    # Pin to raw mode so floor/sharpe are in the same unit (Sharpe).
    raw = {"select_compare_metric": "raw", "select_improvement_tol": 0.0}

    # With tradeoff = 0.1 and floor = 0.80 -> Fails due to absolute floor breach
    config_floor_fail = {**raw, "metric_return_tradeoff": 0.1, "metric_floor": 0.80}
    res_floor_fail = accept(cand, baseline=base, config=config_floor_fail)
    assert not res_floor_fail.accepted
    assert res_floor_fail.failed_gate == "metric_floor"

    # With tradeoff = 0.1 and floor = 0.60 -> Passes (floor is 0.60, hurdle is 0.64)
    config_floor_pass = {**raw, "metric_return_tradeoff": 0.1, "metric_floor": 0.60}
    res_floor_pass = accept(cand, baseline=base, config=config_floor_pass)
    assert res_floor_pass.accepted


def test_gate_tradeoff_negative_return_diff() -> None:
    """Verifies tradeoff raises the hurdle when candidate returns are below baseline."""
    base = _create_mock_report(sharpe=1.14, annualized_return=0.10)
    cand = _create_mock_report(sharpe=0.80, annualized_return=0.08)

    # hurdle = 1.14 + 0.0 - 0.1 * (-0.02) * 100 = 1.14 + 0.2 = 1.34
    config = {"metric_return_tradeoff": 0.1}
    res = accept(cand, baseline=base, config=config)
    assert not res.accepted
    assert res.failed_gate == "target_metric_improvement"


def test_gate_tradeoff_nan_return() -> None:
    """Verifies NaN annualized return is rejected when tradeoff is active."""
    base = _create_mock_report(sharpe=1.0, annualized_return=0.15)
    cand = _create_mock_report(sharpe=1.5, annualized_return=float("nan"))

    config = {"metric_return_tradeoff": 0.1}
    res = accept(cand, baseline=base, config=config)
    assert not res.accepted
    assert res.failed_gate == "target_metric_improvement"


def test_gate_tradeoff_boundary_hurdle() -> None:
    """Verifies candidate at exactly the hurdle value is rejected."""
    base = _create_mock_report(sharpe=2.0, annualized_return=0.10)
    cand = _create_mock_report(sharpe=1.50, annualized_return=0.11)

    # hurdle = 2.0 + 0.0 - 0.5 * (0.11 - 0.10) * 100 = 2.0 - 0.5 = 1.50
    config = {"metric_return_tradeoff": 0.5, "select_compare_metric": "raw", "select_improvement_tol": 0.0}
    res = accept(cand, baseline=base, config=config)
    assert not res.accepted
    assert res.failed_gate == "target_metric_improvement"


def test_gate_tradeoff_with_sortino() -> None:
    """Verifies tradeoff works with non-Sharpe target metrics."""
    base = _create_mock_report(sortino=2.0, annualized_return=0.10)
    cand = _create_mock_report(sortino=1.20, annualized_return=0.15)

    # No tradeoff -> Fails (pin to raw comparison to use Sortino values)
    res_none = accept(
        cand,
        baseline=base,
        target_metric=TargetMetric.SORTINO,
        config={"select_compare_metric": "raw", "select_improvement_tol": 0.0},
    )
    assert not res_none.accepted

    # With tradeoff -> Passes (hurdle = 2.0 - 0.1 * 0.05 * 100 = 1.50, cand 1.20 < 1.50)
    # Actually 1.20 < 1.50 so still fails. Let's use stronger tradeoff.
    config = {"metric_return_tradeoff": 0.5, "select_compare_metric": "raw", "select_improvement_tol": 0.0}
    res_trade = accept(cand, baseline=base, target_metric=TargetMetric.SORTINO, config=config)
    assert res_trade.accepted


def test_gate_floor_without_tradeoff() -> None:
    """Verifies metric_floor is enforced independently of the tradeoff (raw mode)."""
    base = _create_mock_report(sharpe=1.50, annualized_return=0.10)
    # Pin to raw mode: floor value 1.60 is in Sharpe units (not DSR).
    raw = {"select_compare_metric": "raw", "select_improvement_tol": 0.0}

    # Candidate exceeds baseline but is below absolute floor -> rejected by floor
    cand_fail = _create_mock_report(sharpe=1.55, annualized_return=0.12)
    res_fail = accept(cand_fail, baseline=base, config={**raw, "metric_floor": 1.60})
    assert not res_fail.accepted
    assert res_fail.failed_gate == "metric_floor"

    # Candidate exceeds both baseline and floor -> accepted
    cand_pass = _create_mock_report(sharpe=1.70, annualized_return=0.12)
    res_pass = accept(cand_pass, baseline=base, config={**raw, "metric_floor": 1.60})
    assert res_pass.accepted


def test_gate_tradeoff_default_parity() -> None:
    """Verifies raw compare mode (tradeoff=0, floor=None, tol=0) gives pre-hybrid code behavior."""
    base = _create_mock_report(sharpe=1.2, annualized_return=0.10)
    cand = _create_mock_report(sharpe=1.1, annualized_return=0.12)
    raw_config = {"select_compare_metric": "raw", "select_improvement_tol": 0.0}

    res = accept(cand, baseline=base, config=raw_config)
    assert not res.accepted
    assert res.failed_gate == "target_metric_improvement"

    # Better candidate passes
    cand2 = _create_mock_report(sharpe=1.3, annualized_return=0.08)
    res2 = accept(cand2, baseline=base, config=raw_config)
    assert res2.accepted


# ---------------------------------------------------------------------------
# Recalibration: DSR-based comparison (deflated mode default)
# ---------------------------------------------------------------------------


def test_select_compare_metric_deflated_default():
    """Default mode compares on DSR, not raw Sharpe."""
    # Baseline: high raw Sharpe (overfit) but modest DSR
    base = _create_mock_report(sharpe=4.0, deflated_sharpe=0.60)
    # Candidate: lower raw Sharpe but similar DSR — should PASS under deflated mode
    cand = _create_mock_report(sharpe=1.5, deflated_sharpe=0.65)
    res = select(cand, baseline=base, config={})  # default: compare_metric="deflated"
    assert res.accepted, f"Expected pass under DSR mode but got: {res.reason}"


def test_select_compare_metric_raw_explicit():
    """Raw mode compares on in-sample Sharpe."""
    base = _create_mock_report(sharpe=4.0, deflated_sharpe=0.60)
    cand = _create_mock_report(sharpe=1.5, deflated_sharpe=0.65)
    res = select(cand, baseline=base, config={"select_compare_metric": "raw", "select_improvement_tol": 0.0})
    assert not res.accepted
    assert res.failed_gate == "target_metric_improvement"


def test_select_improvement_tol_rescues_near_tie():
    """Candidate with DSR within tolerance band passes in default deflated mode if non-degradation is disabled."""
    base = _create_mock_report(deflated_sharpe=0.80)
    # Candidate is 0.01 below baseline DSR — within 0.02 tolerance.
    cand = _create_mock_report(deflated_sharpe=0.79)
    res = select(cand, baseline=base, config={"require_dsr_non_degradation": False})
    assert res.accepted, f"Expected near-tie to pass but got: {res.reason}"


def test_select_improvement_tol_still_rejects_large_gap():
    """Candidate well below baseline DSR still fails in default deflated mode."""
    base = _create_mock_report(deflated_sharpe=0.80)
    # Candidate is 0.10 below — outside 0.02 tolerance.
    cand = _create_mock_report(deflated_sharpe=0.70)
    res = select(cand, baseline=base, config={})
    assert not res.accepted
    assert res.failed_gate == "target_metric_improvement"


# ---------------------------------------------------------------------------
# Recalibration: Regime failure message contains breach details
# ---------------------------------------------------------------------------


def test_regime_message_contains_breach_details():
    """Regime failure message should name the breached regime and show observed vs limit."""
    import dataclasses

    # Set up a candidate that breaches 2008_GFC (limit 0.25) with 0.35 drawdown
    rep = _create_mock_report(regime_passed=False)
    rep = dataclasses.replace(rep, regime_drawdowns={"2008_GFC": 0.35, "2020_COVID": 0.10})
    res = select(rep, baseline=None)
    assert not res.accepted
    assert res.failed_gate == "regimes"
    assert "2008_GFC" in res.reason
    assert "35.0%" in res.reason or "0.35" in res.reason.replace(",", ".")
    assert "25.0%" in res.reason or "0.25" in res.reason.replace(",", ".")


def test_regime_message_no_overlap():
    """Regime failure with no regime_drawdowns should mention window overlap."""
    import dataclasses

    rep = _create_mock_report(regime_passed=False)
    rep = dataclasses.replace(rep, regime_drawdowns={})
    res = select(rep, baseline=None)
    assert not res.accepted
    assert "does not overlap" in res.reason or "backtest window" in res.reason


def test_metric_floor_deflated_mode_uses_raw_scale():
    """In deflated mode (default) metric_floor is compared against raw Sharpe, not DSR."""
    # Candidate has raw Sharpe=2.0 and DSR=0.50
    cand = _create_mock_report(sharpe=2.0, deflated_sharpe=0.50)
    # Floor of 2.20 in Sharpe units — candidate should fail
    res_fail = select(cand, baseline=None, config={"metric_floor": 2.20})
    assert not res_fail.accepted
    assert res_fail.failed_gate == "metric_floor"
    assert "sharpe" in res_fail.reason

    # Floor of 1.80 in Sharpe units — candidate should pass (2.0 > 1.80)
    res_pass = select(cand, baseline=None, config={"metric_floor": 1.80})
    assert res_pass.accepted


def test_dsr_non_degradation_enforced_in_deflated_mode_by_default():
    """In deflated mode, DSR non-degradation gate is enforced by default."""
    base = _create_mock_report(deflated_sharpe=0.85)
    cand = _create_mock_report(deflated_sharpe=0.84)
    res = select(cand, baseline=base, config={})  # default deflated mode, require_dsr_non_degradation=True
    assert not res.accepted
    assert res.failed_gate == "dsr_non_degradation"


def test_dsr_non_degradation_can_be_disabled_in_deflated_mode():
    """In deflated mode, DSR non-degradation gate is skipped if explicitly disabled."""
    base = _create_mock_report(deflated_sharpe=0.85)
    cand = _create_mock_report(deflated_sharpe=0.84)
    res = select(cand, baseline=base, config={"require_dsr_non_degradation": False})  # default deflated mode
    assert res.accepted, f"Expected pass (DSR within tolerance) but got: {res.reason}"


def test_dsr_non_degradation_nan_candidate_rejected():
    """NaN candidate DSR is rejected by non-degradation gate (raw mode)."""
    # Base has lower raw Sharpe so the improvement gate passes, then DSR gate fires.
    base = _create_mock_report(sharpe=1.0, deflated_sharpe=0.80)
    cand = _create_mock_report(sharpe=1.5, deflated_sharpe=float("nan"))
    # Use raw mode so non-degradation gate is active
    res = select(cand, baseline=base, config={"select_compare_metric": "raw", "select_improvement_tol": 0.0})
    assert not res.accepted
    assert res.failed_gate == "dsr_non_degradation"


def test_dsr_non_degradation_nan_baseline_passes():
    """NaN baseline DSR causes non-degradation gate to be skipped (raw mode)."""
    # Base has lower raw Sharpe so the improvement gate passes, then DSR gate fires.
    base = _create_mock_report(sharpe=1.0, deflated_sharpe=float("nan"))
    cand = _create_mock_report(sharpe=1.5, deflated_sharpe=0.80)
    # Use raw mode; NaN baseline → no DSR floor to enforce → gate is skipped → accepted
    res = select(cand, baseline=base, config={"select_compare_metric": "raw", "select_improvement_tol": 0.0})
    assert res.accepted


def test_gate_hybrid_tradeoff_deflated_mode() -> None:
    """Verifies return tradeoff is scaled to DSR units in deflated mode."""
    # Baseline: DSR=0.50, raw Sharpe=2.0 (scale = 0.25). Return = 0.10.
    base = _create_mock_report(sharpe=2.0, deflated_sharpe=0.50, annualized_return=0.10)

    # 1. Candidate DSR=0.20. Return = 0.12 (+2pp).
    # Tradeoff coeff = 0.5.
    # Raw tradeoff term = 0.5 * 2.0 = 1.0.
    # Scaled tradeoff term = 1.0 * 0.25 = 0.25.
    # Hurdle = 0.50 - 0.25 = 0.25.
    # Candidate DSR is 0.20 (< 0.25 hurdle) -> fails
    cand_fail = _create_mock_report(sharpe=1.8, deflated_sharpe=0.20, annualized_return=0.12)
    config_tradeoff = {
        "metric_return_tradeoff": 0.5,
        "select_compare_metric": "deflated",
        "select_improvement_tol": 0.0,
        "require_dsr_non_degradation": False,
    }
    res_fail = select(cand_fail, baseline=base, config=config_tradeoff)
    assert not res_fail.accepted
    assert res_fail.failed_gate == "target_metric_improvement"

    # 2. Candidate DSR=0.30. Return = 0.12 (+2pp).
    # Candidate DSR is 0.30 (> 0.25 hurdle) -> passes
    cand_pass = _create_mock_report(sharpe=1.8, deflated_sharpe=0.30, annualized_return=0.12)
    res_pass = select(cand_pass, baseline=base, config=config_tradeoff)
    assert res_pass.accepted


def test_accept_forwards_parameters_and_resolves_epsilon() -> None:
    """Verifies that accept() correctly forwards holdout_min_improvement and dsr_non_degradation_epsilon."""
    base = _create_mock_report()
    base.holdout_deflated_sharpe = 0.85
    base.deflated_sharpe = 0.85

    cand = _create_mock_report()
    cand.holdout_deflated_sharpe = 0.80
    cand.deflated_sharpe = 0.80

    # 1. Default accept (fails due to degradation: cand DSR 0.80 < base DSR 0.85)
    # We pass select_improvement_tol=0.10 to allow target_metric_improvement to pass.
    res_fail = accept(cand, baseline=base, config={"select_improvement_tol": 0.10})
    assert not res_fail.accepted
    assert res_fail.failed_gate in ("dsr_non_degradation", "holdout_dsr_non_degradation")

    # 2. Set holdout_min_improvement=0.06 and select_improvement_tol=0.06 via config,
    # and disable in-sample degradation to test holdout
    config = {
        "select_improvement_tol": 0.06,
        "require_dsr_non_degradation": False,
    }
    # With holdout_min_improvement=0.06:
    # cand holdout DSR 0.80 >= base holdout DSR 0.85 - 1e-6 - 0.06 = 0.789999 -> should pass
    res_pass = accept(
        cand,
        baseline=base,
        holdout_min_improvement=0.06,
        config=config,
    )
    assert res_pass.accepted

    # 3. Test dsr_non_degradation_epsilon resolution
    # cand in-sample DSR 0.80, base in-sample DSR 0.85. Degradation is 0.05.
    # If we set dsr_non_degradation_epsilon=0.06 and select_improvement_tol=0.06, in-sample DSR passes.
    # Holdout: cand holdout DSR 0.80, base holdout DSR 0.85. With epsilon=0.06 and holdout_min_improvement=0.0,
    # cand holdout DSR 0.80 >= base holdout DSR 0.85 - 0.06 = 0.79 -> passes.
    config_eps = {
        "select_improvement_tol": 0.06,
        "require_dsr_non_degradation": True,
    }
    res_eps = accept(
        cand,
        baseline=base,
        dsr_non_degradation_epsilon=0.06,
        holdout_min_improvement=0.0,
        config=config_eps,
    )
    assert res_eps.accepted
