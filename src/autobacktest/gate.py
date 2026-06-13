"""Two-phase strategy improvement gate (select + confirm).

The gate system is split into two distinct phases to prevent holdout
overfitting:

- **select** — in-sample walk-forward aggregate checks (drawdown, turnover,
  regime stress, PBO, target-metric improvement, DSR non-degradation).
  Evaluated on **every** candidate.
- **confirm** — holdout checks (drawdown, turnover, DSR non-degradation).
  Only reached when ``select`` passes.  Each call consumes one holdout
  peek (budgeted).

A backward-compatible ``accept()`` wrapper composes both gates for the
standalone evaluation path.

Edge cases handled:
- NaN / Inf in any metric causes a hard rejection with a descriptive reason.
- Config resolution follows a priority chain: explicit arg → Pydantic
  model attribute → dict key → ``params`` sub-dict → schema default.
"""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from autobacktest.evaluator.regime import REGIMES
from autobacktest.evaluator.report import EvaluationReport


class TargetMetric(StrEnum):
    """Optimization objective metric used as the primary gating criterion.

    Members:
        SHARPE: Annualised Sharpe ratio (default).
        SORTINO: Sortino ratio (downside deviation only).
        INFORMATION_RATIO: Active return over tracking error vs. benchmark.
    """

    SHARPE = "sharpe"
    SORTINO = "sortino"
    INFORMATION_RATIO = "information_ratio"


@dataclass
class GateResult:
    """The outcome result of a gate evaluation (select, confirm, or accept).

    Attributes:
        accepted: Whether the candidate passed the gate.
        reason: Human-readable explanation of the outcome.
        failed_gate: Identifier of the specific gate check that failed
            (e.g. ``"dd_limit"``, ``"turnover_limit"``, ``"dsr_degradation"``).
    """

    accepted: bool
    reason: str | None = None
    failed_gate: str | None = None


def _get_config_val(config: Any, key: str, default: Any) -> Any:
    """Safely retrieve a configuration value from a Pydantic model or dict.

    Resolution priority:
    1. Pydantic model attribute (via ``getattr``).
    2. Top-level dict key.
    3. ``params`` sub-dict key.
    4. Fallback to ``default``.

    Args:
        config: A ``StrategyConfig`` instance, a flat dict, or ``None``.
        key: The configuration key to look up.
        default: Value returned when *config* is ``None`` or *key* is not found.

    Returns:
        The resolved value or *default*.
    """
    if config is None:
        return default
    if hasattr(config, key):
        return getattr(config, key)
    if isinstance(config, dict):
        if key in config:
            return config[key]
        params = config.get("params", {})
        if isinstance(params, dict) and key in params:
            return params[key]
    return default


def _get_in_sample_metric_val(report: EvaluationReport, target_metric: TargetMetric) -> float:
    """Extract target metric from a report's in-sample walk-forward aggregate.

    Infinity and NaN are capped to 0.0 so that a "risk-free" strategy
    (e.g. all-non-negative-returns producing +inf Sortino) does not
    dominate or derail the lexicographic gate.

    Args:
        report: Candidate evaluation report.
        target_metric: Which metric to extract (Sharpe, Sortino, or IR).

    Returns:
        The metric value, capped to 0.0 for NaN/Inf.
    """
    if target_metric == TargetMetric.SHARPE:
        val = report.in_sample_metrics.sharpe_ratio
    elif target_metric == TargetMetric.SORTINO:
        val = report.in_sample_metrics.sortino_ratio
    elif target_metric == TargetMetric.INFORMATION_RATIO:
        val = report.in_sample_metrics.information_ratio or 0.0
    else:
        raise ValueError(f"Unsupported target metric choice: {target_metric}")

    if math.isnan(val) or math.isinf(val):
        return 0.0
    return val


def _get_compare_metric_val(report: EvaluationReport, target_metric: TargetMetric, compare_metric: str) -> float:
    """Return the value used by the select-gate improvement comparison.

    ``"deflated"`` → ``report.deflated_sharpe`` (DSR, robust to overfitting).
    ``"raw"`` → in-sample target metric (legacy Sharpe/Sortino/IR).
    NaN/Inf are capped to 0.0 for consistent comparison.

    Args:
        report: Evaluation report to extract the metric from.
        target_metric: Metric choice (used only when ``compare_metric`` is ``"raw"``).
        compare_metric: ``"deflated"`` for DSR or ``"raw"`` for in-sample metric.

    Returns:
        The comparison metric value, capped to 0.0 for NaN/Inf.
    """
    if compare_metric == "deflated" and target_metric == TargetMetric.SHARPE:
        val = report.deflated_sharpe
    else:
        return _get_in_sample_metric_val(report, target_metric)
    if math.isnan(val) or math.isinf(val):
        return 0.0
    return val


def _write_gates_passed(report: EvaluationReport, checks: dict[str, bool]) -> None:
    """Write gate outcomes to the report's ``gates_passed`` dict.

    Args:
        report: Evaluation report to update.
        checks: Mapping of gate name to pass/fail boolean.
    """
    report.gates_passed.update(checks)


# ---------------------------------------------------------------------------
# Selection gate (in-sample walk-forward aggregate)
# ---------------------------------------------------------------------------


def select(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    target_metric: TargetMetric = TargetMetric.SHARPE,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    min_improvement: float | None = None,
    require_dsr_non_degradation: bool | None = None,
    min_return_ratio: float | None = None,
    pbo_limit: float | None = None,
    config: Any = None,
) -> GateResult:
    """In-sample selection gate — evaluated on every candidate.

    Hard constraints (in-sample walk-forward aggregate):
    1. Drawdown: ``in_sample_metrics.max_drawdown <= dd_limit``
    2. Regime tests: ``regime_passed`` is True
    3. Turnover: ``in_sample_metrics.turnover <= turnover_limit``
    4. PBO limit: ``report.pbo <= pbo_limit`` (if set)

    Unconditional soft check (always, no baseline needed):
    5. Absolute metric floor: ``candidate_val > metric_floor``
       (config key ``metric_floor``, default ``None`` = disabled).

    Tie-breaker (only when baseline is present):
    6. Target metric improvement (config key ``min_improvement``),
       optionally adjusted by ``metric_return_tradeoff``:
       ``candidate > baseline + min_improvement - tradeoff_coeff * (cand_ret - base_ret) * 100``
       where ``tradeoff_coeff`` is per-1pp (0.01) return increase.
       The metric used for comparison is determined by ``select_compare_metric``:
       ``"deflated"`` uses DSR (overfit-adjusted), ``"raw"`` uses the in-sample
       target metric directly.
    7. Near-tie tolerance: candidate accepted when metric >= incumbent - ``select_improvement_tol``
       (config key ``select_improvement_tol``, default ``0.02``).
    8. Annualized return must be at least ``min_return_ratio`` of baseline return.
    9. DSR non-degradation: always-on when baseline is present
       (config can disable via ``require_dsr_non_degradation: false``).

    Args:
        report: Evaluation report for the candidate strategy.
        baseline: Evaluation report for the current incumbent strategy, or ``None``
            for the first iteration (skips baseline-dependent checks).
        target_metric: Metric choice for improvement comparison (default SHARPE).
        dd_limit: Max drawdown threshold. Resolved from config if ``None``.
        turnover_limit: Max turnover threshold. Resolved from config if ``None``.
        min_improvement: Minimum metric improvement over baseline. Resolved from config if ``None``.
        require_dsr_non_degradation: Enforce DSR non-degradation. Resolved from config if ``None``.
        min_return_ratio: Minimum fraction of baseline return. Resolved from config if ``None``.
        pbo_limit: Maximum PBO threshold. Resolved from config if ``None``.
        config: StrategyConfig or dict providing fallback values for unresolved parameters.

    Returns:
        GateResult: Decision outcome. The holdout is **never** consulted.
    """
    (
        dd_limit,
        turnover_limit,
        pbo_limit,
        min_improvement,
        min_return_ratio,
        require_dsr,
        tradeoff,
        floor,
        compare_metric,
        improvement_tol,
    ) = _resolve_select_config(
        config,
        dd_limit,
        turnover_limit,
        pbo_limit,
        min_improvement,
        min_return_ratio,
        require_dsr_non_degradation,
    )

    result = _check_hard_gates(report, dd_limit, turnover_limit, pbo_limit)
    if not result.accepted:
        return result

    result = _check_soft_gates(
        report,
        baseline,
        target_metric,
        min_improvement,
        min_return_ratio,
        require_dsr,
        tradeoff,
        floor,
        compare_metric,
        improvement_tol,
    )
    if not result.accepted:
        return result

    return GateResult(accepted=True)


def _resolve_select_config(
    config: Any,
    dd_limit: float | None,
    turnover_limit: float | None,
    pbo_limit: float | None,
    min_improvement: float | None,
    min_return_ratio: float | None,
    require_dsr_non_degradation: bool | None,
) -> tuple[float, float, float | None, float, float, bool, float, float | None, str, float]:
    """Resolve select gate parameters from explicit args or config object.

    Resolution priority: explicit argument > config object attribute/key > schema default.

    Args:
        config: StrategyConfig or dict providing fallback values, or ``None``.
        dd_limit: Explicit drawdown limit override, or ``None`` for config default.
        turnover_limit: Explicit turnover limit override, or ``None`` for config default.
        pbo_limit: Explicit PBO limit override, or ``None`` for config default.
        min_improvement: Explicit min improvement override, or ``None`` for config default.
        min_return_ratio: Explicit min return ratio override, or ``None`` for config default.
        require_dsr_non_degradation: Explicit DSR requirement override, or ``None`` for config default.

    Returns:
        10-tuple of resolved values: (dd_limit, turnover_limit, pbo_limit,
        min_improvement, min_return_ratio, require_dsr, tradeoff, floor,
        compare_metric, improvement_tol).
    """
    dd_limit = dd_limit if dd_limit is not None else _get_config_val(config, "max_drawdown_limit", 0.20)
    turnover_limit = turnover_limit if turnover_limit is not None else _get_config_val(config, "turnover_limit", 2.0)
    pbo_limit = pbo_limit if pbo_limit is not None else _get_config_val(config, "pbo_limit", None)
    min_improvement = (
        min_improvement if min_improvement is not None else _get_config_val(config, "min_improvement", 0.0)
    )
    min_return_ratio = (
        min_return_ratio if min_return_ratio is not None else _get_config_val(config, "select_min_return_ratio", 0.5)
    )
    if require_dsr_non_degradation is not None:
        require_dsr = require_dsr_non_degradation
    else:
        require_dsr = _get_config_val(config, "require_dsr_non_degradation", True)
    tradeoff = _get_config_val(config, "metric_return_tradeoff", 0.0)
    floor = _get_config_val(config, "metric_floor", None)
    compare_metric = _get_config_val(config, "select_compare_metric", "deflated")
    improvement_tol = _get_config_val(config, "select_improvement_tol", 0.02)
    return (
        dd_limit,
        turnover_limit,
        pbo_limit,
        min_improvement,
        min_return_ratio,
        require_dsr,
        tradeoff,
        floor,
        compare_metric,
        improvement_tol,
    )


def _check_hard_gates(
    report: EvaluationReport,
    dd_limit: float,
    turnover_limit: float,
    pbo_limit: float | None,
) -> GateResult:
    """Check hard constraints: drawdown, regime, turnover, PBO.

    These checks are evaluated first and cause immediate rejection when failed.
    NaN values in any metric are treated as gate failures.

    Args:
        report: Candidate evaluation report.
        dd_limit: Maximum allowed in-sample max drawdown.
        turnover_limit: Maximum allowed in-sample annualized turnover.
        pbo_limit: Maximum allowed Probability of Backtest Overfitting, or ``None``
            to skip the PBO check.

    Returns:
        GateResult: Decision outcome. If rejected, ``failed_gate`` indicates which
        check failed (``"max_drawdown"``, ``"regimes"``, ``"turnover"``, or ``"pbo"``).
    """
    max_dd = report.in_sample_metrics.max_drawdown
    max_dd_passed = not (math.isnan(max_dd) or max_dd > dd_limit)

    regime_passed = bool(report.regime_passed)

    turnover = report.in_sample_metrics.turnover
    turnover_passed = not (math.isnan(turnover) or turnover > turnover_limit)

    pbo_passed = True
    if pbo_limit is not None and report.pbo is not None:
        pbo_passed = not (math.isnan(report.pbo) or report.pbo > pbo_limit)

    _write_gates_passed(
        report,
        {"max_drawdown": max_dd_passed, "turnover": turnover_passed, "regimes": regime_passed, "pbo": pbo_passed},
    )

    if not max_dd_passed:
        if math.isnan(max_dd):
            msg = "In-sample max drawdown is NaN."
        else:
            msg = f"In-sample max drawdown {max_dd:.4f} exceeds limit of {dd_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="max_drawdown")

    if not regime_passed:
        breaches = []
        for name, (_s, _e, limit) in REGIMES.items():
            dd = report.regime_drawdowns.get(name)
            if dd is not None and dd > limit:
                breaches.append(f"{name}: drawdown {dd:.1%} exceeds {limit:.1%} limit")
        if breaches:
            msg = "Failed crisis-regime stress test — " + "; ".join(breaches) + "."
        elif not report.regime_drawdowns or all(v == 0.0 for v in report.regime_drawdowns.values()):
            msg = (
                "Failed crisis-regime stress test: backtest window does not overlap "
                "any crisis regime (2008 GFC / 2020 COVID / 2022 bear). "
                "Extend the backtest start date earlier to establish robustness."
            )
        else:
            msg = (
                "Failed crisis-regime stress test: sustained low market exposure "
                "(>80% cash) during a crisis window. Reduce de-risking duration."
            )
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="regimes")

    if not turnover_passed:
        if math.isnan(turnover):
            msg = "In-sample turnover is NaN."
        else:
            msg = f"In-sample turnover {turnover:.4f} exceeds limit of {turnover_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="turnover")

    if not pbo_passed:
        msg = f"In-sample PBO {report.pbo:.4f} exceeds limit of {pbo_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="pbo")

    return GateResult(accepted=True)


def _check_soft_gates(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    target_metric: TargetMetric,
    min_improvement: float,
    min_return_ratio: float,
    require_dsr: bool,
    tradeoff_coeff: float = 0.0,
    metric_floor: float | None = None,
    compare_metric: str = "deflated",
    improvement_tol: float = 0.0,
) -> GateResult:
    """Check soft constraints: metric improvement, return floor, DSR non-degradation.

    Evaluation order:
    1. Absolute ``metric_floor`` — enforced unconditionally (no baseline required).
    2. Metric improvement with optional return tradeoff (requires baseline).
    3. Return ratio floor (requires baseline).
    4. DSR non-degradation (requires baseline, controlled by ``require_dsr``).

    Args:
        report: Candidate evaluation report.
        baseline: Incumbent evaluation report, or ``None`` for first iteration.
        target_metric: Metric choice for improvement comparison.
        min_improvement: Minimum metric improvement over baseline.
        min_return_ratio: Minimum fraction of baseline annualized return.
        require_dsr: Whether to enforce DSR non-degradation.
        tradeoff_coeff: Metric reduction tolerated per 1pp return increase (``0.0`` disables).
        metric_floor: Absolute metric floor below which candidates are always rejected.
        compare_metric: ``"deflated"`` for DSR or ``"raw"`` for in-sample metric.
        improvement_tol: Near-tie tolerance — candidate accepted when metric >= incumbent - tol.

    Returns:
        GateResult: Decision outcome.
    """
    if metric_floor is not None:
        candidate_val = _get_in_sample_metric_val(report, target_metric)
        if math.isnan(candidate_val) or candidate_val <= metric_floor:
            basis = target_metric.value
            msg = f"Candidate {basis} ({candidate_val:.4f}) is below the required floor of {metric_floor:.4f}."
            report.is_accepted = False
            report.rejection_reason = msg
            return GateResult(accepted=False, reason=msg, failed_gate="metric_floor")

    if baseline is not None:
        result = _check_metric_improvement(
            report,
            baseline,
            target_metric,
            min_improvement,
            tradeoff_coeff,
            compare_metric=compare_metric,
            improvement_tol=improvement_tol,
        )
        if not result.accepted:
            return result

        result = _check_return_floor(report, baseline, min_return_ratio)
        if not result.accepted:
            return result

    if require_dsr and baseline is not None:
        result = _check_dsr_non_degradation(report, baseline)
        if not result.accepted:
            return result

    return GateResult(accepted=True)


def _check_metric_improvement(
    report: EvaluationReport,
    baseline: EvaluationReport,
    target_metric: TargetMetric,
    min_improvement: float,
    tradeoff_coeff: float = 0.0,
    compare_metric: str = "deflated",
    improvement_tol: float = 0.0,
) -> GateResult:
    """Check candidate metric improves over baseline, incorporating trade-off.

    The absolute ``metric_floor`` is NOT checked here — it is enforced by the
    caller (``_check_soft_gates``) before this function is reached.

    When ``compare_metric`` is ``"deflated"``, the Deflated Sharpe Ratio (DSR)
    is used for comparison (robust to overfitting). When ``"raw"``, the in-sample
    target metric (Sharpe/Sortino/IR) is used directly.

    The ``improvement_tol`` parameter allows near-tie acceptance: the candidate
    is accepted when its metric >= baseline metric + min_improvement - improvement_tol.

    When ``tradeoff_coeff > 0.0``, the required metric threshold is reduced by
    ``tradeoff_coeff * (candidate_return - baseline_return) * 100``, allowing
    candidates with higher returns to pass with lower metric improvement.

    Args:
        report: Candidate evaluation report.
        baseline: Incumbent evaluation report.
        target_metric: Metric choice for comparison (when compare_metric is ``"raw"``).
        min_improvement: Minimum metric improvement over baseline.
        tradeoff_coeff: Metric reduction tolerated per 1pp return increase.
        compare_metric: ``"deflated"`` for DSR or ``"raw"`` for in-sample metric.
        improvement_tol: Near-tie tolerance for acceptance.

    Returns:
        GateResult: Decision outcome.
    """
    candidate_val = _get_compare_metric_val(report, target_metric, compare_metric)
    baseline_val = _get_compare_metric_val(baseline, target_metric, compare_metric)

    cand_ret = report.in_sample_metrics.annualized_return
    base_ret = baseline.in_sample_metrics.annualized_return

    if tradeoff_coeff > 0.0 and (math.isnan(cand_ret) or math.isnan(base_ret)):
        msg = f"Candidate or baseline annualized return is NaN while metric_return_tradeoff={tradeoff_coeff} is active."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="target_metric_improvement")

    required_metric = baseline_val + min_improvement - improvement_tol
    if tradeoff_coeff > 0.0:
        tradeoff_term = tradeoff_coeff * (cand_ret - base_ret) * 100
        if compare_metric == "deflated":
            base_raw = baseline.observed_sharpe
            base_dsr = baseline.deflated_sharpe
            scale = max(0.0, min(1.0, base_dsr / base_raw)) if base_raw > 1e-6 else 0.0
            tradeoff_term *= scale
        required_metric -= tradeoff_term

    if math.isnan(candidate_val) or math.isnan(baseline_val) or candidate_val <= required_metric:
        basis = "DSR" if compare_metric == "deflated" else target_metric.value
        msg = (
            f"Candidate in-sample {basis} ({candidate_val:.4f}) does not "
            f"improve upon baseline in-sample {basis} ({baseline_val:.4f}) "
            f"by at least {min_improvement:.4f} (tolerance {improvement_tol:.4f})."
        )
        if tradeoff_coeff > 0.0:
            msg = (
                f"Candidate {basis} ({candidate_val:.4f}) is below "
                f"the adjusted hurdle of {required_metric:.4f} "
                f"(baseline: {baseline_val:.4f}, return diff: {cand_ret - base_ret:+.2%})."
            )

        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="target_metric_improvement")
    return GateResult(accepted=True)


def _check_return_floor(
    report: EvaluationReport,
    baseline: EvaluationReport,
    min_return_ratio: float,
) -> GateResult:
    """Check candidate return is at least min_return_ratio of baseline.

    When the baseline has positive annualized return, the candidate's
    annualized return must be at least ``min_return_ratio * baseline_return``.
    If baseline return is zero or negative, this check is always passed.

    Args:
        report: Candidate evaluation report.
        baseline: Incumbent evaluation report.
        min_return_ratio: Minimum fraction of baseline return (0.0 to 1.0).

    Returns:
        GateResult: Decision outcome. On failure, ``failed_gate`` is ``"min_return_ratio"``.
    """
    cand_ret = report.in_sample_metrics.annualized_return
    base_ret = baseline.in_sample_metrics.annualized_return
    if math.isnan(cand_ret):
        msg = "Candidate in-sample annualized return is NaN."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="min_return_ratio")
    if base_ret > 0 and cand_ret < base_ret * min_return_ratio:
        msg = (
            f"Candidate in-sample annualized return ({cand_ret:.4f}) is below "
            f"{min_return_ratio:.0%} of baseline annualized return ({base_ret:.4f})."
        )
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="min_return_ratio")
    return GateResult(accepted=True)


def _check_dsr_non_degradation(
    report: EvaluationReport,
    baseline: EvaluationReport,
) -> GateResult:
    """Check candidate DSR does not degrade below baseline DSR.

    Uses an epsilon of 1e-6 for floating-point comparison. If the baseline DSR
    is NaN, the check is automatically passed (no valid baseline to compare against).

    Args:
        report: Candidate evaluation report.
        baseline: Incumbent evaluation report.

    Returns:
        GateResult: Decision outcome. On failure, ``failed_gate`` is ``"dsr_non_degradation"``.
    """
    eps = 1e-6
    cand_dsr = report.deflated_sharpe
    base_dsr = baseline.deflated_sharpe
    if math.isnan(cand_dsr):
        msg = "Candidate in-sample DSR is NaN."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="dsr_non_degradation")
    if math.isnan(base_dsr):
        return GateResult(accepted=True)
    if cand_dsr < base_dsr - eps:
        msg = (
            f"Candidate in-sample DSR ({cand_dsr:.6f}) degrades below "
            f"baseline in-sample DSR ({base_dsr:.6f}) by more than {eps}."
        )
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="dsr_non_degradation")
    return GateResult(accepted=True)


# ---------------------------------------------------------------------------
# Confirmation gate (holdout — budgeted peeks only)
# ---------------------------------------------------------------------------


def confirm(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    holdout_min_improvement: float | None = None,
    config: Any = None,
) -> GateResult:
    """Holdout confirmation gate — reached only when ``select`` passes.

    Hard constraints on the holdout:
    1. Drawdown: ``holdout_metrics.max_drawdown <= dd_limit``
    2. Turnover: ``holdout_metrics.turnover <= turnover_limit``

    Confirmation (only when baseline is present):
    3. Holdout DSR non-degradation:
       ``report.holdout_deflated_sharpe >= baseline.holdout_deflated_sharpe - eps``
       with ``holdout_min_improvement`` tolerance (default 0.0 → strict non-degradation).

    Args:
        report: Candidate evaluation report (must have holdout metrics populated).
        baseline: Incumbent evaluation report, or ``None`` for first iteration.
        dd_limit: Max holdout drawdown threshold. Resolved from config if ``None``.
        turnover_limit: Max holdout turnover threshold. Resolved from config if ``None``.
        holdout_min_improvement: Tolerance for holdout DSR comparison. Resolved from config if ``None``.
        config: StrategyConfig or dict providing fallback values for unresolved parameters.

    Returns:
        GateResult: Decision outcome. Each call counts as one holdout peek.
    """
    dd_limit = dd_limit if dd_limit is not None else _get_config_val(config, "max_drawdown_limit", 0.20)
    turnover_limit = turnover_limit if turnover_limit is not None else _get_config_val(config, "turnover_limit", 2.0)
    holdout_min_improvement = (
        holdout_min_improvement
        if holdout_min_improvement is not None
        else _get_config_val(config, "holdout_min_improvement", 0.0)
    )

    # --- Hard constraints on holdout_metrics ---
    max_dd = report.holdout_metrics.max_drawdown
    max_dd_passed = not (math.isnan(max_dd) or max_dd > dd_limit)

    turnover = report.holdout_metrics.turnover
    turnover_passed = not (math.isnan(turnover) or turnover > turnover_limit)

    # 1. Max Drawdown
    if not max_dd_passed:
        if math.isnan(max_dd):
            msg = "Holdout max drawdown is NaN."
        else:
            msg = f"Holdout max drawdown {max_dd:.4f} exceeds limit of {dd_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="max_drawdown")

    # 2. Turnover
    if not turnover_passed:
        if math.isnan(turnover):
            msg = "Holdout turnover is NaN."
        else:
            msg = f"Holdout turnover {turnover:.4f} exceeds limit of {turnover_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="turnover")

    # 3. Holdout DSR non-degradation
    hd_candidate = report.holdout_deflated_sharpe
    if math.isnan(hd_candidate):
        msg = "Candidate holdout DSR is NaN."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="holdout_dsr_non_degradation")

    if baseline is not None:
        eps = 1e-6
        hd_baseline = baseline.holdout_deflated_sharpe

        if not math.isnan(hd_baseline) and hd_candidate < hd_baseline - eps - holdout_min_improvement:
            msg = (
                f"Candidate holdout DSR ({hd_candidate:.6f}) degrades below "
                f"baseline holdout DSR ({hd_baseline:.6f}) by more than "
                f"{eps + holdout_min_improvement:.6f}."
            )
            report.is_accepted = False
            report.rejection_reason = msg
            return GateResult(accepted=False, reason=msg, failed_gate="holdout_dsr_non_degradation")

    report.is_accepted = True
    report.rejection_reason = None
    return GateResult(accepted=True)


# ---------------------------------------------------------------------------
# Backward-compatible wrapper (standalone eval path)
# ---------------------------------------------------------------------------


def accept(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    target_metric: TargetMetric = TargetMetric.SHARPE,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    min_improvement: float | None = None,
    require_dsr_non_degradation: bool | None = None,
    min_return_ratio: float | None = None,
    pbo_limit: float | None = None,
    config: Any = None,
) -> GateResult:
    """Backward-compatible wrapper: composes ``select`` + ``confirm``.

    Intended for the standalone CLI ``evaluate`` path. During an
    optimization run the orchestrator calls ``select`` and ``confirm``
    as separate stages.

    Args:
        report: Candidate evaluation report.
        baseline: Incumbent evaluation report, or ``None`` for first iteration.
        target_metric: Metric choice for improvement comparison (default SHARPE).
        dd_limit: Max drawdown threshold. Resolved from config if ``None``.
        turnover_limit: Max turnover threshold. Resolved from config if ``None``.
        min_improvement: Minimum metric improvement over baseline. Resolved from config if ``None``.
        require_dsr_non_degradation: Enforce DSR non-degradation. Resolved from config if ``None``.
        min_return_ratio: Minimum fraction of baseline return. Resolved from config if ``None``.
        pbo_limit: Maximum PBO threshold. Resolved from config if ``None``.
        config: StrategyConfig or dict providing fallback values.

    Returns:
        GateResult: Decision outcome from the composed select + confirm gates.
    """
    sel = select(
        report,
        baseline=baseline,
        target_metric=target_metric,
        dd_limit=dd_limit,
        turnover_limit=turnover_limit,
        min_improvement=min_improvement,
        require_dsr_non_degradation=require_dsr_non_degradation,
        min_return_ratio=min_return_ratio,
        pbo_limit=pbo_limit,
        config=config,
    )
    if not sel.accepted:
        return sel

    cnf = confirm(
        report,
        baseline=baseline,
        dd_limit=dd_limit,
        turnover_limit=turnover_limit,
        config=config,
    )
    return cnf
