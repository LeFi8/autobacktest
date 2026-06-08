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


def _write_gates_passed(report: EvaluationReport, checks: dict[str, bool]) -> None:
    """Write gate outcomes to the report."""
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

    If all pass, tie-breaker (only when baseline is present):
    5. Target metric improvement on the in-sample aggregate
       (``candidate > baseline + min_improvement``)
    6. Annualized return must be at least ``min_return_ratio`` of baseline return.
    7. DSR non-degradation: always-on when baseline is present
       (config can disable via ``require_dsr_non_degradation: false``).

    Returns:
        GateResult: Decision outcome. The holdout is **never** consulted.
    """
    dd_limit, turnover_limit, pbo_limit, min_improvement, min_return_ratio, require_dsr, tradeoff, floor = (
        _resolve_select_config(
            config,
            dd_limit,
            turnover_limit,
            pbo_limit,
            min_improvement,
            min_return_ratio,
            require_dsr_non_degradation,
        )
    )

    result = _check_hard_gates(report, dd_limit, turnover_limit, pbo_limit)
    if not result.accepted:
        return result

    result = _check_soft_gates(
        report, baseline, target_metric, min_improvement, min_return_ratio, require_dsr, tradeoff, floor
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
) -> tuple[float, float, float | None, float, float, bool, float, float | None]:
    """Resolve select gate parameters from explicit args or config object."""
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
    return dd_limit, turnover_limit, pbo_limit, min_improvement, min_return_ratio, require_dsr, tradeoff, floor


def _check_hard_gates(
    report: EvaluationReport,
    dd_limit: float,
    turnover_limit: float,
    pbo_limit: float | None,
) -> GateResult:
    """Check hard constraints: drawdown, regime, turnover, PBO."""
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
        msg = "Strategy failed to pass historical crisis regime drawdown stress test."
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
) -> GateResult:
    """Check soft constraints: metric improvement, return floor, DSR non-degradation."""
    if baseline is not None:
        result = _check_metric_improvement(
            report, baseline, target_metric, min_improvement, tradeoff_coeff, metric_floor
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
    metric_floor: float | None = None,
) -> GateResult:
    """Check candidate metric improves over baseline, incorporating trade-off and floor."""
    candidate_val = _get_in_sample_metric_val(report, target_metric)
    baseline_val = _get_in_sample_metric_val(baseline, target_metric)

    cand_ret = report.in_sample_metrics.annualized_return
    base_ret = baseline.in_sample_metrics.annualized_return

    if tradeoff_coeff > 0.0 and (math.isnan(cand_ret) or math.isnan(base_ret)):
        msg = f"Candidate or baseline annualized return is NaN while metric_return_tradeoff={tradeoff_coeff} is active."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="target_metric_improvement")

    required_metric = baseline_val + min_improvement
    if tradeoff_coeff > 0.0:
        required_metric -= tradeoff_coeff * (cand_ret - base_ret) * 100

    is_below_hurdle = candidate_val <= required_metric
    is_below_floor = metric_floor is not None and candidate_val <= metric_floor

    if math.isnan(candidate_val) or math.isnan(baseline_val) or is_below_hurdle or is_below_floor:
        if is_below_floor:
            msg = (
                f"Candidate {target_metric.value} ({candidate_val:.4f}) is below the"
                f" required floor of {metric_floor:.4f}."
            )
            failed_gate = "metric_floor"
        else:
            msg = (
                f"Candidate in-sample {target_metric.value} ({candidate_val:.4f}) does not "
                f"improve upon baseline in-sample {target_metric.value} ({baseline_val:.4f}) "
                f"by at least {min_improvement:.4f}."
            )
            if tradeoff_coeff > 0.0:
                msg = (
                    f"Candidate {target_metric.value} ({candidate_val:.4f}) is below "
                    f"the adjusted hurdle of {required_metric:.4f} "
                    f"(baseline: {baseline_val:.4f}, return diff: {cand_ret - base_ret:+.2%})."
                )
            failed_gate = "target_metric_improvement"

        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate=failed_gate)
    return GateResult(accepted=True)


def _check_return_floor(
    report: EvaluationReport,
    baseline: EvaluationReport,
    min_return_ratio: float,
) -> GateResult:
    """Check candidate return is at least min_return_ratio of baseline."""
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
    """Check candidate DSR does not degrade below baseline DSR."""
    eps = 1e-6
    if report.deflated_sharpe < baseline.deflated_sharpe - eps:
        msg = (
            f"Candidate in-sample DSR ({report.deflated_sharpe:.6f}) degrades below "
            f"baseline in-sample DSR ({baseline.deflated_sharpe:.6f}) by more than {eps}."
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
    if baseline is not None:
        eps = 1e-6
        hd_candidate = report.holdout_deflated_sharpe
        hd_baseline = baseline.holdout_deflated_sharpe

        if hd_candidate < hd_baseline - eps - holdout_min_improvement:
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

    Intended for the standalone CLI ``evaluate`` path.  During an
    optimisation run the orchestrator calls ``select`` and ``confirm``
    as separate stages.

    The ``require_dsr_non_degradation`` parameter is propagated to the
    underlying ``select`` gate.  When ``None`` the value is resolved
    from the ``config`` (default ``True``).
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
