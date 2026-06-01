"""Improvement gate rules checks and comparison of strategy metrics."""

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from autobacktest.evaluator.report import EvaluationReport


class TargetMetric(StrEnum):
    """Optimization objective metrics for target strategy optimization."""

    SHARPE = "sharpe"
    SORTINO = "sortino"
    INFORMATION_RATIO = "information_ratio"


@dataclass
class GateResult:
    """The outcome result of a lexicographic gate evaluation."""

    accepted: bool
    reason: str | None = None
    failed_gate: str | None = None


def _get_config_val(config: Any, key: str, default: Any) -> Any:
    """Safely retrieve configuration key from Pydantic model or dict."""
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
    """Extract target metric from a report's in-sample walk-forward aggregate."""
    if target_metric == TargetMetric.SHARPE:
        return report.in_sample_metrics.sharpe_ratio
    if target_metric == TargetMetric.SORTINO:
        return report.in_sample_metrics.sortino_ratio
    if target_metric == TargetMetric.INFORMATION_RATIO:
        return report.in_sample_metrics.information_ratio
    raise ValueError(f"Unsupported target metric choice: {target_metric}")


def _get_holdout_metric_val(report: EvaluationReport, target_metric: TargetMetric) -> float:
    """Extract target metric from a report's holdout window."""
    if target_metric == TargetMetric.SHARPE:
        return report.holdout_metrics.sharpe_ratio
    if target_metric == TargetMetric.SORTINO:
        return report.holdout_metrics.sortino_ratio
    if target_metric == TargetMetric.INFORMATION_RATIO:
        return report.holdout_metrics.information_ratio
    raise ValueError(f"Unsupported target metric choice: {target_metric}")


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
    config: Any = None,
) -> GateResult:
    """In-sample selection gate — evaluated on every candidate.

    Hard constraints (in-sample walk-forward aggregate):
    1. Drawdown: ``in_sample_metrics.max_drawdown <= dd_limit``
    2. Regime tests: ``regime_passed`` is True
    3. Turnover: ``in_sample_metrics.turnover <= turnover_limit``

    If all pass, tie-breaker (only when baseline is present):
    4. Target metric improvement on the in-sample aggregate
       (``candidate > baseline + min_improvement``)
    5. DSR non-degradation: always-on when baseline is present
       (config can disable via ``require_dsr_non_degradation: false``).

    Returns:
        GateResult: Decision outcome. The holdout is **never** consulted.
    """
    dd_limit = dd_limit if dd_limit is not None else _get_config_val(config, "max_drawdown_limit", 0.20)
    turnover_limit = turnover_limit if turnover_limit is not None else _get_config_val(config, "turnover_limit", 2.0)
    min_improvement = (
        min_improvement if min_improvement is not None else _get_config_val(config, "min_improvement", 0.0)
    )
    require_dsr = _get_config_val(config, "require_dsr_non_degradation", True)

    # --- Hard constraints on in_sample_metrics ---
    max_dd = report.in_sample_metrics.max_drawdown
    max_dd_passed = not (math.isnan(max_dd) or max_dd > dd_limit)

    regime_passed = bool(report.regime_passed)

    turnover = report.in_sample_metrics.turnover
    turnover_passed = not (math.isnan(turnover) or turnover > turnover_limit)

    _write_gates_passed(
        report,
        {
            "max_drawdown": max_dd_passed,
            "turnover": turnover_passed,
            "regimes": regime_passed,
        },
    )

    # 1. Max Drawdown
    if not max_dd_passed:
        if math.isnan(max_dd):
            msg = "In-sample max drawdown is NaN."
        else:
            msg = f"In-sample max drawdown {max_dd:.4f} exceeds limit of {dd_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="max_drawdown")

    # 2. Regime stress tests
    if not regime_passed:
        msg = "Strategy failed to pass historical crisis regime drawdown stress test."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="regimes")

    # 3. Turnover
    if not turnover_passed:
        if math.isnan(turnover):
            msg = "In-sample turnover is NaN."
        else:
            msg = f"In-sample turnover {turnover:.4f} exceeds limit of {turnover_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="turnover")

    # 4. Target Metric Improvement on in-sample aggregate
    if baseline is not None:
        candidate_val = _get_in_sample_metric_val(report, target_metric)
        baseline_val = _get_in_sample_metric_val(baseline, target_metric)

        if math.isnan(candidate_val) or math.isnan(baseline_val) or candidate_val <= baseline_val + min_improvement:
            msg = (
                f"Candidate in-sample {target_metric.value} ({candidate_val:.4f}) does not "
                f"improve upon baseline in-sample {target_metric.value} ({baseline_val:.4f}) "
                f"by at least {min_improvement:.4f}."
            )
            report.is_accepted = False
            report.rejection_reason = msg
            return GateResult(accepted=False, reason=msg, failed_gate="target_metric_improvement")

    # 5. DSR Non-Degradation (always-on by default)
    if require_dsr and baseline is not None:
        eps = 1e-6
        if report.deflated_sharpe < baseline.deflated_sharpe - eps:
            msg = (
                f"Candidate in-sample DSR ({report.deflated_sharpe:.6f}) degrades below "
                f"baseline in-sample DSR ({baseline.deflated_sharpe:.6f}) by more than {eps}."
            )
            report.is_accepted = False
            report.rejection_reason = msg
            return GateResult(accepted=False, reason=msg, failed_gate="dsr_non_degradation")

    # All gates passed
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
    require_dsr_non_degradation: bool | None = None,  # noqa: ARG001
    config: Any = None,
) -> GateResult:
    """Backward-compatible wrapper: composes ``select`` + ``confirm``.

    Intended for the standalone CLI ``evaluate`` path.  During an
    optimisation run the orchestrator calls ``select`` and ``confirm``
    as separate stages.

    The ``require_dsr_non_degradation`` parameter is conventionally
    ignored — the in-sample selection gate always enforces DSR
    non-degradation (configurable via ``config.require_dsr_non_degradation``).
    It is accepted here only for signature compatibility with existing
    callers who may pass it explicitly.
    """
    sel = select(
        report,
        baseline=baseline,
        target_metric=target_metric,
        dd_limit=dd_limit,
        turnover_limit=turnover_limit,
        min_improvement=min_improvement,
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
