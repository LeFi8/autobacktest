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


def accept(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    target_metric: TargetMetric = TargetMetric.SHARPE,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    dsr_threshold: float | None = None,
    min_improvement: float | None = None,
    config: Any = None,
) -> GateResult:
    """Evaluate candidate EvaluationReport against lexicographic gates.

    Hard constraints checked in sequence:
    1. Drawdown: holdout_metrics.max_drawdown <= dd_limit
    2. Regime tests: regime_passed is True
    3. Turnover: holdout_metrics.turnover <= turnover_limit
    4. DSR: deflated_sharpe >= dsr_threshold (e.g. 0.95 probability)

    If all pass, tie-breaker:
    5. Improvement: target metric value > baseline target metric value + min_improvement
       (if baseline is present)

    Args:
        report: EvaluationReport of candidate strategy.
        baseline: Optional baseline strategy EvaluationReport.
        target_metric: Target optimization metric choice.
        dd_limit: Maximum allowed drawdown in holdout.
        turnover_limit: Maximum allowed annualized turnover rate in holdout.
        dsr_threshold: Minimum required Deflated Sharpe Ratio.
        min_improvement: Minimum required improvement epsilon.
        config: Optional strategy configuration to resolve limits.

    Returns:
        GateResult: Decision outcome.
    """
    # Resolve limits and parameters from config if not explicitly passed (Finding 10)
    dd_limit = (
        dd_limit
        if dd_limit is not None
        else _get_config_val(config, "max_drawdown_limit", 0.15)
    )
    turnover_limit = (
        turnover_limit
        if turnover_limit is not None
        else _get_config_val(config, "turnover_limit", 1.0)
    )
    dsr_threshold = (
        dsr_threshold
        if dsr_threshold is not None
        else _get_config_val(config, "dsr_threshold", 0.95)
    )
    min_improvement = (
        min_improvement
        if min_improvement is not None
        else _get_config_val(config, "min_improvement", 0.0)
    )

    # Evaluate gates individually and populate gates_passed
    max_dd = report.holdout_metrics.max_drawdown
    max_dd_passed = not (math.isnan(max_dd) or max_dd > dd_limit)

    regime_passed = bool(report.regime_passed)

    turnover = report.holdout_metrics.turnover
    turnover_passed = not (math.isnan(turnover) or turnover > turnover_limit)

    dsr = report.deflated_sharpe
    dsr_passed = not (dsr is None or math.isnan(dsr) or dsr < dsr_threshold)

    # Let gate.accept write back outcomes to the report (Finding 7)
    report.gates_passed = {
        "max_drawdown": max_dd_passed,
        "turnover": turnover_passed,
        "regimes": regime_passed,
        "deflated_sharpe": dsr_passed,
    }

    # 1. Max Drawdown Limit
    if not max_dd_passed:
        # Differentiate NaN vs float limit breach messages (Finding 15)
        if math.isnan(max_dd):
            msg = "Holdout max drawdown is NaN."
        else:
            msg = f"Holdout max drawdown {max_dd:.4f} exceeds limit of {dd_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(
            accepted=False,
            reason=msg,
            failed_gate="max_drawdown",
        )

    # 2. Regimes stress tests
    if not regime_passed:
        msg = "Strategy failed to pass historical crisis regime drawdown stress test."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(
            accepted=False,
            reason=msg,
            failed_gate="regimes",
        )

    # 3. Turnover limit
    if not turnover_passed:
        if math.isnan(turnover):
            msg = "Holdout turnover is NaN."
        else:
            msg = (
                f"Holdout turnover {turnover:.4f} "
                f"exceeds limit of {turnover_limit:.4f}."
            )
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(
            accepted=False,
            reason=msg,
            failed_gate="turnover",
        )

    # 4. Deflated Sharpe Ratio threshold
    if not dsr_passed:
        if dsr is None or math.isnan(dsr):
            msg = "Deflated Sharpe Ratio is missing or NaN."
        else:
            msg = (
                f"Deflated Sharpe Ratio {dsr:.4f} "
                f"is below threshold of {dsr_threshold:.4f}."
            )
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(
            accepted=False,
            reason=msg,
            failed_gate="deflated_sharpe",
        )

    # 5. Optimization Target Metric Improvement over Baseline
    if baseline is not None:

        def _get_metric_val(rep: EvaluationReport) -> float:
            if target_metric == TargetMetric.SHARPE:
                return rep.holdout_metrics.sharpe_ratio
            if target_metric == TargetMetric.SORTINO:
                return rep.holdout_metrics.sortino_ratio
            if target_metric == TargetMetric.INFORMATION_RATIO:
                return rep.holdout_metrics.information_ratio
            raise ValueError(f"Unsupported target metric choice: {target_metric}")

        candidate_val = _get_metric_val(report)
        baseline_val = _get_metric_val(baseline)

        if (
            math.isnan(candidate_val)
            or math.isnan(baseline_val)
            or candidate_val <= baseline_val + min_improvement
        ):
            msg = (
                f"Candidate {target_metric.value} ({candidate_val:.4f}) does not "
                f"improve upon baseline {target_metric.value} ({baseline_val:.4f}) "
                f"by at least {min_improvement:.4f}."
            )
            report.is_accepted = False
            report.rejection_reason = msg
            return GateResult(
                accepted=False,
                reason=msg,
                failed_gate="target_metric_improvement",
            )

    report.is_accepted = True
    report.rejection_reason = None
    return GateResult(accepted=True)
