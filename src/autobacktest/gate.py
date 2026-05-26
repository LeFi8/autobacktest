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


def accept(
    report: EvaluationReport,
    baseline: EvaluationReport | None,
    target_metric: TargetMetric = TargetMetric.SHARPE,
    dd_limit: float | None = None,
    turnover_limit: float | None = None,
    dsr_threshold: float = 0.95,
    min_improvement: float = 0.0,
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
    # Resolve limits from config if not explicitly passed
    if dd_limit is None:
        if config is not None:
            if hasattr(config, "max_drawdown_limit"):
                dd_limit = config.max_drawdown_limit
            elif isinstance(config, dict) and "max_drawdown_limit" in config:
                dd_limit = config["max_drawdown_limit"]
            else:
                dd_limit = 0.15
        else:
            dd_limit = 0.15

    if turnover_limit is None:
        if config is not None:
            if hasattr(config, "turnover_limit"):
                turnover_limit = config.turnover_limit
            elif isinstance(config, dict) and "turnover_limit" in config:
                turnover_limit = config["turnover_limit"]
            else:
                turnover_limit = 1.0
        else:
            turnover_limit = 1.0

    # 1. Max Drawdown Limit
    max_dd = report.holdout_metrics.max_drawdown
    if math.isnan(max_dd) or max_dd > dd_limit:
        msg = (
            f"Holdout max drawdown {max_dd:.4f} is NaN or "
            f"exceeds limit of {dd_limit:.4f}."
            if not math.isnan(max_dd)
            else "Holdout max drawdown is NaN."
        )
        return GateResult(
            accepted=False,
            reason=msg,
            failed_gate="max_drawdown",
        )

    # 2. Regimes stress tests
    if not report.regime_passed:
        msg = "Strategy failed to pass historical crisis regime drawdown stress test."
        return GateResult(
            accepted=False,
            reason=msg,
            failed_gate="regimes",
        )

    # 3. Turnover limit
    turnover = report.holdout_metrics.turnover
    if math.isnan(turnover) or turnover > turnover_limit:
        msg = (
            f"Holdout turnover {turnover:.4f} is NaN or "
            f"exceeds limit of {turnover_limit:.4f}."
            if not math.isnan(turnover)
            else "Holdout turnover is NaN."
        )
        return GateResult(
            accepted=False,
            reason=msg,
            failed_gate="turnover",
        )

    # 4. Deflated Sharpe Ratio threshold
    dsr = report.deflated_sharpe
    if math.isnan(dsr) or dsr < dsr_threshold:
        msg = (
            f"Deflated Sharpe Ratio {dsr:.4f} is NaN or "
            f"below threshold of {dsr_threshold:.4f}."
            if not math.isnan(dsr)
            else "Deflated Sharpe Ratio is NaN."
        )
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
            return GateResult(
                accepted=False,
                reason=(
                    f"Candidate {target_metric.value} ({candidate_val:.4f}) does not "
                    f"improve upon baseline {target_metric.value} ({baseline_val:.4f}) "
                    f"by at least {min_improvement:.4f}."
                ),
                failed_gate="target_metric_improvement",
            )

    return GateResult(accepted=True)
