"""Improvement gate rules checks and comparison of strategy metrics."""

from dataclasses import dataclass
from enum import StrEnum

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
    dd_limit: float = 0.15,
    turnover_limit: float = 1.0,
    dsr_threshold: float = 0.95,
) -> GateResult:
    """Evaluate candidate EvaluationReport against lexicographic gates.

    Hard constraints checked in sequence:
    1. Drawdown: holdout_metrics.max_drawdown <= dd_limit
    2. Regime tests: regime_passed is True
    3. Turnover: holdout_metrics.turnover <= turnover_limit
    4. DSR: deflated_sharpe >= dsr_threshold (e.g. 0.95 probability)

    If all pass, tie-breaker:
    5. Improvement: target metric value > baseline target metric value
       (if baseline is present)

    Args:
        report: EvaluationReport of candidate strategy.
        baseline: Optional baseline strategy EvaluationReport.
        target_metric: Target optimization metric choice.
        dd_limit: Maximum allowed drawdown in holdout.
        turnover_limit: Maximum allowed annualized turnover rate in holdout.
        dsr_threshold: Minimum required Deflated Sharpe Ratio.

    Returns:
        GateResult: Decision outcome.
    """
    # 1. Max Drawdown Limit
    if report.holdout_metrics.max_drawdown > dd_limit:
        msg = (
            f"Holdout max drawdown {report.holdout_metrics.max_drawdown:.4f} "
            f"exceeds limit of {dd_limit:.4f}."
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
    if report.holdout_metrics.turnover > turnover_limit:
        msg = (
            f"Holdout turnover {report.holdout_metrics.turnover:.4f} "
            f"exceeds limit of {turnover_limit:.4f}."
        )
        return GateResult(
            accepted=False,
            reason=msg,
            failed_gate="turnover",
        )

    # 4. Deflated Sharpe Ratio threshold
    if report.deflated_sharpe < dsr_threshold:
        msg = (
            f"Deflated Sharpe Ratio {report.deflated_sharpe:.4f} "
            f"is below threshold of {dsr_threshold:.4f}."
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

        if candidate_val <= baseline_val:
            return GateResult(
                accepted=False,
                reason=(
                    f"Candidate {target_metric.value} ({candidate_val:.4f}) does not "
                    f"improve upon baseline {target_metric.value} ({baseline_val:.4f})."
                ),
                failed_gate="target_metric_improvement",
            )

    return GateResult(accepted=True)
