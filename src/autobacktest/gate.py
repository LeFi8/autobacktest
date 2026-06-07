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

    # --- Hard constraints on in_sample_metrics ---
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
        {
            "max_drawdown": max_dd_passed,
            "turnover": turnover_passed,
            "regimes": regime_passed,
            "pbo": pbo_passed,
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

    # 4. PBO limit check
    if not pbo_passed:
        msg = f"In-sample PBO {report.pbo:.4f} exceeds limit of {pbo_limit:.4f}."
        report.is_accepted = False
        report.rejection_reason = msg
        return GateResult(accepted=False, reason=msg, failed_gate="pbo")

    # 5. Target Metric Improvement on in-sample aggregate
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

    # 6. Annualized Return Floor (must maintain min_return_ratio of baseline return)
    if baseline is not None:
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

    # 7. DSR Non-Degradation (always-on by default)
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
