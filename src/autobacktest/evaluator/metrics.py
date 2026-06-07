"""Return metrics, Sharpe, Sortino, and Information ratio calculations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from autobacktest.evaluator.report import WindowReport


def _compute_returns_metrics(
    returns: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> WindowReport:
    """Compute standard performance metrics directly from a return series."""
    period_returns = returns.loc[start:end]
    if period_returns.empty:
        return WindowReport(
            start_date=start.strftime("%Y-%m-%d"),
            end_date=end.strftime("%Y-%m-%d"),
            annualized_return=0.0,
            annualized_volatility=0.0,
            sharpe_ratio=0.0,
            sortino_ratio=0.0,
            max_drawdown=0.0,
            turnover=0.0,
        )

    mean_ret = period_returns.mean()
    std_ret = period_returns.std(ddof=1) if len(period_returns) >= 2 else 0.0

    cum = (1.0 + period_returns).cumprod()
    total_growth = cum.iloc[-1] if not cum.empty else 1.0
    ann_ret = float(total_growth ** (252.0 / len(period_returns)) - 1.0) if total_growth > 0.0 else -1.0
    ann_vol = float(std_ret * np.sqrt(252))
    sharpe = float((mean_ret / std_ret * np.sqrt(252)) if std_ret > 0.0 else 0.0)

    running_max = cum.cummax()
    drawdowns = (cum - running_max) / running_max
    max_dd = float(abs(drawdowns.min())) if not drawdowns.empty else 0.0

    negative_returns = np.minimum(period_returns, 0.0)
    downside_std = float(np.sqrt((negative_returns**2).mean()))
    sortino = float((mean_ret / downside_std) * np.sqrt(252)) if downside_std > 0.0 else 0.0

    return WindowReport(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        turnover=0.0,
        information_ratio=None,
    )


def calculate_sortino_ratio(net_returns: pd.Series) -> float:
    """Calculate the Sortino Ratio of a daily net returns series."""
    if net_returns.empty:
        return 0.0
    mean_ret = net_returns.mean()
    negative_returns = np.minimum(net_returns, 0.0)
    downside_std = np.sqrt((negative_returns**2).mean())
    if downside_std == 0.0:
        return float("inf") if mean_ret > 0.0 else 0.0
    return float((mean_ret / downside_std) * np.sqrt(252))


def calculate_information_ratio(net_returns: pd.Series, benchmark_returns: pd.Series) -> float:
    """Calculate the Information Ratio of daily returns relative to benchmark."""
    if net_returns.empty or benchmark_returns.empty:
        return 0.0
    aligned = pd.concat([net_returns, benchmark_returns], axis=1, sort=True).dropna()
    if aligned.empty:
        return 0.0
    active_returns = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    if len(active_returns) < 2:
        return 0.0
    mean_active = active_returns.mean()
    tracking_error = active_returns.std(ddof=1)
    if tracking_error == 0.0 or np.isnan(tracking_error):
        return 0.0
    return float((mean_active / tracking_error) * np.sqrt(252))


def aggregate_walk_forward(wf_reports: list[WindowReport]) -> WindowReport:
    """Aggregate individual walk-forward window reports into a single summary."""
    if not wf_reports:
        raise ValueError("At least one walk-forward window is required to compute in-sample metrics.")

    n = len(wf_reports)

    def _mean(attr: str) -> float:
        return float(sum(getattr(r, attr) for r in wf_reports) / n)

    def _max(attr: str) -> float:
        return float(max(getattr(r, attr) for r in wf_reports))

    return WindowReport(
        start_date=wf_reports[0].start_date,
        end_date=wf_reports[-1].end_date,
        annualized_return=_mean("annualized_return"),
        annualized_volatility=_mean("annualized_volatility"),
        sharpe_ratio=_mean("sharpe_ratio"),
        sortino_ratio=_mean("sortino_ratio"),
        max_drawdown=_max("max_drawdown"),
        turnover=_max("turnover"),
        information_ratio=_mean("information_ratio"),
    )
