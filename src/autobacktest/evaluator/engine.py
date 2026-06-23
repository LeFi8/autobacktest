"""Vectorized window execution, dataloading checks, and caching engine.

Orchestrates walk-forward evaluation by:
1. Partitioning price data into train/test windows.
2. Running vectorized backtests on each window in parallel.
3. Computing per-window performance metrics (Sharpe, Sortino, IR, costs).
4. Caching evaluated results keyed by dataset hash to avoid redundant work.

The ``_CacheProtocol`` interface allows both plain dicts and the
``_LRUCache`` (used in orchestrator) to serve as transparent caches.
"""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Protocol

import numpy as np
import pandas as pd

from autobacktest.evaluator.backtest import run_vectorized_backtest
from autobacktest.evaluator.costs import calculate_turnover_and_costs
from autobacktest.evaluator.metrics import calculate_information_ratio, calculate_sortino_ratio
from autobacktest.evaluator.report import WindowReport


class _CacheProtocol(Protocol):
    """Minimal eval-result cache interface — satisfies both ``dict`` and ``_LRUCache``.

    Provides dict-like ``__getitem__`` / ``__setitem__`` access to enable
    transparent use as the ``_eval_cache`` parameter in
    ``evaluate_strategy_detailed``.
    """

    def get(self, key: int, default: Any = None) -> Any: ...

    def __getitem__(self, key: int) -> Any: ...

    def __setitem__(self, key: int, value: Any) -> None: ...


def compute_dataset_hash(
    tickers: list[str],
    start_date: str = "",
    end_date: str = "",
    holdout_years: int = 3,
) -> str:
    """Compute a stable dataset hash from universe tickers and date parameters.

    Used as the ``dataset_hash`` key in the SQLite ledger for correlating
    attempts across runs against the same universe.

    Args:
        tickers: List of ticker symbols in the universe.
        start_date: Start date (YYYY-MM-DD) of the backtest period.
        end_date: End date (YYYY-MM-DD) of the backtest period.
        holdout_years: Number of holdout years.

    Returns:
        str: First 16 hex characters of the SHA-256 hash.
    """
    data = "|".join(
        [
            ",".join(sorted(tickers)),
            start_date,
            end_date,
            str(holdout_years),
        ]
    )
    return hashlib.sha256(data.encode()).hexdigest()[:16]


def generate_window_report(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    benchmark_returns: pd.Series | None = None,
    *,
    asset_returns: pd.DataFrame | None = None,
    borrow_cost_bps: float = 100.0,
    adaptive_slippage: bool = False,
    slippage_vol_window: int = 21,
    slippage_vol_cap: float = 3.0,
    impact_coef: float = 0.0,
) -> WindowReport:
    """Run backtest and cost assessment for a specific date window.

    Executes the vectorized backtest, applies transaction costs
    (commissions, spreads, impact, borrowing), and computes all
    performance metrics for the window.

    Args:
        prices: Daily close prices DataFrame (DatetimeIndex).
        weights: Portfolio weights DataFrame (DatetimeIndex).
        start: Window start date (inclusive).
        end: Window end date (inclusive).
        benchmark_returns: Benchmark returns for Information Ratio.
        asset_returns: Pre-computed daily asset returns (optimisation).
        borrow_cost_bps: Annualised short borrowing cost in bps.
        adaptive_slippage: Use volatility-adaptive slippage.
        slippage_vol_window: Volatility estimation window.
        slippage_vol_cap: Volatility cap multiplier.
        impact_coef: Market impact parameter (quadratic/linear cost).

    Returns:
        WindowReport: All performance metrics for this window.
    """
    window_prices = prices.loc[start:end]
    window_weights = weights.loc[start:end]

    portfolio_returns, _, daily_weights = run_vectorized_backtest(
        window_prices,
        window_weights,
        asset_returns=asset_returns,
    )

    net_returns, net_equity, turnover = calculate_turnover_and_costs(
        portfolio_returns,
        daily_weights,
        window_prices,
        asset_returns=asset_returns,
        borrow_cost_bps=borrow_cost_bps,
        adaptive_slippage=adaptive_slippage,
        slippage_vol_window=slippage_vol_window,
        slippage_vol_cap=slippage_vol_cap,
        impact_coef=impact_coef,
    )

    mean_ret = net_returns.mean() if not net_returns.empty else 0.0
    std_ret = net_returns.std(ddof=1) if len(net_returns) >= 2 else 0.0

    ann_ret = 0.0
    if not net_returns.empty:
        total_growth = net_equity.iloc[-1] if not net_equity.empty else 1.0
        ann_ret = float(total_growth ** (252.0 / len(net_returns)) - 1.0) if total_growth > 0.0 else -1.0

    ann_vol = float(std_ret * np.sqrt(252))
    sharpe = float((mean_ret / std_ret * np.sqrt(252)) if std_ret > 0.0 else 0.0)
    sortino = calculate_sortino_ratio(net_returns)

    running_max = net_equity.cummax()
    drawdowns = (net_equity - running_max) / running_max
    max_dd = float(abs(drawdowns.min())) if not drawdowns.empty else 0.0

    ir = 0.0
    if benchmark_returns is not None:
        window_bench = benchmark_returns.loc[start:end]
        ir = calculate_information_ratio(net_returns, window_bench)

    return WindowReport(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d"),
        annualized_return=ann_ret,
        annualized_volatility=ann_vol,
        sharpe_ratio=sharpe,
        sortino_ratio=sortino,
        max_drawdown=max_dd,
        turnover=turnover,
        information_ratio=ir,
    )


def _run_walk_forward_windows(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    wf_windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    bench_returns: pd.Series | None = None,
    *,
    asset_returns: pd.DataFrame | None = None,
    borrow_cost_bps: float = 100.0,
    adaptive_slippage: bool = False,
    slippage_vol_window: int = 21,
    slippage_vol_cap: float = 3.0,
    impact_coef: float = 0.0,
) -> list[WindowReport]:
    """Run walk-forward window evaluations in parallel via thread pool.

    Distributes each test window to a ``ThreadPoolExecutor`` (max 4 workers)
    for concurrent backtest and cost assessment.

    Args:
        prices: Full-period price DataFrame.
        weights: Full-period weight DataFrame.
        wf_windows: List of (train_start, train_end, test_start, test_end) tuples.
        bench_returns: Benchmark return series for Information Ratio calculation.
        asset_returns: Pre-computed daily asset returns (optimisation).
        borrow_cost_bps: Annualised short borrowing cost in basis points.
        adaptive_slippage: Use volatility-adaptive slippage model.
        slippage_vol_window: Rolling window for volatility estimation.
        slippage_vol_cap: Cap multiplier for volatility-adaptive slippage.
        impact_coef: Market impact parameter (quadratic/linear cost).

    Returns:
        list[WindowReport]: One report per walk-forward window.
    """
    n_windows = len(wf_windows)
    if n_windows == 0:
        return []

    max_workers = min(4, n_windows)
    reports: list[WindowReport | None] = [None] * n_windows

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_map: dict[Any, int] = {}
        for i, (_, _, test_start, test_end) in enumerate(wf_windows):
            future = executor.submit(
                generate_window_report,
                prices,
                weights,
                test_start,
                test_end,
                bench_returns,
                asset_returns=asset_returns,
                borrow_cost_bps=borrow_cost_bps,
                adaptive_slippage=adaptive_slippage,
                slippage_vol_window=slippage_vol_window,
                slippage_vol_cap=slippage_vol_cap,
                impact_coef=impact_coef,
            )
            future_map[future] = i

        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                reports[idx] = future.result()
            except Exception as e:
                raise RuntimeError(f"Walk-forward window {idx} failed: {e}") from e

    return [r for r in reports if r is not None]
