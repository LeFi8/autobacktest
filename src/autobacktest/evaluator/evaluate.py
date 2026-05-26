"""Orchestration of walk-forward and holdout backtest evaluations."""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np
import pandas as pd

from autobacktest.data.cache import CachedDataProvider
from autobacktest.data.yfinance_provider import YFinanceProvider
from autobacktest.evaluator.backtest import run_vectorized_backtest
from autobacktest.evaluator.costs import calculate_turnover_and_costs
from autobacktest.evaluator.deflated_sharpe import calculate_psr_dsr
from autobacktest.evaluator.holdout import partition_holdout_data
from autobacktest.evaluator.monte_carlo import run_block_bootstrap
from autobacktest.evaluator.regime import evaluate_stress_regimes
from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.evaluator.walk_forward import generate_walk_forward_windows
from autobacktest.strategy.config_schema import StrategyConfig


def calculate_sortino_ratio(net_returns: pd.Series) -> float:
    """Calculate the Sortino Ratio of a daily net returns series."""
    if net_returns.empty:
        return 0.0
    mean_ret = net_returns.mean()
    # Downside deviation target = 0.0, replace positive returns with 0
    negative_returns = np.minimum(net_returns, 0.0)
    # Compute downside deviation over the FULL sample size N
    downside_std = np.sqrt((negative_returns**2).mean())
    if downside_std == 0.0:
        return float("inf") if mean_ret > 0.0 else 0.0
    return float((mean_ret / downside_std) * np.sqrt(252))


def calculate_information_ratio(
    net_returns: pd.Series, benchmark_returns: pd.Series
) -> float:
    """Calculate the Information Ratio of daily returns relative to benchmark."""
    if net_returns.empty or benchmark_returns.empty:
        return 0.0
    # Align dates
    aligned = pd.concat([net_returns, benchmark_returns], axis=1).dropna()
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


def generate_window_report(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
    benchmark_returns: pd.Series | None = None,
) -> WindowReport:
    """Run backtest and cost assessment for a specific date window."""
    window_prices = prices.loc[start:end]
    window_weights = weights.loc[start:end]

    portfolio_returns, _, daily_weights = run_vectorized_backtest(
        window_prices, window_weights
    )

    # Compute net returns and turnover
    net_returns, net_equity, turnover = calculate_turnover_and_costs(
        portfolio_returns, daily_weights, window_prices
    )

    # Standard performance metrics
    mean_ret = net_returns.mean() if not net_returns.empty else 0.0
    std_ret = net_returns.std(ddof=1) if len(net_returns) >= 2 else 0.0

    ann_ret = 0.0
    if not net_returns.empty:
        total_growth = net_equity.iloc[-1] if not net_equity.empty else 1.0
        if total_growth > 0.0:
            ann_ret = float(total_growth ** (252.0 / len(net_returns)) - 1.0)
        else:
            ann_ret = -1.0  # complete loss

    ann_vol = float(std_ret * np.sqrt(252))
    sharpe = float((mean_ret / std_ret * np.sqrt(252)) if std_ret > 0.0 else 0.0)
    sortino = calculate_sortino_ratio(net_returns)

    # Maximum drawdown
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


def evaluate_strategy_detailed(
    strategy_name: str,
    generate_signals_fn: Any,
    config: dict[str, Any] | StrategyConfig,
    start_date: str = "2015-01-01",
    end_date: str = "2026-01-01",
) -> tuple[EvaluationReport, pd.Series[Any]]:
    """Run full deterministic walk-forward & holdout evaluation lifecycle.

    Returns a tuple of (EvaluationReport, holdout_net_returns Series).
    """
    if isinstance(config, StrategyConfig):
        flat_config = config.to_flat_dict()
    else:
        flat_config = dict(config)
        params = flat_config.get("params", {})
        if isinstance(params, dict):
            for k, v in params.items():
                if k not in flat_config:
                    flat_config[k] = v

    tickers = flat_config.get("universe", [])
    benchmark_ticker = flat_config.get("benchmark", "SPY")

    # Fetch daily price series
    raw_provider = YFinanceProvider()
    provider = CachedDataProvider(raw_provider)

    prices = provider.get_prices(tickers, start_date, end_date)
    benchmark_prices = provider.get_prices([benchmark_ticker], start_date, end_date)

    if prices.empty:
        raise ValueError("No price history returned for requested strategy universe.")
    if benchmark_prices.empty or benchmark_ticker not in benchmark_prices.columns:
        raise ValueError(
            f"No price history returned for benchmark ticker: {benchmark_ticker}"
        )

    # Calculate daily percentage returns of the benchmark
    bench_returns = benchmark_prices[benchmark_ticker].pct_change().fillna(0.0)

    # Dynamic dynamic weights generation from custom strategy function
    weights = generate_signals_fn(prices, flat_config)

    # Sanity validate output weights contract (Finding 14)
    from autobacktest.strategy.contract import validate_output

    ok, err = validate_output(weights, tickers, expected_index=prices.index)
    if not ok:
        raise ValueError(f"Strategy weights validation failed: {err}")

    # Partition holdout data (default last 3 years)
    in_sample_idx, holdout_idx = partition_holdout_data(prices.index, holdout_years=3)
    if in_sample_idx.empty or holdout_idx.empty:
        raise ValueError(
            "In-sample or holdout period is empty. "
            "Ensure the backtest period is sufficiently long."
        )

    # Walk-forward on In-Sample index
    wf_windows = generate_walk_forward_windows(
        in_sample_idx, train_years=5, test_years=1
    )
    wf_reports = []

    for _, _, test_start, test_end in wf_windows:
        wf_reports.append(
            generate_window_report(prices, weights, test_start, test_end, bench_returns)
        )

    # Evaluate holdout window
    holdout_start = holdout_idx.min()
    holdout_end = holdout_idx.max()
    holdout_report = generate_window_report(
        prices, weights, holdout_start, holdout_end, bench_returns
    )

    # Compute holdout net returns from the same window backtest used for holdout_report
    # (consistent source for both observed_sharpe and the DSR test statistic)
    holdout_prices = prices.loc[holdout_start:holdout_end]
    holdout_weights = weights.loc[holdout_start:holdout_end]
    h_portfolio_returns, _, h_daily_weights = run_vectorized_backtest(
        holdout_prices, holdout_weights
    )
    holdout_net_returns, _, _ = calculate_turnover_and_costs(
        h_portfolio_returns, h_daily_weights, holdout_prices
    )

    # Evaluate full period net returns for Monte Carlo and Regime tests
    full_returns, _, daily_weights = run_vectorized_backtest(prices, weights)
    net_returns, _, _ = calculate_turnover_and_costs(
        full_returns, daily_weights, prices
    )

    # Labeled stress testing regimes
    regime_drawdowns, regime_passed = evaluate_stress_regimes(net_returns)

    # Monte carlo block bootstrap (1000 paths) with seed for strict determinism
    mc_5th, mc_50th, mc_95th = run_block_bootstrap(net_returns, n_paths=1000, seed=42)

    # Sharpe ratio DSR
    observed_sharpe = holdout_report.sharpe_ratio
    effective_trials = int(flat_config.get("effective_trials", 1))
    historical_sharpes = flat_config.get("historical_sharpes")

    dsr = calculate_psr_dsr(
        holdout_net_returns,
        historical_sharpes=historical_sharpes,
        effective_trials=effective_trials,
    )

    # Generate complete stable dataset hash using hashlib
    tickers_str = ",".join(sorted(tickers))
    dataset_hash = hashlib.sha256(tickers_str.encode()).hexdigest()[:16]

    # Generate complete report (Finding 7)
    report = EvaluationReport(
        strategy_name=strategy_name,
        dataset_hash=dataset_hash,
        gates_passed={},
        is_accepted=False,
        rejection_reason=None,
        holdout_metrics=holdout_report,
        walk_forward_metrics=wf_reports,
        regime_drawdowns=regime_drawdowns,
        regime_passed=regime_passed,
        mc_sharpe_5th=mc_5th,
        mc_sharpe_50th=mc_50th,
        mc_sharpe_95th=mc_95th,
        observed_sharpe=observed_sharpe,
        effective_trials=effective_trials,
        deflated_sharpe=dsr,
    )

    # Delegate gate checks to unified gate.accept (Finding 7)
    from autobacktest.gate import accept as gate_accept

    gate_accept(report, baseline=None, config=flat_config)

    return report, holdout_net_returns


def evaluate_strategy(
    strategy_name: str,
    generate_signals_fn: Any,
    config: dict[str, Any] | StrategyConfig,
    start_date: str = "2015-01-01",
    end_date: str = "2026-01-01",
) -> EvaluationReport:
    """Run full deterministic walk-forward & holdout evaluation lifecycle.

    Thin wrapper around evaluate_strategy_detailed that returns only the report.
    """
    report, _ = evaluate_strategy_detailed(
        strategy_name, generate_signals_fn, config, start_date, end_date
    )
    return report
