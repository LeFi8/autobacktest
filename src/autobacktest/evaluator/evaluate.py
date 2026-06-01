"""Orchestration of walk-forward and holdout backtest evaluations."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from autobacktest.config import settings
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


def calculate_information_ratio(net_returns: pd.Series, benchmark_returns: pd.Series) -> float:
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
    *,
    asset_returns: pd.DataFrame | None = None,
) -> WindowReport:
    """Run backtest and cost assessment for a specific date window."""
    window_prices = prices.loc[start:end]
    window_weights = weights.loc[start:end]

    portfolio_returns, _, daily_weights = run_vectorized_backtest(
        window_prices,
        window_weights,
        asset_returns=asset_returns,
    )

    # Compute net returns and turnover
    net_returns, net_equity, turnover = calculate_turnover_and_costs(
        portfolio_returns,
        daily_weights,
        window_prices,
        asset_returns=asset_returns,
    )

    # Standard performance metrics
    mean_ret = net_returns.mean() if not net_returns.empty else 0.0
    std_ret = net_returns.std(ddof=1) if len(net_returns) >= 2 else 0.0

    ann_ret = 0.0
    if not net_returns.empty:
        total_growth = net_equity.iloc[-1] if not net_equity.empty else 1.0
        ann_ret = float(total_growth ** (252.0 / len(net_returns)) - 1.0) if total_growth > 0.0 else -1.0

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


def aggregate_walk_forward(wf_reports: list[WindowReport]) -> WindowReport:
    """Aggregate individual walk-forward window reports into a single summary.

    Returns the mean of return/vol/Sharpe/Sortino/IR across all folds,
    and the **worst-case** (maximum) drawdown and turnover, so that the
    aggregate represents both central tendency and tail risk.

    Args:
        wf_reports: One ``WindowReport`` per walk-forward fold.

    Returns:
        A single ``WindowReport`` spanning the full in-sample period.

    Raises:
        ValueError: If ``wf_reports`` is empty.
    """
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


def _run_walk_forward_windows(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    wf_windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]],
    bench_returns: pd.Series | None = None,
    *,
    asset_returns: pd.DataFrame | None = None,
) -> list[WindowReport]:
    """Run walk-forward window evaluations in parallel via thread pool.

    Each window is fully independent (read-only access to ``prices`` /
    ``weights``), so we evaluate them concurrently for a modest speedup
    on multi-core machines.
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
            )
            future_map[future] = i

        for future in as_completed(future_map):
            idx = future_map[future]
            try:
                reports[idx] = future.result()
            except Exception as e:
                raise RuntimeError(f"Walk-forward window {idx} failed: {e}") from e

    return [r for r in reports if r is not None]


def evaluate_strategy_detailed(
    strategy_name: str,
    generate_signals_fn: Any,
    config: dict[str, Any] | StrategyConfig,
    start_date: str = settings.default_start_date,
    end_date: str = settings.default_end_date,
    *,
    _prices: pd.DataFrame | None = None,
    _bench_returns: pd.Series | None = None,
    _eval_cache: dict[int, tuple[EvaluationReport, pd.Series]] | None = None,
    _strategy_code: str | None = None,
) -> tuple[EvaluationReport, pd.Series[Any]]:
    """Run full deterministic walk-forward & holdout evaluation lifecycle.

    Returns a tuple of (EvaluationReport, in_sample_net_returns Series).
    The report carries ``holdout_net_returns`` for the holdout confirmation
    gate; the second element is the in-sample basis used for the selection
    DSR and diversity correlation checks.

    Parameters prefixed with ``_`` are internal optimisations:

    * ``_prices`` / ``_bench_returns`` — pre-fetched price data reused
      across iterations within a single run.
    * ``_eval_cache`` — memoization dict keyed by
      ``hash((hash(strategy_code), hash(json(config))))``.  When the same
      edit is proposed twice the full evaluation is skipped.
    * ``_strategy_code`` — source text of the strategy being evaluated,
      required for the eval cache key.
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

    # ---- Evaluation cache check (Opt 4) ----
    if _eval_cache is not None and _strategy_code is not None:
        try:
            _code_hash = hash(_strategy_code)
            _config_hash = hash(json.dumps(flat_config, sort_keys=True, default=str))
            _ckey = hash((_code_hash, _config_hash))
            cached = _eval_cache.get(_ckey)
            if cached is not None:
                cached_report, cached_returns = cached
                return deepcopy(cached_report), cached_returns.copy()
        except Exception:
            pass

    tickers = flat_config.get("universe", [])
    benchmark_ticker = flat_config.get("benchmark", "SPY")

    if _prices is not None and _bench_returns is not None:
        missing_assets = [t for t in tickers if t not in _prices.columns]
        if missing_assets:
            raise ValueError(f"Cached prices missing tickers: {missing_assets}")
        if benchmark_ticker not in _prices.columns:
            raise ValueError(f"Cached prices missing benchmark ticker: {benchmark_ticker}")
        prices = _prices
        bench_returns = _bench_returns
    else:
        raw_provider = YFinanceProvider()
        provider = CachedDataProvider(raw_provider, cache_dir=str(settings.cache_dir))
        prices = provider.get_prices(tickers, start_date, end_date)
        benchmark_prices = provider.get_prices([benchmark_ticker], start_date, end_date)
        if prices.empty:
            raise ValueError("No price history returned for requested strategy universe.")
        if benchmark_prices.empty or benchmark_ticker not in benchmark_prices.columns:
            raise ValueError(f"No price history returned for benchmark ticker: {benchmark_ticker}")
        bench_returns = benchmark_prices[benchmark_ticker].pct_change().fillna(0.0)

    # Pre-compute asset returns once — reused across all window evaluations (Opt 2)
    _asset_returns = prices.pct_change().fillna(0.0)

    # Dynamic dynamic weights generation from custom strategy function
    weights = generate_signals_fn(prices, flat_config)

    # Sanity validate output weights contract (Finding 14)
    from autobacktest.strategy.contract import validate_output

    ok, err = validate_output(weights, tickers, expected_index=prices.index)
    if not ok:
        raise ValueError(f"Strategy weights validation failed: {err}")

    # Partition holdout data (configurable via AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS)
    in_sample_idx, holdout_idx = partition_holdout_data(prices.index, holdout_years=settings.default_holdout_years)
    if in_sample_idx.empty or holdout_idx.empty:
        raise ValueError("In-sample or holdout period is empty. Ensure the backtest period is sufficiently long.")

    wf_windows = generate_walk_forward_windows(in_sample_idx, train_years=5, test_years=1)
    wf_reports = _run_walk_forward_windows(
        prices,
        weights,
        wf_windows,
        bench_returns,
        asset_returns=_asset_returns,
    )

    # Guard: walk-forward must produce at least one window
    if not wf_reports:
        raise ValueError(
            "No walk-forward windows could be generated from the in-sample period. "
            "Ensure the backtest range is long enough for at least one 5y-train/1y-test fold."
        )

    # Aggregate walk-forward folds into a single in-sample summary
    in_sample_metrics = aggregate_walk_forward(wf_reports)

    # Evaluate holdout window
    holdout_start = holdout_idx.min()
    holdout_end = holdout_idx.max()
    holdout_report = generate_window_report(
        prices,
        weights,
        holdout_start,
        holdout_end,
        bench_returns,
        asset_returns=_asset_returns,
    )

    # Compute holdout net returns
    holdout_prices = prices.loc[holdout_start:holdout_end]
    holdout_weights = weights.loc[holdout_start:holdout_end]
    h_portfolio_returns, _, h_daily_weights = run_vectorized_backtest(
        holdout_prices,
        holdout_weights,
        asset_returns=_asset_returns,
    )
    holdout_net_returns, _, _ = calculate_turnover_and_costs(
        h_portfolio_returns,
        h_daily_weights,
        holdout_prices,
        asset_returns=_asset_returns,
    )

    # Evaluate full period net returns for Monte Carlo and Regime tests
    full_returns, _, daily_weights = run_vectorized_backtest(
        prices,
        weights,
        asset_returns=_asset_returns,
    )
    net_returns, _, _ = calculate_turnover_and_costs(
        full_returns,
        daily_weights,
        prices,
        asset_returns=_asset_returns,
    )

    # Derive in-sample net returns from the full-period series
    in_sample_net_returns = net_returns.reindex(in_sample_idx).dropna()

    # Labeled stress testing regimes
    regime_drawdowns, regime_passed = evaluate_stress_regimes(net_returns)

    # Monte carlo block bootstrap (1000 paths) with seed for strict determinism
    mc_5th, mc_50th, mc_95th = run_block_bootstrap(net_returns, n_paths=1000, seed=42)

    # --- DSR accounting ---
    # Selection DSR uses in-sample walk-forward returns (deflated by config's trial count)
    effective_trials = int(flat_config.get("effective_trials", 1))
    historical_sharpes = flat_config.get("historical_sharpes")

    selection_dsr = calculate_psr_dsr(
        in_sample_net_returns,
        historical_sharpes=historical_sharpes,
        effective_trials=effective_trials,
    )

    # Holdout PSR (will be deflated by _deflate_holdout in the orchestrator)
    holdout_dsr = calculate_psr_dsr(
        holdout_net_returns,
        effective_trials=1,
    )

    # Generate complete stable dataset hash using hashlib
    tickers_str = ",".join(sorted(tickers))
    dataset_hash = hashlib.sha256(tickers_str.encode()).hexdigest()[:16]

    # Generate complete report
    report = EvaluationReport(
        strategy_name=strategy_name,
        dataset_hash=dataset_hash,
        gates_passed={},
        is_accepted=False,
        rejection_reason=None,
        holdout_metrics=holdout_report,
        in_sample_metrics=in_sample_metrics,
        walk_forward_metrics=wf_reports,
        regime_drawdowns=regime_drawdowns,
        regime_passed=regime_passed,
        mc_sharpe_5th=mc_5th,
        mc_sharpe_50th=mc_50th,
        mc_sharpe_95th=mc_95th,
        observed_sharpe=in_sample_metrics.sharpe_ratio,
        effective_trials=effective_trials,
        deflated_sharpe=selection_dsr,
        holdout_deflated_sharpe=holdout_dsr,
        holdout_net_returns=holdout_net_returns,
    )

    # Delegate standalone gate checks to backward-compat accept (hard constraints only)
    from autobacktest.gate import accept as gate_accept

    gate_accept(report, baseline=None, config=flat_config)

    if _eval_cache is not None and _strategy_code is not None:
        try:
            _code_hash = hash(_strategy_code)
            _config_hash = hash(json.dumps(flat_config, sort_keys=True, default=str))
            _ckey = hash((_code_hash, _config_hash))
            _eval_cache[_ckey] = (deepcopy(report), in_sample_net_returns.copy())
        except Exception:
            pass

    return report, in_sample_net_returns


def evaluate_strategy(
    strategy_name: str,
    generate_signals_fn: Any,
    config: dict[str, Any] | StrategyConfig,
    start_date: str = settings.default_start_date,
    end_date: str = settings.default_end_date,
    **kwargs: Any,
) -> EvaluationReport:
    """Run full deterministic walk-forward & holdout evaluation lifecycle.

    Thin wrapper around evaluate_strategy_detailed that returns only the report.
    """
    report, _ = evaluate_strategy_detailed(
        strategy_name,
        generate_signals_fn,
        config,
        start_date,
        end_date,
        **kwargs,
    )
    return report
