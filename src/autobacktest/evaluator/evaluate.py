"""Orchestration of walk-forward and holdout backtest evaluations."""

from __future__ import annotations

import hashlib
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from typing import Any, Protocol

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

logger = logging.getLogger(__name__)


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
        information_ratio=0.0,  # N/A for standalone benchmark self-comparison
    )


class _CacheProtocol(Protocol):
    """Minimal eval-result cache interface — satisfies both ``dict`` and ``_LRUCache``."""

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

    Includes start/end dates and holdout years so that the same universe with
    different time ranges produces distinct hashes.

    Args:
        tickers: Asset tickers in the universe.
        start_date: Backtest start date string (YYYY-MM-DD).
        end_date: Backtest end date string (YYYY-MM-DD).
        holdout_years: Number of years reserved for holdout.

    Returns:
        str: 16-character hex hash digest.
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
    _eval_cache: _CacheProtocol | None = None,
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
            from autobacktest.strategy.normalization import normalize_python_code

            _norm_code = normalize_python_code(_strategy_code)
            _code_hash = hash(_norm_code)
            _config_hash = hash(json.dumps(flat_config, sort_keys=True, default=str))
            _ckey = hash((_code_hash, _config_hash))
            cached = _eval_cache.get(_ckey)
            if cached is not None:
                cached_report, cached_returns = cached
                return deepcopy(cached_report), cached_returns.copy()
        except Exception as e:
            logger.warning("Eval cache read failed: %s", e)

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

    # Partition holdout data (configurable via AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS)
    in_sample_idx, holdout_idx = partition_holdout_data(prices.index, holdout_years=settings.default_holdout_years)
    if in_sample_idx.empty or holdout_idx.empty:
        raise ValueError("In-sample or holdout period is empty. Ensure the backtest period is sufficiently long.")

    # ------------------------------------------------------------------
    # Causal walk-forward: generate signals per fold, concat test weights
    # ------------------------------------------------------------------
    from autobacktest.strategy.contract import validate_output

    wf_windows = generate_walk_forward_windows(in_sample_idx, train_years=5, test_years=1)
    if not wf_windows:
        raise ValueError(
            "No walk-forward windows could be generated from the in-sample period. "
            "Ensure the backtest range is long enough for at least one 5y-train/1y-test fold."
        )

    wf_test_weights_list: list[pd.DataFrame] = []
    for _train_start, _train_end, test_start, test_end in wf_windows:
        prices_truncated = prices.loc[:test_end]
        fold_weights = generate_signals_fn(prices_truncated, flat_config)
        ok, err = validate_output(fold_weights, tickers, expected_index=prices.index)
        if not ok:
            raise ValueError(f"Strategy weights validation failed at fold test={test_start}: {err}")
        test_segment = fold_weights.loc[test_start:test_end]
        wf_test_weights_list.append(test_segment)

    # Concatenate all test-period weights into one continuous series
    wf_weights = pd.concat(wf_test_weights_list)
    wf_weights = wf_weights[~wf_weights.index.duplicated(keep="first")]

    # Single continuous backtest on pooled walk-forward weights (shifted once across entire series)
    wf_portfolio_returns, _wf_equity, wf_daily_weights = run_vectorized_backtest(
        prices,
        wf_weights,
        asset_returns=_asset_returns,
    )
    wf_net_returns, wf_net_equity, wf_turnover = calculate_turnover_and_costs(
        wf_portfolio_returns,
        wf_daily_weights,
        prices,
        asset_returns=_asset_returns,
    )

    # Slice pooled returns to the contiguous walk-forward test-window span
    # to avoid leading zeros (pre-first-window cash) and holdout leakage.
    wf_start = wf_windows[0][2]
    wf_end = wf_windows[-1][3]
    wf_net_returns = wf_net_returns.loc[wf_start:wf_end]
    wf_net_equity = wf_net_equity.loc[wf_start:wf_end]

    if wf_net_returns.empty:
        raise ValueError("Walk-forward test returns are empty after backtest.")

    # --- Pooled metrics from the continuous walk-forward stream ---
    wf_mean = wf_net_returns.mean()
    wf_std = wf_net_returns.std(ddof=1) if len(wf_net_returns) >= 2 else 0.0
    wf_total_growth = wf_net_equity.iloc[-1] if not wf_net_equity.empty else 1.0
    wf_ann_ret = float(wf_total_growth ** (252.0 / len(wf_net_returns)) - 1.0) if wf_total_growth > 0 else -1.0
    wf_ann_vol = float(wf_std * np.sqrt(252))
    pooled_sharpe = float((wf_mean / wf_std * np.sqrt(252)) if wf_std > 0 else 0.0)
    pooled_sortino = calculate_sortino_ratio(wf_net_returns)
    running_max = wf_net_equity.cummax()
    wf_drawdowns = (wf_net_equity - running_max) / running_max
    pooled_max_dd = float(abs(wf_drawdowns.min())) if not wf_drawdowns.empty else 0.0
    pooled_ir = calculate_information_ratio(wf_net_returns, bench_returns)

    in_sample_metrics = WindowReport(
        start_date=wf_windows[0][2].strftime("%Y-%m-%d"),
        end_date=wf_windows[-1][3].strftime("%Y-%m-%d"),
        annualized_return=wf_ann_ret,
        annualized_volatility=wf_ann_vol,
        sharpe_ratio=pooled_sharpe,
        sortino_ratio=pooled_sortino,
        max_drawdown=pooled_max_dd,
        turnover=wf_turnover,
        information_ratio=pooled_ir,
    )

    # Per-fold sub-reports for diagnostics (on the pooled causal weights)
    wf_reports = _run_walk_forward_windows(
        prices,
        wf_weights,
        wf_windows,
        bench_returns,
        asset_returns=_asset_returns,
    )

    # ------------------------------------------------------------------
    # Full period signal generation for holdout + regime + MC evaluation
    # ------------------------------------------------------------------
    weights = generate_signals_fn(prices, flat_config)
    ok, err = validate_output(weights, tickers, expected_index=prices.index)
    if not ok:
        raise ValueError(f"Strategy weights validation failed (full period): {err}")

    # Holdout evaluation
    holdout_start = holdout_idx.min()
    holdout_end = holdout_idx.max()
    bench_holdout = bench_returns.loc[holdout_start:holdout_end]
    holdout_report = generate_window_report(
        prices,
        weights,
        holdout_start,
        holdout_end,
        bench_returns,
        asset_returns=_asset_returns,
    )

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

    # Full period evaluation for Monte Carlo and Regime tests
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

    regime_drawdowns, regime_passed = evaluate_stress_regimes(
        net_returns,
        daily_weights=daily_weights,
        n_tickers=len(tickers),
    )
    mc_5th, mc_50th, mc_95th, mc_sharpes = run_block_bootstrap(net_returns, n_paths=1000, seed=42)

    # --- DSR accounting ---
    # Selection DSR uses POOLED walk-forward returns (same basis as observed_sharpe)
    # NOTE: effective_trials and historical_sharpes are intentionally NOT read
    # from config — the only legitimate source is ledger transaction history
    # via the orchestrator's _deflate() call. Standalone evaluate uses PSR.
    effective_trials = 1
    historical_sharpes = None

    in_sample_net_returns = wf_net_returns
    selection_dsr = calculate_psr_dsr(
        wf_net_returns,
        historical_sharpes=historical_sharpes,
        effective_trials=effective_trials,
    )

    holdout_dsr = calculate_psr_dsr(
        holdout_net_returns,
        effective_trials=1,
    )

    # Compute benchmark metrics for both periods
    benchmark_in_sample = _compute_returns_metrics(bench_returns, in_sample_idx.min(), in_sample_idx.max())
    benchmark_holdout_m = _compute_returns_metrics(bench_returns, holdout_start, holdout_end)

    # Generate complete stable dataset hash including date parameters
    dataset_hash = compute_dataset_hash(
        tickers,
        start_date=start_date,
        end_date=end_date,
        holdout_years=settings.default_holdout_years,
    )

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
        mc_sharpes=mc_sharpes,
        observed_sharpe=pooled_sharpe,
        effective_trials=effective_trials,
        deflated_sharpe=selection_dsr,
        holdout_deflated_sharpe=holdout_dsr,
        holdout_net_returns=holdout_net_returns,
        benchmark_returns=bench_holdout,
        benchmark_ticker=benchmark_ticker,
        benchmark_in_sample_metrics=benchmark_in_sample,
        benchmark_holdout_metrics=benchmark_holdout_m,
    )

    # Delegate standalone gate checks to backward-compat accept (hard constraints only)
    from autobacktest.gate import accept as gate_accept

    gate_accept(report, baseline=None, config=flat_config)

    if _eval_cache is not None and _strategy_code is not None:
        try:
            from autobacktest.strategy.normalization import normalize_python_code

            _norm_code = normalize_python_code(_strategy_code)
            _code_hash = hash(_norm_code)
            _config_hash = hash(json.dumps(flat_config, sort_keys=True, default=str))
            _ckey = hash((_code_hash, _config_hash))
            _eval_cache[_ckey] = (deepcopy(report), in_sample_net_returns.copy())
        except Exception as e:
            logger.warning("Eval cache write failed: %s", e)

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
