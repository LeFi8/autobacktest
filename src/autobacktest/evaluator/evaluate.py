"""Orchestration of walk-forward and holdout backtest evaluations."""

from __future__ import annotations

import json
import logging
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd

from autobacktest.config import settings
from autobacktest.data.cache import CachedDataProvider
from autobacktest.evaluator.backtest import run_vectorized_backtest
from autobacktest.evaluator.costs import calculate_turnover_and_costs
from autobacktest.evaluator.deflated_sharpe import calculate_psr_dsr
from autobacktest.evaluator.engine import (
    _CacheProtocol as _CacheProtocol,
)
from autobacktest.evaluator.engine import (
    _run_walk_forward_windows as _run_walk_forward_windows,
)
from autobacktest.evaluator.engine import (
    compute_dataset_hash as compute_dataset_hash,
)
from autobacktest.evaluator.engine import (
    generate_window_report as generate_window_report,
)
from autobacktest.evaluator.holdout import partition_holdout_data

# Re-expose helper functions to maintain test backward compatibility
from autobacktest.evaluator.metrics import (
    _compute_returns_metrics as _compute_returns_metrics,
)
from autobacktest.evaluator.metrics import (
    calculate_information_ratio as calculate_information_ratio,
)
from autobacktest.evaluator.metrics import (
    calculate_sortino_ratio as calculate_sortino_ratio,
)
from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.evaluator.stress_testing import (
    get_regime_haircut as get_regime_haircut,
)
from autobacktest.evaluator.stress_testing import (
    run_stress_and_bootstrap_tests as run_stress_and_bootstrap_tests,
)
from autobacktest.evaluator.walk_forward import generate_walk_forward_windows
from autobacktest.strategy.config_schema import StrategyConfig

logger = logging.getLogger(__name__)


def _flatten_config(config: dict[str, Any] | StrategyConfig) -> dict[str, Any]:
    """Normalise *config* to a flat ``dict`` for downstream evaluation.

    When *config* is a :class:`StrategyConfig`, delegates to its
    ``to_flat_dict`` method.  Otherwise merges the top-level ``params``
    sub-dict into the root dict.

    Args:
        config: Strategy configuration as a raw dict or a Pydantic model.

    Returns:
        dict[str, Any]: Flat key-value mapping ready for signal generation.
    """
    if isinstance(config, StrategyConfig):
        return config.to_flat_dict()
    flat_config = dict(config)
    params = flat_config.get("params", {})
    if isinstance(params, dict):
        for k, v in params.items():
            if k not in flat_config:
                flat_config[k] = v
    return flat_config


def _fetch_and_join_prices(
    tickers: list[str],
    benchmark_ticker: str,
    regime_bench_ticker: str,
    start_date: str,
    end_date: str,
) -> tuple[pd.DataFrame, pd.Series]:
    """Download, validate, and join all required price series from YFinance.

    Fetches the strategy universe, benchmark, and regime-benchmark prices,
    raising ``ValueError`` when any required ticker returns no data.  The
    benchmark and regime-benchmark columns are joined into the main frame
    when they are not already present.

    Args:
        tickers: Strategy asset tickers.
        benchmark_ticker: Benchmark ticker symbol.
        regime_bench_ticker: Regime-benchmark ticker (may equal *benchmark_ticker*).
        start_date: Start of the data range.
        end_date: End of the data range.

    Returns:
        tuple: ``(prices, bench_returns)``.

    Raises:
        ValueError: When any required ticker returns an empty price series.
    """
    from autobacktest.data.yfinance_provider import YFinanceProvider

    raw_provider = YFinanceProvider()
    provider = CachedDataProvider(raw_provider, cache_dir=str(settings.cache_dir))
    prices = provider.get_prices(tickers, start_date, end_date)
    benchmark_prices = provider.get_prices([benchmark_ticker], start_date, end_date)
    if prices.empty:
        raise ValueError("No price history returned for requested strategy universe.")
    if benchmark_prices.empty or benchmark_ticker not in benchmark_prices.columns:
        raise ValueError(f"No price history returned for benchmark ticker: {benchmark_ticker}")
    bench_returns = benchmark_prices[benchmark_ticker].pct_change().fillna(0.0)

    if benchmark_ticker not in prices.columns:
        prices = prices.join(benchmark_prices[[benchmark_ticker]], how="left")

    if regime_bench_ticker == benchmark_ticker:
        regime_bench_prices = benchmark_prices
    else:
        regime_bench_prices = provider.get_prices([regime_bench_ticker], start_date, end_date)
        if regime_bench_prices.empty or regime_bench_ticker not in regime_bench_prices.columns:
            raise ValueError(f"No price history returned for regime benchmark ticker: {regime_bench_ticker}")

    if regime_bench_ticker not in prices.columns:
        prices = prices.join(regime_bench_prices[[regime_bench_ticker]], how="left")

    return prices, bench_returns


def _load_evaluation_data(
    flat_config: dict[str, Any],
    start_date: str,
    end_date: str,
    _prices: pd.DataFrame | None,
    _bench_returns: pd.Series | None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Return ``(prices, bench_returns)`` for the evaluation period.

    When *_prices* and *_bench_returns* are provided, validates that all
    required tickers are present and returns them directly.  Otherwise
    downloads data from YFinance via :func:`_fetch_and_join_prices`.

    Args:
        flat_config: Flat strategy configuration dict.
        start_date: Start of the data range.
        end_date: End of the data range.
        _prices: Optional pre-fetched price DataFrame.
        _bench_returns: Optional pre-fetched benchmark return series.

    Returns:
        tuple: ``(prices, bench_returns)``.

    Raises:
        ValueError: When required tickers are missing or data is empty.
    """
    tickers = flat_config.get("universe", [])
    benchmark_ticker = flat_config.get("benchmark", "SPY")
    regime_bench_ticker = flat_config.get("regime_benchmark") or benchmark_ticker

    if _prices is not None and _bench_returns is not None:
        missing_assets = [t for t in tickers if t not in _prices.columns]
        if missing_assets:
            raise ValueError(f"Cached prices missing tickers: {missing_assets}")
        if benchmark_ticker not in _prices.columns:
            raise ValueError(f"Cached prices missing benchmark ticker: {benchmark_ticker}")
        if regime_bench_ticker not in _prices.columns:
            raise ValueError(f"Cached prices missing regime benchmark ticker: {regime_bench_ticker}")
        return _prices, _bench_returns

    return _fetch_and_join_prices(tickers, benchmark_ticker, regime_bench_ticker, start_date, end_date)


def _generate_wf_weights(
    prices: pd.DataFrame,
    generate_signals_fn: Any,
    flat_config: dict[str, Any],
    tickers: list[str],
    wf_windows: list[tuple[Any, Any, Any, Any]],
) -> pd.DataFrame:
    """Generate concatenated walk-forward test-period weights.

    Runs ``generate_signals_fn`` on each fold (up to ``test_end``) and
    splices the test-segment weights together, deduplicating any overlapping
    index rows.

    Args:
        prices: Full price DataFrame.
        generate_signals_fn: The strategy signal function.
        flat_config: Flat strategy configuration dict.
        tickers: Asset tickers in the universe.
        wf_windows: Walk-forward window list of ``(train_start, train_end, test_start, test_end)``.

    Returns:
        pd.DataFrame: Concatenated and deduplicated test-period weights.

    Raises:
        ValueError: When any fold's weights fail output validation.
    """
    from autobacktest.strategy.contract import validate_output

    wf_test_weights_list: list[pd.DataFrame] = []
    for _train_start, _train_end, test_start, test_end in wf_windows:
        prices_truncated = prices.loc[:test_end]
        fold_weights = generate_signals_fn(prices_truncated, flat_config)
        ok, err = validate_output(fold_weights, tickers, expected_index=prices.index)
        if not ok:
            raise ValueError(f"Strategy weights validation failed at fold test={test_start}: {err}")
        wf_test_weights_list.append(fold_weights.loc[test_start:test_end])

    wf_weights = pd.concat(wf_test_weights_list)
    return wf_weights[~wf_weights.index.duplicated(keep="first")]


def _apply_regime_haircut(
    haircut: float,
    in_sample_metrics: WindowReport,
    holdout_report: WindowReport,
    wf_reports: list[WindowReport],
) -> None:
    """Scale return and risk-adjusted metrics by ``(1 - haircut)`` in-place.

    Applied when the regime-benchmark haircut is non-zero to penalise
    strategies evaluated in harsh market regimes.

    Args:
        haircut: Fraction by which to reduce all ratio metrics (0 = no effect).
        in_sample_metrics: In-sample aggregate ``WindowReport``.
        holdout_report: Holdout-period ``WindowReport``.
        wf_reports: Per-fold walk-forward ``WindowReport`` list.
    """
    factor = 1.0 - haircut
    in_sample_metrics.annualized_return *= factor
    in_sample_metrics.sharpe_ratio *= factor
    in_sample_metrics.sortino_ratio *= factor
    holdout_report.annualized_return *= factor
    holdout_report.sharpe_ratio *= factor
    holdout_report.sortino_ratio *= factor
    for r in wf_reports:
        r.annualized_return *= factor
        r.sharpe_ratio *= factor
        r.sortino_ratio *= factor


def _check_eval_cache(
    flat_config: dict[str, Any],
    _eval_cache: _CacheProtocol | None,
    _strategy_code: str | None,
) -> tuple[EvaluationReport, pd.Series] | None:
    """Return a cached evaluation result when an identical edit was previously evaluated.

    Normalises the strategy code, hashes both code and config, and checks the
    memoisation dict.  Silently skips on any exception (e.g. missing normaliser).

    Args:
        flat_config: Flat strategy configuration dict.
        _eval_cache: Memoisation dict keyed by ``(code_hash, config_hash)``.
        _strategy_code: Raw strategy source used to compute the code hash.

    Returns:
        tuple ``(report, returns)`` on a cache hit, or ``None`` on a miss.
    """
    if _eval_cache is None or _strategy_code is None:
        return None
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
    return None


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

    Evaluates a strategy by running walk-forward windows (5y train / 1y test)
    on an in-sample period, then running a holdout window on the out-of-sample
    tail.  Regime haircuts, Monte Carlo bootstrap, and stress tests are applied
    automatically.

    Args:
        strategy_name: Name of the strategy being evaluated.
        generate_signals_fn: The strategy's ``generate_signals(prices, config)`` function.
        config: Strategy configuration as dict or ``StrategyConfig`` instance.
        start_date: Start of the full evaluation period (inclusive).
        end_date: End of the full evaluation period (exclusive).

    Keyword Args:
        _prices: Pre-fetched price DataFrame (skips YFinance download).
        _bench_returns: Pre-fetched benchmark return series.
        _eval_cache: Memoisation dict to skip re-evaluation of identical edits.
        _strategy_code: Source text required for the eval cache key.

    Returns:
        tuple[EvaluationReport, pd.Series]: ``(report, in_sample_net_returns)``.
        The report carries all aggregated metrics, holdout returns, and
        robustness diagnostics.  The returns series is used for DSR deflation
        and diversity correlation checks.

    Raises:
        ValueError: When prices are empty, in-sample/holdout periods are
            empty, or walk-forward windows cannot be generated.
    """
    flat_config = _flatten_config(config)

    cached = _check_eval_cache(flat_config, _eval_cache, _strategy_code)
    if cached is not None:
        return cached

    tickers = flat_config.get("universe", [])
    benchmark_ticker = flat_config.get("benchmark", "SPY")
    regime_bench_ticker = flat_config.get("regime_benchmark") or benchmark_ticker

    prices, bench_returns = _load_evaluation_data(flat_config, start_date, end_date, _prices, _bench_returns)
    _asset_returns = prices.pct_change().fillna(0.0)

    in_sample_idx, holdout_idx = partition_holdout_data(prices.index, holdout_years=settings.default_holdout_years)
    if in_sample_idx.empty or holdout_idx.empty:
        raise ValueError("In-sample or holdout period is empty. Ensure the backtest period is sufficiently long.")

    from autobacktest.strategy.contract import validate_output

    wf_windows = generate_walk_forward_windows(in_sample_idx, train_years=5, test_years=1)
    if not wf_windows:
        raise ValueError(
            "No walk-forward windows could be generated from the in-sample period. "
            "Ensure the backtest range is long enough for at least one 5y-train/1y-test fold."
        )

    wf_weights = _generate_wf_weights(prices, generate_signals_fn, flat_config, tickers, wf_windows)

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
        borrow_cost_bps=flat_config.get("borrow_cost_bps", 100.0),
        adaptive_slippage=flat_config.get("adaptive_slippage", False),
        slippage_vol_window=flat_config.get("slippage_vol_window", 21),
        slippage_vol_cap=flat_config.get("slippage_vol_cap", 3.0),
    )

    wf_start = wf_windows[0][2]
    wf_end = wf_windows[-1][3]
    wf_net_returns = wf_net_returns.loc[wf_start:wf_end]
    wf_net_equity = wf_net_equity.loc[wf_start:wf_end]

    if wf_net_returns.empty:
        raise ValueError("Walk-forward test returns are empty after backtest.")

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

    wf_reports = _run_walk_forward_windows(
        prices,
        wf_weights,
        wf_windows,
        bench_returns,
        asset_returns=_asset_returns,
        borrow_cost_bps=flat_config.get("borrow_cost_bps", 100.0),
        adaptive_slippage=flat_config.get("adaptive_slippage", False),
        slippage_vol_window=flat_config.get("slippage_vol_window", 21),
        slippage_vol_cap=flat_config.get("slippage_vol_cap", 3.0),
    )

    weights = generate_signals_fn(prices, flat_config)
    ok, err = validate_output(weights, tickers, expected_index=prices.index)
    if not ok:
        raise ValueError(f"Strategy weights validation failed (full period): {err}")

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
        borrow_cost_bps=flat_config.get("borrow_cost_bps", 100.0),
        adaptive_slippage=flat_config.get("adaptive_slippage", False),
        slippage_vol_window=flat_config.get("slippage_vol_window", 21),
        slippage_vol_cap=flat_config.get("slippage_vol_cap", 3.0),
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
        borrow_cost_bps=flat_config.get("borrow_cost_bps", 100.0),
        adaptive_slippage=flat_config.get("adaptive_slippage", False),
        slippage_vol_window=flat_config.get("slippage_vol_window", 21),
        slippage_vol_cap=flat_config.get("slippage_vol_cap", 3.0),
    )

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
        borrow_cost_bps=flat_config.get("borrow_cost_bps", 100.0),
        adaptive_slippage=flat_config.get("adaptive_slippage", False),
        slippage_vol_window=flat_config.get("slippage_vol_window", 21),
        slippage_vol_cap=flat_config.get("slippage_vol_cap", 3.0),
    )

    regime_drawdowns, regime_passed, mc_5th, mc_50th, mc_95th, mc_sharpes = run_stress_and_bootstrap_tests(
        net_returns,
        daily_weights=daily_weights,
        n_tickers=len(tickers),
        mc_bootstrap_method=flat_config.get("mc_bootstrap_method", "stationary"),
    )

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

    benchmark_in_sample = _compute_returns_metrics(bench_returns, wf_start, wf_end)
    benchmark_holdout_m = _compute_returns_metrics(bench_returns, holdout_start, holdout_end)

    haircut = get_regime_haircut(prices[regime_bench_ticker], holdout_start)
    if haircut > 0.0:
        _apply_regime_haircut(haircut, in_sample_metrics, holdout_report, wf_reports)

    dataset_hash = compute_dataset_hash(
        tickers,
        start_date=start_date,
        end_date=end_date,
        holdout_years=settings.default_holdout_years,
    )

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

    Thin wrapper around ``evaluate_strategy_detailed`` that discards the
    in-sample returns series and returns only the ``EvaluationReport``.

    Args:
        strategy_name: Name of the strategy being evaluated.
        generate_signals_fn: The strategy's ``generate_signals`` function.
        config: Strategy configuration (dict or ``StrategyConfig``).
        start_date: Start of the backtest period.
        end_date: End of the backtest period.
        **kwargs: Passed through to ``evaluate_strategy_detailed``.

    Returns:
        EvaluationReport: Full evaluation report with all metrics.
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
