#!/usr/bin/env python3
"""Empirical validation script for AutoBacktest engine.

Validates:
1. SPY 30% buy-and-hold Sharpe ratio calculation (Engine vs Independent calculation)
2. Crisis-regime drawdowns (Engine vs Independent calculation)
3. Overfitting battery via CSCV PBO on over-tuned alternatives
4. Gate check for the honest 30% buy-and-hold strategy
5. Cost tracking with impact_coef > 0 vs impact_coef = 0
"""

# ruff: noqa: E402

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add src/ to python path
src_dir = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(src_dir))

from autobacktest.data.cache import CachedDataProvider
from autobacktest.data.yfinance_provider import YFinanceProvider
from autobacktest.evaluator.costs import calculate_turnover_and_costs
from autobacktest.evaluator.cscv import calculate_pbo
from autobacktest.evaluator.evaluate import evaluate_strategy
from autobacktest.evaluator.regime import REGIMES, calculate_max_drawdown
from autobacktest.gate import accept


def run_validation():
    print("======================================================================")
    print("RUNNING EMPIRICAL VALIDATIONS")
    print("======================================================================")

    # 1. Fetch SPY Data
    print("\n[Step 1] Fetching SPY prices for validations...")
    provider = YFinanceProvider()
    cache_dir = Path(__file__).resolve().parent.parent / "data" / "cache"
    cached_provider = CachedDataProvider(provider, cache_dir=str(cache_dir))

    # We need data covering 2007-01-01 to 2023-12-31 to cover all regimes
    tickers = ["SPY"]
    prices = cached_provider.get_prices(tickers, start="2007-01-01", end="2024-01-01")
    if prices.empty:
        print("FAIL: Could not download SPY data.")
        return

    print(f"Loaded {len(prices)} days of SPY prices.")

    # Verification 1: Sharpe Ratio over 2010-2019
    print("\n[Step 2] Validating Sharpe ratio (Engine vs Independent calculation)...")

    # Define simple 30% Buy-and-Hold strategy function to reduce drawdowns
    def bnh_strategy_30(prices_df, _config):
        signals = pd.DataFrame(0.3, index=prices_df.index, columns=["SPY"])
        return signals

    # Evaluate using engine (with zero transaction costs to match raw returns)
    bnh_config = {
        "universe": ["SPY"],
        "benchmark": "SPY",
        "momentum_lookback": 12,
        "max_drawdown_limit": 0.99,
        "turnover_limit": 99.0,
        "borrow_cost_bps": 0.0,
        "commission_bps": 0.0,
        "spread_bps": 0.0,
        "impact_coef": 0.0,
        "enable_holdout_confirmation": True,
    }

    report = evaluate_strategy(
        strategy_name="bnh_spy_30",
        generate_signals_fn=bnh_strategy_30,
        config=bnh_config,
        start_date="2010-01-01",
        end_date="2020-01-01",
    )

    # Manual Selection Sharpe calculation (2015-01-05 to 2016-12-30)
    spy_sel = prices.loc["2015-01-05":"2016-12-30", "SPY"]
    rets_sel = spy_sel.pct_change().dropna() * 0.3
    independent_sel_sharpe = float(rets_sel.mean() / rets_sel.std() * np.sqrt(252))

    # Manual Holdout Sharpe calculation (2017-01-03 to 2019-12-31)
    spy_hold = prices.loc["2017-01-03":"2019-12-31", "SPY"]
    rets_hold = spy_hold.pct_change().dropna() * 0.3
    independent_hold_sharpe = float(rets_hold.mean() / rets_hold.std() * np.sqrt(252))

    engine_sel_sharpe = report.in_sample_metrics.sharpe_ratio
    engine_hold_sharpe = report.holdout_metrics.sharpe_ratio

    print("Selection Window (2015-2016):")
    print(f"  Independent Sharpe: {independent_sel_sharpe:.4f}")
    print(f"  Engine Sharpe:      {engine_sel_sharpe:.4f}")
    print("Holdout Window (2017-2019):")
    print(f"  Independent Sharpe: {independent_hold_sharpe:.4f}")
    print(f"  Engine Sharpe:      {engine_hold_sharpe:.4f}")

    diff_sel = abs(independent_sel_sharpe - engine_sel_sharpe)
    diff_hold = abs(independent_hold_sharpe - engine_hold_sharpe)

    # Tolerances are small (accounting for daily compounding vs simple returns)
    if diff_sel < 0.02 and diff_hold < 0.02:
        print("RESULT 1: PASS")
    else:
        print(f"RESULT 1: FAIL (Diff Sel: {diff_sel:.4f}, Diff Hold: {diff_hold:.4f})")

    # Verification 2: Crisis Regimes Drawdowns
    print("\n[Step 3] Validating crisis-regime drawdowns...")
    # Evaluate over longer window to capture all 3 regimes
    bnh_config_long = bnh_config.copy()
    bnh_config_long["max_drawdown_limit"] = 0.99
    report_long = evaluate_strategy(
        strategy_name="bnh_spy_30_long",
        generate_signals_fn=bnh_strategy_30,
        config=bnh_config_long,
        start_date="2007-01-01",
        end_date="2023-12-31",
    )

    regime_results = report_long.regime_drawdowns
    all_passed = True
    for name, (start, end, _limit) in REGIMES.items():
        # Manual drawdown recompute
        sub_prices = prices.loc[start:end, "SPY"]
        if sub_prices.empty:
            print(f"Skipping {name}: no price data.")
            continue
        # Strategy return is 0.3 * SPY return
        sub_rets = sub_prices.pct_change().fillna(0.0) * 0.3
        sub_equity = (1.0 + sub_rets).cumprod()
        manual_dd = calculate_max_drawdown(sub_equity)
        engine_dd = regime_results.get(name, 0.0)

        print(f"Regime {name}:")
        print(f"  Manual Max Drawdown: {manual_dd:.4f}")
        print(f"  Engine Max Drawdown: {engine_dd:.4f}")
        if abs(manual_dd - engine_dd) < 0.01:
            print("  Result: PASS")
        else:
            print("  Result: FAIL")
            all_passed = False

    if all_passed:
        print("RESULT 2: PASS")
    else:
        print("RESULT 2: FAIL")

    # Verification 3: Overfitting Battery via CSCV PBO
    print("\n[Step 4] Validating CSCV PBO on over-tuned alternatives...")
    # Generate 20 random walk daily returns to represent over-fitted strategy alternatives
    np.random.seed(42)
    n_days = 500
    n_strategies = 30

    # Base strategy has no edge
    base_returns = np.random.normal(0.0, 0.01, n_days)

    # Alternatives are pertubations of base returns with noise, representing parameterized search
    alt_returns = []
    for _ in range(n_strategies):
        noise = np.random.normal(0.0, 0.005, n_days)
        # Add slight bias to select candidates to simulate overfitting selection
        alt_returns.append(base_returns + noise)

    alt_df = pd.DataFrame(alt_returns).T

    # Calculate PBO
    pbo = calculate_pbo(alt_df, n_blocks=10)
    print(f"CSCV Probability of Backtest Overfitting (PBO): {pbo}")
    if pbo is not None and pbo > 0.30:
        print("RESULT 3: PASS (High overfitting detected successfully)")
    else:
        print(f"RESULT 3: FAIL (PBO {pbo} too low or None for random search)")

    # Verification 4: Gate Check
    print("\n[Step 5] Validating Gate behavior...")
    # An honest 30% buy-and-hold strategy should pass gates because its drawdowns are scaled down
    res = accept(report_long, baseline=None, dd_limit=0.25, turnover_limit=2.0)
    print(f"Gate accept (no baseline): {res.accepted} (Reason: {res.reason})")
    if res.accepted:
        print("RESULT 4: PASS")
    else:
        print("RESULT 4: FAIL")

    # Verification 5: cost tracking with impact_coef
    print("\n[Step 6] Validating Cost model with impact_coef...")
    # High turnover strategy: rebalance daily to alternate weights
    dates = pd.date_range("2023-01-01", periods=100)
    price_series = pd.DataFrame({"AssetA": np.linspace(100, 105, 100)}, index=dates)
    daily_rets = pd.Series([0.01] * 100, index=dates)

    # Alternating weights (1.0 vs 0.0) -> high turnover
    weights = pd.DataFrame({"AssetA": [1.0 if idx % 2 == 0 else 0.0 for idx in range(100)]}, index=dates)

    # 1. No impact_coef
    net_ret_0, _, _ = calculate_turnover_and_costs(
        daily_rets,
        weights,
        price_series,
        commission_bps=0.0,
        spread_bps=0.0,
        impact_coef=0.0,
        borrow_cost_bps=0.0,
    )

    # 2. Positive impact_coef
    net_ret_pos, _, _ = calculate_turnover_and_costs(
        daily_rets,
        weights,
        price_series,
        commission_bps=0.0,
        spread_bps=0.0,
        impact_coef=5.0,
        borrow_cost_bps=0.0,
    )

    mean_ret_0 = net_ret_0.mean()
    mean_ret_pos = net_ret_pos.mean()
    print(f"Mean return with impact_coef=0:   {mean_ret_0:.6f}")
    print(f"Mean return with impact_coef=5.0: {mean_ret_pos:.6f}")

    if mean_ret_pos < mean_ret_0:
        print("RESULT 5: PASS")
    else:
        print("RESULT 5: FAIL")

    print("\n======================================================================")
    print("ALL EMPIRICAL VALIDATIONS COMPLETED")
    print("======================================================================")


if __name__ == "__main__":
    run_validation()
