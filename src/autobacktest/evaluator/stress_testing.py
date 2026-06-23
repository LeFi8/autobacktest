"""Regime stress testing and Monte Carlo block bootstrap analysis wrappers.

Combines two complementary robustness checks:
1. **Regime stress tests** — measure strategy drawdowns during historical
   crash periods (2008 GFC, 2020 COVID, 2022 bear market) to assess
   tail-risk resilience.
2. **Monte Carlo block bootstrap** — resample daily returns via stationary
   or circular block bootstrap to estimate the distribution of Sharpe
   ratios under resampling, providing confidence intervals and p-values.

Both methods use seed=42 for reproducibility.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from autobacktest.evaluator.monte_carlo import run_block_bootstrap
from autobacktest.evaluator.regime import calculate_regime_haircut, evaluate_stress_regimes


def run_stress_and_bootstrap_tests(
    net_returns: pd.Series,
    daily_weights: pd.DataFrame,
    n_tickers: int,
    mc_bootstrap_method: str = "stationary",
) -> tuple[dict[str, float], bool, float, float, float, np.ndarray]:
    """Run stress regime and bootstrap simulations.

    Evaluates drawdowns during historical crash regimes (2008 GFC, 2020
    COVID, 2022 bear) and computes Monte Carlo Sharpe percentiles via
    stationary block bootstrap.

    Args:
        net_returns: Daily net returns series.
        daily_weights: Daily asset weights DataFrame.
        n_tickers: Number of tickers in the universe.
        mc_bootstrap_method: Bootstrap method (``"stationary"`` or ``"circular"``).

    Returns:
        tuple: ``(regime_drawdowns, regime_passed, mc_5th, mc_50th, mc_95th, mc_sharpes)``.
    """
    regime_drawdowns, regime_passed = evaluate_stress_regimes(
        net_returns,
        daily_weights=daily_weights,
        n_tickers=n_tickers,
    )
    mc_5th, mc_50th, mc_95th, mc_sharpes = run_block_bootstrap(
        net_returns,
        n_paths=1000,
        seed=42,
        method=mc_bootstrap_method,
    )
    return regime_drawdowns, regime_passed, mc_5th, mc_50th, mc_95th, mc_sharpes


def get_regime_haircut(
    benchmark_prices: pd.Series,
    holdout_start: pd.Timestamp,
) -> float:
    """Calculate the launch regime haircut based on benchmark prices.

    Convenience wrapper around ``calculate_regime_haircut``.  Applies a
    proportional penalty to performance metrics when the strategy launches
    at a cyclical peak in the benchmark.

    Args:
        benchmark_prices: Historical daily prices of the benchmark.
        holdout_start: Strategy launch date (start of holdout period).

    Returns:
        float: Haircut fraction (0.0 when no peak detected).
    """
    return calculate_regime_haircut(benchmark_prices, holdout_start)
