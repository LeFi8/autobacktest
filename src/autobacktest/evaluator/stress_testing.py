"""Regime stress testing and Monte Carlo block bootstrap analysis wrappers."""

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
    """Run stress regime and bootstrap simulations."""
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
    """Calculate the launch regime haircut based on benchmark prices."""
    return calculate_regime_haircut(benchmark_prices, holdout_start)
