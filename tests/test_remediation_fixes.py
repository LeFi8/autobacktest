"""Unit tests verifying the PR review fixes for regime evaluation and HAA strategy."""

import numpy as np
import pandas as pd

from autobacktest.evaluator.regime import evaluate_stress_regimes
from strategies.haa import generate_signals


def test_regime_ends_in_low_exposure() -> None:
    """Verifies that low-exposure runs ending at the series boundary are detected."""
    # COVID regime: "2020-02-20" to "2020-04-30"
    dates = pd.date_range("2020-02-20", "2020-04-30", freq="D")

    # Positive returns so drawdown limit is not violated
    returns = pd.Series(0.0001, index=dates)

    # 2 assets, 1 gets weight 0.05, other gets 0.05, sum = 0.10 (low exposure)
    daily_weights = pd.DataFrame(0.0, index=dates, columns=["SPY", "TLT"])

    # First half has normal exposure
    half_len = len(dates) // 2
    daily_weights.iloc[:half_len] = 0.5

    # Second half ends in low exposure (e.g. 0.05 per asset, sum = 0.10 < 0.20)
    daily_weights.iloc[half_len:] = 0.05

    # Evaluate stress regimes with universe of 3 tickers so it hard rejects
    # Let's check if the low exposure is flagged correctly!
    _drawdowns, passed = evaluate_stress_regimes(returns, daily_weights=daily_weights, n_tickers=3)

    # Since the low-exposure run lasted for half the series (well over 10 days limit),
    # it should HARD REJECT (passed=False)
    assert not passed


def test_haa_missing_defensive_assets() -> None:
    """Verifies HAA strategy does not crash when defensive assets are missing from the inputs."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    # Missing BIL and IEF from columns, only has SPY, VEA, VWO, VNQ, DBC, TLT, TIP
    columns = ["SPY", "VEA", "VWO", "VNQ", "DBC", "TLT", "TIP"]

    # Create simple prices series
    prices = pd.DataFrame(100.0, index=dates, columns=columns)

    # Add a bit of upward trend so canary TIP has positive momentum
    prices["TIP"] = np.linspace(100.0, 110.0, len(dates))

    # Standard HAA config
    config = {
        "params": {
            "offensive_assets": ["SPY", "VEA", "VWO", "VNQ", "DBC", "TLT"],
            "defensive_assets": ["BIL", "IEF"],
            "canary_asset": "TIP",
        }
    }

    # Since TIP momentum is positive (clear skies), it will rank offensive assets, choose top 4.
    # Since prices are flat (100.0) for offensive assets, momentum will be <= 0.
    # Therefore, they will try to allocate to best defensive asset.
    # Since defensive assets (BIL, IEF) are missing, best_def should be None, and it should run without crashing.
    weights = generate_signals(prices, config)

    # No crash! All dates should have weights computed.
    assert not weights.empty


def test_haa_standard_execution() -> None:
    """Verifies HAA strategy with iloc lookup executes correctly under standard conditions."""
    dates = pd.date_range("2023-01-01", "2023-12-31", freq="D")
    columns = ["SPY", "VEA", "VWO", "VNQ", "DBC", "TLT", "TIP", "BIL", "IEF"]
    prices = pd.DataFrame(100.0, index=dates, columns=columns)

    # upward trend for all to ensure positive momentum
    for c in columns:
        prices[c] = np.linspace(100.0, 120.0, len(dates))

    config = {
        "params": {
            "offensive_assets": ["SPY", "VEA", "VWO", "VNQ", "DBC", "TLT"],
            "defensive_assets": ["BIL", "IEF"],
            "canary_asset": "TIP",
        }
    }

    weights = generate_signals(prices, config)
    assert not weights.empty
    # Only check dates where signals were actually generated (skip the initial warm-up period
    # where weights are 0 because lookback data is not yet available or the strategy returns
    # a full price-index DataFrame with ffill from the first rebalance).
    active_dates = weights.index[weights.sum(axis=1) > 0]
    assert len(active_dates) > 0, "Expected non-zero weights after warm-up period"
    for date in active_dates:
        assert abs(weights.loc[date].sum() - 1.0) < 1e-5
