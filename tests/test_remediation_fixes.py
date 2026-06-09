"""Unit tests verifying the PR review fixes for regime evaluation."""

import pandas as pd

from autobacktest.evaluator.regime import evaluate_stress_regimes


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
