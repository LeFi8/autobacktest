"""Unit tests for the Deflated Sharpe Ratio and returns clustering."""

import numpy as np
import pandas as pd

from autobacktest.evaluator.deflated_sharpe import (
    calculate_effective_trials,
    calculate_psr_dsr,
)


def test_effective_trials_clustering() -> None:
    """Verifies that highly correlated return series are clustered together."""
    np.random.seed(42)

    # 100 days of daily returns
    a = np.random.normal(0, 0.01, 100)
    # b is a plus very tiny noise (highly correlated)
    b = a + np.random.normal(0, 0.0001, 100)
    # c is completely independent
    c = np.random.normal(0, 0.01, 100)

    df = pd.DataFrame({"A": a, "B": b, "C": c})

    # With a distance threshold of 0.5:
    # A and B are correlation ~1.0, distance = sqrt(0.5 * 0) = 0.0. They merge.
    # C is correlation ~0, distance = sqrt(0.5 * 1) = 0.707. C stays separate.
    # Total independent clusters should be exactly 2.
    n_eff = calculate_effective_trials(df, threshold=0.5)
    assert n_eff == 2


def test_psr_dsr_deflation() -> None:
    """Checks that DSR behaves as expected on trials counts and returns."""
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    # Positive drift returns
    returns = pd.Series(np.random.normal(0.001, 0.01, 200), index=dates)

    # Calculate standard PSR (N = 1)
    psr = calculate_psr_dsr(returns, effective_trials=1)
    # Since returns have highly positive drift, PSR should be high
    assert psr > 0.8

    # Calculate DSR with large number of effective trials N = 10
    # The significance of the observed Sharpe should deflate
    dsr = calculate_psr_dsr(
        returns,
        historical_sharpes=[0.2, 0.5, 0.8, 1.0, 1.2, 1.5, 1.8],
        effective_trials=10,
    )
    assert dsr < psr
