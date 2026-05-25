"""Unit tests for robust edge cases and configuration fallbacks."""

import numpy as np
import pandas as pd

from autobacktest.evaluator.backtest import run_vectorized_backtest
from autobacktest.evaluator.deflated_sharpe import (
    calculate_effective_trials,
    calculate_psr_dsr,
)
from autobacktest.evaluator.evaluate import (
    calculate_information_ratio,
)


def test_information_ratio_insufficient_data() -> None:
    """Verifies that tracking error returns 0.0 with < 2 elements."""
    ret = pd.Series([0.01])
    bench = pd.Series([0.005])
    ir = calculate_information_ratio(ret, bench)
    assert ir == 0.0


def test_deflated_sharpe_single_historical() -> None:
    """Verifies that DSR does not crash when historical sharpes < 2."""
    dates = pd.date_range("2023-01-01", periods=200, freq="B")
    returns = pd.Series(np.random.normal(0.001, 0.01, 200), index=dates)

    # N = 2, but only 1 historical sharpe provided. ddof=1 is bypassed.
    dsr = calculate_psr_dsr(
        returns,
        historical_sharpes=[1.0],
        effective_trials=2,
    )
    assert dsr > 0.0


def test_deflated_sharpe_extreme_skew_variance_clipping() -> None:
    """Verifies variance clipping prevents NaN under negative num_var."""
    dates = pd.date_range("2023-01-01", periods=5, freq="B")
    # Highly skewed returns to force negative num_var if not clipped
    returns = pd.Series([10.0, -1.0, -1.0, -1.0, -1.0], index=dates)

    dsr = calculate_psr_dsr(returns)
    assert not np.isnan(dsr)
    assert dsr >= 0.0


def test_effective_trials_correlation_clipping() -> None:
    """Verifies exact correlation 1.0 does not produce NaN."""
    # Construct a mock returns dataframe with perfect correlation
    df = pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [1.0, 2.0, 3.0],
        }
    )
    n_eff = calculate_effective_trials(df)
    assert n_eff == 1


def test_backtest_column_mismatch_alignment() -> None:
    """Verifies weights with mismatched columns align properly."""
    prices = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "TLT": [50.0, 51.0, 52.0],
        },
        index=pd.date_range("2023-01-01", periods=3),
    )

    weights = pd.DataFrame(
        {
            "SPY": [1.0, 1.0, 1.0],
            # TLT is completely missing in weights columns
        },
        index=prices.index,
    )

    portfolio_returns, _eq, daily_weights = run_vectorized_backtest(prices, weights)

    assert "TLT" in daily_weights.columns
    assert daily_weights["TLT"].iloc[0] == 0.0
    assert not portfolio_returns.isna().any()
