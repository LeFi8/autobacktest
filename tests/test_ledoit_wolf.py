"""Unit tests for the Ledoit-Wolf shrinkage correlation matrix estimator."""

import numpy as np
import pandas as pd

from autobacktest.evaluator.deflated_sharpe import (
    _ledoit_wolf_correlation,
    calculate_effective_trials,
)


def test_ledoit_wolf_basic_properties() -> None:
    """Verifies that the shrunk correlation matrix satisfies basic mathematical properties."""
    np.random.seed(42)
    # Generate 100 days of returns for 5 strategy trials
    returns = pd.DataFrame(np.random.normal(0, 0.01, (100, 5)))

    corr = _ledoit_wolf_correlation(returns)

    # Check index and columns
    assert corr.index.equals(returns.columns)
    assert corr.columns.equals(returns.columns)

    # Check that diagonal elements are exactly 1.0
    np.testing.assert_allclose(np.diag(corr.values), 1.0, atol=1e-7)

    # Check symmetry
    np.testing.assert_allclose(corr.values, corr.values.T, atol=1e-7)

    # Check bounding in [-1.0, 1.0]
    assert np.all(corr.values >= -1.0)
    assert np.all(corr.values <= 1.0)


def test_ledoit_wolf_uncorrelated() -> None:
    """Verifies that independent return series are shrunk towards the identity correlation matrix."""
    np.random.seed(42)
    # Generate independent daily returns for 5 strategies
    returns = pd.DataFrame(np.random.normal(0, 0.01, (1000, 5)))

    corr = _ledoit_wolf_correlation(returns)
    # For truly uncorrelated returns, Ledoit-Wolf should shrink heavily towards the identity target (0 correlation)
    # The off-diagonal entries in the shrunk correlation matrix should be very close to 0.0
    off_diags = corr.values[~np.eye(5, dtype=bool)]
    np.testing.assert_allclose(off_diags, 0.0, atol=0.01)


def test_ledoit_wolf_high_correlation() -> None:
    """Verifies that highly correlated returns are successfully shrunk and clustered."""
    np.random.seed(42)
    a = np.random.normal(0, 0.01, 100)
    # b is almost perfectly correlated with a
    b = a + np.random.normal(0, 0.0001, 100)
    c = np.random.normal(0, 0.01, 100)

    df = pd.DataFrame({"A": a, "B": b, "C": c})

    corr = _ledoit_wolf_correlation(df)

    # The correlation between A and B should be shrunk slightly but still very high
    assert corr.loc["A", "B"] > 0.85
    assert corr.loc["A", "C"] < 0.2

    # Check that clustering still works as expected
    n_eff = calculate_effective_trials(df, threshold=0.5)
    assert n_eff == 2


def test_ledoit_wolf_fallbacks_and_edge_cases() -> None:
    """Verifies that the function handles edge cases and fails gracefully."""
    # 1. Empty DataFrame
    assert _ledoit_wolf_correlation(pd.DataFrame()).empty

    # 2. Single observation T = 1
    df_single = pd.DataFrame([[0.1, 0.2, 0.3]], columns=["A", "B", "C"])
    corr_single = _ledoit_wolf_correlation(df_single)
    # Should fallback to empirical correlation (which will be filled with zeros)
    assert np.all(corr_single.values == 0.0)

    # 3. Single strategy trial p = 1
    df_one_col = pd.DataFrame({"A": [0.1, 0.2, -0.1]})
    corr_one_col = _ledoit_wolf_correlation(df_one_col)
    assert corr_one_col.loc["A", "A"] == 1.0

    # 4. Zero variance / constant returns
    df_zeros = pd.DataFrame(np.zeros((100, 3)), columns=["A", "B", "C"])
    corr_zeros = _ledoit_wolf_correlation(df_zeros)
    # Should handle zero variance gracefully without crashing and return exactly 1.0 on the diagonal
    np.testing.assert_allclose(np.diag(corr_zeros.values), 1.0, atol=1e-7)
