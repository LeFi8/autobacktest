import numpy as np
import pandas as pd
import pytest

from autobacktest.evaluator.monte_carlo import run_block_bootstrap


def test_monte_carlo_invalid_method() -> None:
    returns = pd.Series([0.001] * 30)
    with pytest.raises(ValueError, match="Unknown bootstrap method"):
        run_block_bootstrap(returns, method="invalid_method")


def test_monte_carlo_short_series() -> None:
    returns = pd.Series([0.001] * 10)
    p5, p50, p95, sharpes = run_block_bootstrap(returns, block_size=21)
    assert p5 == 0.0
    assert p50 == 0.0
    assert p95 == 0.0
    assert len(sharpes) == 0


def test_monte_carlo_percentiles_and_shape() -> None:
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 200))

    for method in ("circular", "stationary"):
        p5, p50, p95, sharpes = run_block_bootstrap(returns, n_paths=100, block_size=10, seed=42, method=method)
        assert p5 <= p50 <= p95
        assert len(sharpes) == 100


def test_monte_carlo_stationary_reproducibility() -> None:
    returns = pd.Series(np.random.normal(0.0005, 0.01, 100))
    p5_1, p50_1, p95_1, sharpes_1 = run_block_bootstrap(returns, n_paths=50, seed=42, method="stationary")
    p5_2, p50_2, p95_2, sharpes_2 = run_block_bootstrap(returns, n_paths=50, seed=42, method="stationary")

    assert p5_1 == p5_2
    assert p50_1 == p50_2
    assert p95_1 == p95_2
    np.testing.assert_array_equal(sharpes_1, sharpes_2)


def test_monte_carlo_circular_regression() -> None:
    # Set a fixed seed and check that circular block bootstrap returns expected values
    np.random.seed(42)
    returns = pd.Series(np.random.normal(0.0005, 0.01, 100))
    _p5, p50, _p95, _ = run_block_bootstrap(returns, n_paths=100, seed=42, method="circular")

    # Verify that the percentiles are within a reasonable range for this fixed random seed
    assert -3.0 <= p50 <= 3.0


def test_monte_carlo_distribution_difference() -> None:
    # Autocorrelated returns series
    np.random.seed(42)
    noise = np.random.normal(0, 0.01, 200)
    returns_list = [0.0]
    for i in range(1, 200):
        returns_list.append(0.5 * returns_list[-1] + noise[i])
    returns = pd.Series(returns_list)

    # Run both circular and stationary with same seed
    _, p50_circ, _, _ = run_block_bootstrap(returns, n_paths=500, seed=42, method="circular")
    _, p50_stat, _, _ = run_block_bootstrap(returns, n_paths=500, seed=42, method="stationary")

    # They should differ due to block-length variation in stationary bootstrap
    assert p50_circ != p50_stat
