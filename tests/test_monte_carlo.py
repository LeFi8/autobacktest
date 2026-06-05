"""Unit tests for the Monte Carlo block bootstrap simulation."""

import numpy as np
import pandas as pd

from autobacktest.evaluator.monte_carlo import run_block_bootstrap


def test_block_bootstrap_iid_normal() -> None:
    """Verifies that block bootstrap of i.i.d returns yields coherent bands."""
    np.random.seed(42)

    # Generate 1000 days of i.i.d normal returns with positive drift
    # Expected daily Sharpe = 0.0005 / 0.01 = 0.05
    # Expected annualized Sharpe = 0.05 * sqrt(252) = 0.7937
    ret_arr = np.random.normal(loc=0.0005, scale=0.01, size=1000)
    returns = pd.Series(ret_arr)

    p5, p50, p95, path_sharpes = run_block_bootstrap(returns, n_paths=200, block_size=10, seed=42)

    # The 50th percentile Sharpe should be close to the true annualized Sharpe (~0.79)
    assert 0.4 < p50 < 1.2

    # Verify standard sorting behavior: p5 < p50 < p95
    assert p5 < p50 < p95

    # Verify full array is returned with expected length
    assert len(path_sharpes) == 200
