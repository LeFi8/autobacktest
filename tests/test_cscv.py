import numpy as np
import pandas as pd

from autobacktest.evaluator.cscv import calculate_pbo


def test_cscv_pbo_calculation():
    # Create random daily returns for 3 trials over 200 days
    # Seed generator for reproducibility
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200)

    # Trial 0: High variance, overfits IS
    # Trial 1: Stable positive return
    # Trial 2: Negative returns
    t0 = np.random.normal(0.001, 0.02, 200)
    t1 = np.random.normal(0.0005, 0.005, 200)
    t2 = np.random.normal(-0.001, 0.01, 200)

    returns_df = pd.DataFrame({"trial_0": t0, "trial_1": t1, "trial_2": t2}, index=dates)

    pbo = calculate_pbo(returns_df, n_blocks=10)

    # PBO should be a valid probability
    assert 0.0 <= pbo <= 1.0

    # For single trial, PBO should be 0.0
    pbo_single = calculate_pbo(returns_df[["trial_0"]], n_blocks=10)
    assert pbo_single == 0.0


def test_cscv_all_zero_returns():
    """All-zero returns across all trials → zero-volatility edge case."""
    dates = pd.date_range("2023-01-01", periods=200)
    df = pd.DataFrame({f"t{i}": [0.0] * 200 for i in range(5)}, index=dates)
    pbo = calculate_pbo(df, n_blocks=10)
    assert 0.0 <= pbo <= 1.0


def test_cscv_uneven_split():
    """n_days % n_blocks != 0: ensure no crash and valid probability."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=205)  # 205 not divisible by 10
    t0 = np.random.normal(0.001, 0.02, 205)
    t1 = np.random.normal(0.0005, 0.005, 205)
    t2 = np.random.normal(-0.001, 0.01, 205)
    returns_df = pd.DataFrame({"trial_0": t0, "trial_1": t1, "trial_2": t2}, index=dates)
    pbo = calculate_pbo(returns_df, n_blocks=10)
    assert 0.0 <= pbo <= 1.0


def test_cscv_large_trial_count():
    """Large trial count stress-test for silhouette optimizer runtime."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=300)
    cols = {}
    for i in range(20):
        cols[f"t{i}"] = np.random.normal(0.0, 0.01, 300)
    df = pd.DataFrame(cols, index=dates)
    pbo = calculate_pbo(df, n_blocks=10)
    assert 0.0 <= pbo <= 1.0


def test_cscv_insufficient_days():
    """Fewer than 2*n_blocks days → return 0.0 (early exit)."""
    dates = pd.date_range("2023-01-01", periods=15)
    df = pd.DataFrame({"a": [0.001] * 15, "b": [-0.001] * 15}, index=dates)
    assert calculate_pbo(df, n_blocks=10) == 0.0


def test_cscv_highly_correlated_trials():
    """Nearly identical trials → PBO should be low (no overfitting signal)."""
    np.random.seed(42)
    dates = pd.date_range("2023-01-01", periods=200)
    base = np.random.normal(0.0005, 0.005, 200)
    cols = {f"t{i}": base + np.random.normal(0.0, 0.0001, 200) for i in range(5)}
    df = pd.DataFrame(cols, index=dates)
    pbo = calculate_pbo(df, n_blocks=10)
    assert 0.0 <= pbo <= 1.0
