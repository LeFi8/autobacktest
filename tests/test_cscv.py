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
