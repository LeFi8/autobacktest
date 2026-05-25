"""Unit tests for holdout validation, regimes, and metric calculations."""

import pandas as pd

from autobacktest.evaluator.evaluate import (
    calculate_information_ratio,
    calculate_sortino_ratio,
)
from autobacktest.evaluator.holdout import partition_holdout_data
from autobacktest.evaluator.regime import evaluate_stress_regimes
from autobacktest.evaluator.walk_forward import generate_walk_forward_windows


def test_sortino_ratio_zero_downside() -> None:
    """Verifies that Sortino returns infinity on purely positive returns."""
    returns = pd.Series([0.01, 0.02, 0.015, 0.03])
    sortino = calculate_sortino_ratio(returns)
    assert sortino == float("inf")


def test_sortino_ratio_mathematical_precision() -> None:
    """Verifies Sortino aligns with downside deviation over full sample size."""
    returns = pd.Series([0.02, -0.01, 0.03, -0.02])
    # Mean return = (0.02 - 0.01 + 0.03 - 0.02) / 4 = 0.005
    # Negative returns = [0.0, -0.01, 0.0, -0.02]
    # Sum of squares = 0.0 + 0.0001 + 0.0 + 0.0004 = 0.0005
    # Downside std = sqrt(0.0005 / 4) = sqrt(0.000125) = 0.0111803
    # Sortino = (0.005 / 0.0111803) * sqrt(252) = ~7.1
    sortino = calculate_sortino_ratio(returns)
    assert 7.0 < sortino < 7.2


def test_information_ratio() -> None:
    """Verifies that Information Ratio computes correct active returns."""
    ret = pd.Series([0.01, 0.02, 0.01, 0.03])
    bench = pd.Series([0.005, 0.01, 0.005, 0.02])
    ir = calculate_information_ratio(ret, bench)
    assert ir > 40.0


def test_holdout_partitioning() -> None:
    """Checks that partitioning date indices splits correctly."""
    dates = pd.date_range("2015-01-01", "2025-01-01", freq="ME")
    in_sample, holdout = partition_holdout_data(dates, holdout_years=3)

    assert in_sample.max() < holdout.min()
    assert len(holdout) >= 36


def test_walk_forward_windows() -> None:
    """Verifies walk-forward date generation yields correct windows."""
    dates = pd.date_range("2010-01-01", "2020-01-01", freq="D")
    windows = generate_walk_forward_windows(dates, train_years=5, test_years=1)

    assert len(windows) > 0
    for train_start, train_end, test_start, test_end in windows:
        assert train_start < train_end
        assert train_end < test_start
        assert test_start < test_end


def test_stress_regimes_overlapping() -> None:
    """Verifies that stress regime drawdowns calculate correctly."""
    dates = pd.date_range("2008-01-01", "2023-01-01", freq="D")
    returns = pd.Series(0.0001, index=dates)

    # Introduce crash in 2020 (Covid regime)
    covid_dates = pd.date_range("2020-02-20", "2020-04-30", freq="D")
    returns.loc[covid_dates] = -0.05

    drawdowns, passed = evaluate_stress_regimes(returns)
    assert "2020_COVID" in drawdowns
    assert drawdowns["2020_COVID"] > 0.10
    # Covid max drawdown limit is 15%. Passed should be False
    assert passed is False
