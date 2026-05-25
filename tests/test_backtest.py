"""Unit tests for the vectorized backtest engine and cost model."""

import numpy as np
import pandas as pd
from autobacktest.evaluator.backtest import run_vectorized_backtest
from autobacktest.evaluator.costs import calculate_turnover_and_costs


def test_vectorized_backtest_constant_returns() -> None:
    """Verifies that a constant weight portfolio matches analytical expectations."""
    # Generate 100 days of prices for asset A growing at 1% daily
    dates = pd.date_range("2023-01-01", periods=100, freq="B")
    prices = pd.DataFrame(
        {"A": [1.0 * (1.01**i) for i in range(100)]},
        index=dates,
    )

    # Constant weight of 1.0 in asset A
    weights = pd.DataFrame({"A": [1.0] * 100}, index=dates)

    portfolio_returns, _equity_curve, daily_weights = run_vectorized_backtest(
        prices, weights
    )

    # First day has 0 returns because of shift(1) lookahead bias guard
    assert portfolio_returns.iloc[0] == 0.0
    # Day 2 returns should be exactly 1.0%
    np.testing.assert_almost_equal(portfolio_returns.iloc[1], 0.01)
    # Realigned weights should be exactly 1.0
    assert daily_weights.iloc[0]["A"] == 1.0


def test_costs_turnover_and_friction() -> None:
    """Checks that turnover is computed correctly and costs reduce net returns."""
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    prices = pd.DataFrame({"A": [10.0] * 10}, index=dates)

    # We shift weights from 0.0 to 1.0 at day 5, representing 100% turnover
    weights = pd.DataFrame(
        {"A": [0.0] * 5 + [1.0] * 5},
        index=dates,
    )

    portfolio_returns, _, daily_weights = run_vectorized_backtest(prices, weights)

    # Calculate net returns with 100bps total fee
    net_returns, _net_equity, turnover = calculate_turnover_and_costs(
        portfolio_returns,
        daily_weights,
        prices,
        commission_bps=50.0,
        spread_bps=50.0,
    )

    # 1.0 turnover over 10 business days (10/252 years)
    # expected turnover rate should be 1.0 / (10/252) = 25.2
    assert turnover > 25.0
    # Cost should be deducted on day 5 rebalance date
    # Weight drifted is 0.0, target is 1.0 -> trade is 1.0 -> fee is 100bps = 0.01
    assert net_returns.iloc[5] == -0.01
