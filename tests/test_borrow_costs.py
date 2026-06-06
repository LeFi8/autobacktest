import numpy as np
import pandas as pd

from autobacktest.evaluator.costs import calculate_turnover_and_costs


def test_borrow_costs_modeling():
    # Setup simple data: 10 days, constant returns of 0.0, constant weights of -0.5 on SPY (shorting)
    dates = pd.date_range("2023-01-01", periods=10, freq="B")
    daily_returns = pd.Series([0.0] * 10, index=dates)
    daily_weights = pd.DataFrame({"SPY": [-0.5] * 10}, index=dates)
    prices = pd.DataFrame({"SPY": [100.0] * 10}, index=dates)

    # Run with 0 transaction costs (commission=0, spread=0) and borrow_cost_bps=200.0
    net_returns, _, _turnover = calculate_turnover_and_costs(
        daily_returns, daily_weights, prices, commission_bps=0.0, spread_bps=0.0, borrow_cost_bps=200.0
    )

    # Expected daily borrow cost for weight -0.5:
    # |-0.5| * (200.0 / (10000.0 * 252.0)) = 0.5 * (200.0 / 2520000.0) = 100.0 / 2520000.0 = 1 / 25200 = 3.96825e-5
    expected_daily_cost = 0.5 * (200.0 / 2520000.0)

    # Since daily_returns is 0.0 and transaction costs are 0.0,
    # net_returns should be -expected_daily_cost on each day
    for val in net_returns:
        assert np.isclose(val, -expected_daily_cost)
