import numpy as np
import pandas as pd

from autobacktest.evaluator.regime import calculate_regime_haircut


def test_calculate_regime_haircut():
    # Setup benchmark price series (300 days of SPY)
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=300, freq="D")

    # Prices are generated with random noise + upward trend
    noise = np.random.normal(0.001, 0.01, 300)
    log_prices = np.cumsum(noise)
    prices = 100.0 * np.exp(log_prices)
    benchmark_prices = pd.Series(prices, index=dates)

    # Launch date is the last day
    launch_date = dates[-1]

    haircut = calculate_regime_haircut(benchmark_prices, launch_date)

    # Haircut should be calculated and non-negative
    assert haircut >= 0.0

    # Let's force a positive z-score by setting the last price to be extremely high
    extreme_prices = benchmark_prices.copy()
    extreme_prices.iloc[-1] = extreme_prices.iloc[-253] * 10.0  # huge jump

    haircut_extreme = calculate_regime_haircut(extreme_prices, launch_date)
    assert haircut_extreme > 0.0

    # Check that flat prices return 0.0
    flat_prices = pd.Series([100.0] * 300, index=dates)
    assert calculate_regime_haircut(flat_prices, launch_date) == 0.0
