from unittest.mock import MagicMock

import pandas as pd

from autobacktest.data.base import DataProvider
from autobacktest.data.cache import CachedDataProvider


def test_outlier_detection_and_cleaning(tmp_path):
    # Setup mock data provider
    base_provider = MagicMock(spec=DataProvider)
    cache_provider = CachedDataProvider(base_provider, cache_dir=str(tmp_path))

    # 6-day prices series for a single ticker 'SPY'
    # Upward spike on day 3 (indices: 0, 1, 2, 3, 4, 5)
    # Day 0: 100
    # Day 1: 101
    # Day 2: 100
    # Day 3: 160 (60% increase from 100)
    # Day 4: 95 (40.6% decrease from 160)
    # Day 5: 96
    dates = pd.date_range("2023-01-01", periods=6, freq="D")
    prices = pd.DataFrame({"SPY": [100.0, 101.0, 100.0, 160.0, 95.0, 96.0]}, index=dates)

    cleaned = cache_provider._detect_and_clean_outliers(prices)

    # Check that day 3 (160.0) is replaced with day 2 (100.0)
    assert cleaned.loc[dates[3], "SPY"] == 100.0
    # Other values should remain the same
    assert cleaned.loc[dates[0], "SPY"] == 100.0
    assert cleaned.loc[dates[1], "SPY"] == 101.0
    assert cleaned.loc[dates[2], "SPY"] == 100.0
    assert cleaned.loc[dates[4], "SPY"] == 95.0
    assert cleaned.loc[dates[5], "SPY"] == 96.0


def test_outlier_downward_spike(tmp_path):
    base_provider = MagicMock(spec=DataProvider)
    cache_provider = CachedDataProvider(base_provider, cache_dir=str(tmp_path))

    # Downward spike: 100 -> 40 (60% decrease) -> 90 (125% increase)
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    prices = pd.DataFrame({"QQQ": [100.0, 100.0, 40.0, 90.0, 91.0]}, index=dates)

    cleaned = cache_provider._detect_and_clean_outliers(prices)

    # Check that day 2 (40.0) is replaced with day 1 (100.0)
    assert cleaned.loc[dates[2], "QQQ"] == 100.0
    # Other values
    assert cleaned.loc[dates[0], "QQQ"] == 100.0
    assert cleaned.loc[dates[1], "QQQ"] == 100.0
    assert cleaned.loc[dates[3], "QQQ"] == 90.0
    assert cleaned.loc[dates[4], "QQQ"] == 91.0
