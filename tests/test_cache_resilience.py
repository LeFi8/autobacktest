"""Unit tests verifying cache resilience on market holidays, missing tickers, and provider errors."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd

from autobacktest.data.base import DataProvider
from autobacktest.data.cache import CachedDataProvider


class MockResilientDataProvider(DataProvider):
    """Stub DataProvider to test holiday empty returns and provider error cases."""

    def __init__(self) -> None:
        self.fetch_count = 0
        self.raise_error = False
        self.return_empty = False

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        _ = interval
        self.fetch_count += 1

        if self.raise_error:
            raise RuntimeError("Raw provider query timed out or failed.")

        if self.return_empty:
            return pd.DataFrame()

        # Normal mock return
        dates = pd.date_range(start, end, freq="B")
        data = {t: [10.0] * len(dates) for t in tickers}
        return pd.DataFrame(data, index=dates)


def test_market_holiday_skips_redundant_fetches() -> None:
    """Verifies that queries falling on holidays successfully write metadata and skip future fetches."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_provider = MockResilientDataProvider()
        cached_provider = CachedDataProvider(raw_provider, cache_dir=tmp_dir)

        # 1. Fetch normal date range first
        cached_provider.get_prices(["SPY"], "2026-05-20", "2026-05-22")
        assert raw_provider.fetch_count == 1

        # 2. Simulate holiday fetch (return empty DataFrame representing closed market)
        raw_provider.return_empty = True
        cached_provider.get_prices(["SPY"], "2026-05-20", "2026-05-25")
        # Assert provider was called once for the incremental suffix (May 23 to 25)
        assert raw_provider.fetch_count == 2

        # Verify companion JSON metadata covers May 25
        meta_file = Path(tmp_dir) / "SPY_1d.json"
        assert meta_file.exists()
        _, meta_end = cached_provider._load_metadata("SPY", "1d")
        assert meta_end == pd.to_datetime("2026-05-25")

        # 3. Query the same range again (holiday is now covered by metadata bounds)
        raw_provider.return_empty = False
        df = cached_provider.get_prices(["SPY"], "2026-05-20", "2026-05-25")
        # Fetch count MUST remain 2 (complete cache hit on the holiday bounds!)
        assert raw_provider.fetch_count == 2
        assert len(df) == 3  # only has the original 3 business days (May 20, 21, 22)


def test_provider_errors_graceful_handling() -> None:
    """Verifies that provider exceptions are caught gracefully and return empty/cached data."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_provider = MockResilientDataProvider()
        cached_provider = CachedDataProvider(raw_provider, cache_dir=tmp_dir)

        # 1. Populate cache first
        cached_provider.get_prices(["SPY"], "2026-05-20", "2026-05-22")
        assert raw_provider.fetch_count == 1

        # 2. Raw provider starts raising exceptions (e.g. rate limits, disconnects)
        raw_provider.raise_error = True

        # Incremental fetch shouldn't crash, but log a warning and return whatever is in cache
        df = cached_provider.get_prices(["SPY"], "2026-05-20", "2026-05-25")
        assert raw_provider.fetch_count == 2
        # Returns cached slice for May 20-22
        assert len(df) == 3


def test_missing_ticker_caching() -> None:
    """Verifies that missing tickers are recorded in metadata to prevent future provider requests."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_provider = MockResilientDataProvider()
        cached_provider = CachedDataProvider(raw_provider, cache_dir=tmp_dir)

        # Missing ticker returns empty
        raw_provider.return_empty = True

        # First request - cache miss, queries provider
        df1 = cached_provider.get_prices(["INVALID"], "2026-05-20", "2026-05-25")
        assert raw_provider.fetch_count == 1
        assert df1.empty

        # Second request for the same missing ticker - hits cache bounds in metadata, skips provider!
        df2 = cached_provider.get_prices(["INVALID"], "2026-05-20", "2026-05-25")
        assert raw_provider.fetch_count == 1
        assert df2.empty
