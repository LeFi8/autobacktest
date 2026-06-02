"""Unit tests for the Parquet market data cache."""

import tempfile
from pathlib import Path

import pandas as pd

from autobacktest.data.base import DataProvider
from autobacktest.data.cache import CachedDataProvider


class DummyDataProvider(DataProvider):
    """Stub DataProvider returning deterministic growth prices."""

    def __init__(self) -> None:
        self.fetch_count = 0

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        # Prevent unused parameter warning
        _ = interval
        self.fetch_count += 1
        dates = pd.date_range(start, end, freq="D")
        data = {}
        for t in tickers:
            # Deterministic price starting at 10.0 and growing by 1% daily
            data[t] = [10.0 * (1.01**i) for i in range(len(dates))]
        return pd.DataFrame(data, index=dates)


def test_cache_hit_and_incremental_updates() -> None:
    """Verifies CachedDataProvider caching and incremental fetch mechanisms."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_provider = DummyDataProvider()
        cached_provider = CachedDataProvider(raw_provider, cache_dir=tmp_dir)

        # 1. First fetch - full cache miss
        df1 = cached_provider.get_prices(["SPY"], "2023-01-01", "2023-01-10")
        assert raw_provider.fetch_count == 1
        assert len(df1) == 10
        assert "SPY" in df1.columns

        # 2. Same fetch - complete cache hit (fetch count remains 1)
        df2 = cached_provider.get_prices(["SPY"], "2023-01-01", "2023-01-10")
        assert raw_provider.fetch_count == 1
        assert len(df2) == 10
        pd.testing.assert_frame_equal(df1, df2, check_freq=False)

        # 3. Suffix update - extended end date
        df3 = cached_provider.get_prices(["SPY"], "2023-01-01", "2023-01-15")
        assert raw_provider.fetch_count == 2
        assert len(df3) == 15

        # 4. Prefix update - prepended start date
        df4 = cached_provider.get_prices(["SPY"], "2022-12-25", "2023-01-15")
        assert raw_provider.fetch_count == 3
        assert len(df4) == 22


def test_corrupt_parquet_cache_handling() -> None:
    """Verifies CachedDataProvider re-fetches safely on corrupted cache."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_provider = DummyDataProvider()
        cached_provider = CachedDataProvider(raw_provider, cache_dir=tmp_dir)

        # Create a corrupt parquet file
        cache_file = Path(tmp_dir) / "SPY_1d.parquet"
        cache_file.touch()
        with cache_file.open("w") as f:
            f.write("completely corrupted data")

        # Fetch prices should handle error gracefully and fetch freshly
        df = cached_provider.get_prices(["SPY"], "2023-01-01", "2023-01-05")
        assert raw_provider.fetch_count == 1
        assert len(df) == 5
        assert cache_file.exists()


def test_confirmed_empty_ttl_expiry() -> None:
    """Verifies TTL expiry forces re-fetch on confirmed_empty entries."""
    import json

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_provider = DummyDataProvider()
        cached_provider = CachedDataProvider(raw_provider, cache_dir=tmp_dir)
        meta_file = Path(tmp_dir) / "SPY_1d.json"

        # Write confirmed_empty metadata with future TTL (no parquet file)
        future = (pd.Timestamp.now() + pd.Timedelta(days=30)).isoformat()
        with meta_file.open("w") as f:
            json.dump({"start": "2023-01-01", "end": "2023-01-10", "confirmed_empty": True, "expires_at": future}, f)

        # No re-fetch (TTL still valid) — returns empty since cache has no data
        df = cached_provider.get_prices(["SPY"], "2023-01-01", "2023-01-05")
        assert raw_provider.fetch_count == 0
        assert df.empty

        # Now write confirmed_empty metadata with expired TTL
        expired = (pd.Timestamp.now() - pd.Timedelta(days=1)).isoformat()
        with meta_file.open("w") as f:
            json.dump({"start": "2023-01-01", "end": "2023-01-10", "confirmed_empty": True, "expires_at": expired}, f)

        # Request should re-fetch (TTL expired)
        df2 = cached_provider.get_prices(["SPY"], "2023-01-01", "2023-01-05")
        assert raw_provider.fetch_count == 1
        assert len(df2) == 5


def test_confirmed_empty_no_expires_at_treated_expired() -> None:
    """Verifies that old confirmed_empty entries (no expires_at) are treated as expired."""
    import json

    with tempfile.TemporaryDirectory() as tmp_dir:
        raw_provider = DummyDataProvider()
        cached_provider = CachedDataProvider(raw_provider, cache_dir=tmp_dir)
        meta_file = Path(tmp_dir) / "SPY_1d.json"

        # Write confirmed_empty metadata WITHOUT expires_at (legacy format)
        with meta_file.open("w") as f:
            json.dump({"start": "2023-01-01", "end": "2023-01-10", "confirmed_empty": True}, f)

        # Request should re-fetch (no expires_at means treated as expired)
        df = cached_provider.get_prices(["SPY"], "2023-01-01", "2023-01-05")
        assert raw_provider.fetch_count == 1
        assert len(df) == 5
