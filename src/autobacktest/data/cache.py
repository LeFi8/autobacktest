"""Parquet-backed market data cache."""

import logging
from pathlib import Path

import pandas as pd

from autobacktest.data.base import DataProvider

logger = logging.getLogger(__name__)


class CachedDataProvider(DataProvider):
    """Decorator for DataProviders to cache fetched price history in Parquet files."""

    def __init__(self, provider: DataProvider, cache_dir: str = "data/cache"):
        self.provider = provider
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Get prices from cache, fetching missing dates incrementally if necessary."""
        if not tickers:
            return pd.DataFrame()

        start_dt = pd.to_datetime(start)
        end_dt = pd.to_datetime(end)

        merged = pd.DataFrame()

        for ticker in tickers:
            cache_file = self.cache_dir / f"{ticker}_{interval}.parquet"
            cached_df = pd.DataFrame()

            if cache_file.exists():
                try:
                    cached_df = pd.read_parquet(cache_file)
                except Exception as e:
                    logger.warning(
                        "Failed to read Parquet cache file %s: %s. Re-fetching data...",
                        cache_file,
                        e,
                    )

            needs_fetch = True
            if not cached_df.empty:
                cache_start = cached_df.index.min()
                cache_end = cached_df.index.max()

                # If cache covers the requested range
                if cache_start <= start_dt and cache_end >= end_dt:
                    needs_fetch = False
                    ticker_df = cached_df.loc[start_dt:end_dt]
                elif cache_start <= start_dt and cache_end < end_dt:
                    # Incremental update: fetch from cache_end + 1 day to end
                    next_day = cache_end + pd.Timedelta(days=1)
                    fetch_start = next_day.strftime("%Y-%m-%d")
                    new_data = self.provider.get_prices(
                        [ticker], fetch_start, end, interval
                    )
                    if not new_data.empty:
                        cached_df = pd.concat([cached_df, new_data])
                        cached_df = cached_df[~cached_df.index.duplicated(keep="last")]
                        cached_df.sort_index(inplace=True)
                        cached_df.to_parquet(cache_file)
                    ticker_df = cached_df.loc[start_dt:end_dt]
                    needs_fetch = False
                elif cache_start > start_dt and cache_end >= end_dt:
                    # Prepending incremental update:
                    # fetch from start to cache_start - 1 day

                    prev_day = cache_start - pd.Timedelta(days=1)
                    fetch_end = prev_day.strftime("%Y-%m-%d")
                    new_data = self.provider.get_prices(
                        [ticker], start, fetch_end, interval
                    )
                    if not new_data.empty:
                        cached_df = pd.concat([new_data, cached_df])
                        cached_df = cached_df[~cached_df.index.duplicated(keep="last")]
                        cached_df.sort_index(inplace=True)
                        cached_df.to_parquet(cache_file)
                    ticker_df = cached_df.loc[start_dt:end_dt]
                    needs_fetch = False

            if needs_fetch:
                # Fetch full window
                new_data = self.provider.get_prices([ticker], start, end, interval)
                if not new_data.empty:
                    if not cached_df.empty:
                        cached_df = pd.concat([cached_df, new_data])
                        cached_df = cached_df[~cached_df.index.duplicated(keep="last")]
                    else:
                        cached_df = new_data
                    cached_df.sort_index(inplace=True)
                    cached_df.to_parquet(cache_file)
                    ticker_df = cached_df.loc[start_dt:end_dt]
                else:
                    ticker_df = pd.DataFrame()

            # Align prices
            if not ticker_df.empty:
                if merged.empty:
                    merged = ticker_df
                else:
                    # Outer join to align indices, handling possible duplicates
                    merged = merged.join(ticker_df[[ticker]], how="outer")

        # Return aligned columns matching requested order and universe
        return merged.reindex(columns=tickers)
