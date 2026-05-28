"""Parquet-backed market data cache."""

import json
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

    def _load_metadata(self, ticker: str, interval: str) -> tuple[pd.Timestamp | None, pd.Timestamp | None]:
        meta_file = self.cache_dir / f"{ticker}_{interval}.json"
        if meta_file.exists():
            try:
                with meta_file.open() as f:
                    data = json.load(f)
                return pd.to_datetime(data["start"]), pd.to_datetime(data["end"])
            except Exception:
                pass
        return None, None

    def _save_metadata(
        self,
        ticker: str,
        interval: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        confirmed_empty: bool = False,
    ) -> None:
        """Persist coverage boundaries to JSON metadata.

        Args:
            confirmed_empty: If True, the provider was successfully contacted and
                returned no data (e.g. missing ticker, market holiday).  The cache
                may skip future fetches for this range.  When False (default) the
                caller asserts that real rows were stored.
        """
        meta_file = self.cache_dir / f"{ticker}_{interval}.json"
        try:
            with meta_file.open("w") as f:
                json.dump(
                    {
                        # isoformat() preserves time-of-day for sub-daily intervals
                        "start": start.isoformat(),
                        "end": end.isoformat(),
                        "confirmed_empty": confirmed_empty,
                    },
                    f,
                )
        except Exception as e:
            logger.warning("Failed to save cache metadata for %s: %s", ticker, e)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _safe_fetch(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str,
        context: str,
    ) -> tuple[pd.DataFrame, bool]:
        """Fetch from provider, swallowing exceptions.

        Returns:
            (data, fetch_succeeded) — fetch_succeeded is False only when the
            provider raised an exception.  An empty DataFrame with
            fetch_succeeded=True means the provider responded cleanly but has
            no data for the range (holiday / missing ticker).
        """
        try:
            data = self.provider.get_prices([ticker], start, end, interval)
            return data, True
        except Exception as e:
            logger.warning(
                "Failed %s for ticker %s from %s to %s: %s",
                context,
                ticker,
                start,
                end,
                e,
            )
            return pd.DataFrame(), False

    @staticmethod
    def _slice_window(df: pd.DataFrame, start_dt: pd.Timestamp, end_dt: pd.Timestamp) -> pd.DataFrame:
        """Return rows in [start_dt, end_dt]; empty DataFrame if df is empty."""
        if df.empty:
            return pd.DataFrame()
        return df.loc[start_dt:end_dt]

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
            parquet_ok = True

            if cache_file.exists():
                try:
                    cached_df = pd.read_parquet(cache_file)
                except Exception as e:
                    logger.warning(
                        "Failed to read Parquet cache file %s: %s. Re-fetching data...",
                        cache_file,
                        e,
                    )
                    parquet_ok = False

            # P0-2: If parquet read failed, ignore stale metadata — forces re-fetch.
            meta_start, meta_end = None, None
            if parquet_ok:
                meta_start, meta_end = self._load_metadata(ticker, interval)
            if (meta_start is None or meta_end is None) and not cached_df.empty:
                meta_start = cached_df.index.min()
                meta_end = cached_df.index.max()

            needs_fetch = True
            ticker_df = pd.DataFrame()

            if meta_start is not None and meta_end is not None:
                cache_start = meta_start
                cache_end = meta_end

                # Cache covers the requested range — no fetch needed.
                if cache_start <= start_dt and cache_end >= end_dt:
                    needs_fetch = False
                    ticker_df = self._slice_window(cached_df, start_dt, end_dt)

                elif cache_start <= start_dt and cache_end < end_dt:
                    # Incremental suffix: fetch [cache_end+1d, end].
                    next_day = cache_end + pd.Timedelta(days=1)
                    fetch_start = next_day.strftime("%Y-%m-%d")
                    new_data, fetch_ok = self._safe_fetch(
                        ticker, fetch_start, end, interval, "incremental suffix update"
                    )

                    if not fetch_ok:
                        # Provider errored — don't advance metadata.
                        ticker_df = self._slice_window(cached_df, start_dt, end_dt)
                        needs_fetch = False
                    else:
                        if not new_data.empty:
                            cached_df = pd.concat([cached_df, new_data])
                            cached_df = cached_df[~cached_df.index.duplicated(keep="last")]
                            cached_df.sort_index(inplace=True)
                            cached_df.to_parquet(cache_file)
                            # P0-1: advance boundary only when rows were received.
                            self._save_metadata(ticker, interval, cache_start, end_dt)
                        else:
                            # Provider responded cleanly but has no rows (holiday/gap).
                            # Record the boundary so we don't re-query this range.
                            self._save_metadata(
                                ticker,
                                interval,
                                cache_start,
                                end_dt,
                                confirmed_empty=True,
                            )
                        ticker_df = self._slice_window(cached_df, start_dt, end_dt)
                        needs_fetch = False

                elif cache_start > start_dt and cache_end >= end_dt:
                    # Incremental prefix: fetch [start, cache_start-1d].
                    prev_day = cache_start - pd.Timedelta(days=1)
                    fetch_end = prev_day.strftime("%Y-%m-%d")
                    new_data, fetch_ok = self._safe_fetch(
                        ticker, start, fetch_end, interval, "incremental prefix update"
                    )

                    if not fetch_ok:
                        # Provider errored — don't advance metadata.
                        ticker_df = self._slice_window(cached_df, start_dt, end_dt)
                        needs_fetch = False
                    else:
                        if not new_data.empty:
                            cached_df = pd.concat([new_data, cached_df])
                            cached_df = cached_df[~cached_df.index.duplicated(keep="last")]
                            cached_df.sort_index(inplace=True)
                            cached_df.to_parquet(cache_file)
                            # P0-1: advance boundary only when rows were received.
                            self._save_metadata(ticker, interval, start_dt, cache_end)
                        else:
                            self._save_metadata(
                                ticker,
                                interval,
                                start_dt,
                                cache_end,
                                confirmed_empty=True,
                            )
                        ticker_df = self._slice_window(cached_df, start_dt, end_dt)
                        needs_fetch = False

            if needs_fetch:
                # Full window fetch.
                new_data, fetch_ok = self._safe_fetch(ticker, start, end, interval, "full window fetch")

                if not fetch_ok:
                    # Provider errored — return whatever is in cache (may be empty).
                    # Do NOT write metadata so the next call retries.
                    ticker_df = self._slice_window(cached_df, start_dt, end_dt)
                elif not new_data.empty:
                    if not cached_df.empty:
                        cached_df = pd.concat([cached_df, new_data])
                        cached_df = cached_df[~cached_df.index.duplicated(keep="last")]
                    else:
                        cached_df = new_data
                    cached_df.sort_index(inplace=True)
                    cached_df.to_parquet(cache_file)
                    # P0-1: metadata advance gated on actual rows stored.
                    self._save_metadata(ticker, interval, start_dt, end_dt)
                    ticker_df = self._slice_window(cached_df, start_dt, end_dt)
                else:
                    # Provider returned cleanly but no rows (missing ticker).
                    # Record confirmed_empty so future calls don't re-query.
                    self._save_metadata(ticker, interval, start_dt, end_dt, confirmed_empty=True)
                    ticker_df = self._slice_window(cached_df, start_dt, end_dt)

            # Align prices
            if not ticker_df.empty:
                merged = ticker_df if merged.empty else merged.join(ticker_df[[ticker]], how="outer")

        # Return aligned columns matching requested order and universe
        return merged.reindex(columns=tickers)
