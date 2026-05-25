"""Yahoo Finance market data provider."""

import pandas as pd
import yfinance as yf

from autobacktest.data.base import DataProvider


class YFinanceProvider(DataProvider):
    """Data provider using yfinance to fetch historical daily price series."""

    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch Close prices from Yahoo Finance.

        Args:
            tickers: List of tickers.
            start: Start date YYYY-MM-DD.
            end: End date YYYY-MM-DD.
            interval: Bar frequency.

        Returns:
            pd.DataFrame: Dated DataFrame with Close prices of requested tickers.
        """
        if not tickers:
            return pd.DataFrame()

        unique_tickers = list(dict.fromkeys(tickers))

        # download returns a DataFrame
        df = yf.download(
            unique_tickers,
            start=start,
            end=end,
            interval=interval,
            progress=False,
        )

        if df.empty:
            return pd.DataFrame()

        # Handle MultiIndex vs single ticker Flat Index
        if isinstance(df.columns, pd.MultiIndex):
            # Extract Adjusted Close if present, otherwise standard Close
            metric = "Adj Close" if "Adj Close" in df.columns.levels[0] else "Close"
            prices = df[metric]
        else:
            # Single ticker case
            ticker = unique_tickers[0]
            metric = "Adj Close" if "Adj Close" in df.columns else "Close"
            if metric in df.columns:
                prices = df[[metric]].rename(columns={metric: ticker})
            else:
                prices = pd.DataFrame(index=df.index)

        # Force DatetimeIndex to timezone naive standard representation
        if prices.index.tz is not None:
            prices.index = prices.index.tz_localize(None)

        # Retain original ticker duplicates and ordering if requested
        return prices.reindex(columns=tickers)
