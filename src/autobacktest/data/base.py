"""Abstract base classes for market data providers.

Defines the ``DataProvider`` contract that all price-source backends must
implement.  Concrete providers (YFinance, Bloomberg, etc.) are expected to
subclass this and implement ``get_prices``.
"""

from abc import ABC, abstractmethod

import pandas as pd


class DataProvider(ABC):
    """Abstract base class for all data providers.

    Subclasses must implement ``get_prices``.  The returned DataFrame must
    have a ``DatetimeIndex`` and one column per ticker.  This contract is
    consumed by ``CachedDataProvider`` (caching decorator) and ultimately
    by the evaluation engine.
    """

    @abstractmethod
    def get_prices(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Fetch close prices for a list of tickers over a date range.

        Args:
            tickers: List of ticker symbols.
            start: Start date string (YYYY-MM-DD).
            end: End date string (YYYY-MM-DD).
            interval: Data interval (e.g. "1d").

        Returns:
            pd.DataFrame: DataFrame with DatetimeIndex and columns as tickers.
        """
        pass
