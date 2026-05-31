"""Vectorized portfolio backtest engine."""

import pandas as pd


def run_vectorized_backtest(
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    *,
    asset_returns: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.Series, pd.DataFrame]:
    """Execute a vectorized backtest with lookahead-bias protection.

    Args:
        prices: Daily close prices DataFrame (index=DatetimeIndex, columns=tickers).
        weights: Portfolio weights DataFrame (index=DatetimeIndex, columns=tickers).
        asset_returns: Pre-computed daily asset returns (prices.pct_change()).
            When provided, ``prices.pct_change()`` is skipped for efficiency.

    Returns:
        tuple containing:
            - Daily portfolio returns (pd.Series)
            - Portfolio cumulative equity curve (pd.Series)
            - Aligned weights daily DataFrame (pd.DataFrame)
    """
    if prices.empty or weights.empty:
        empty_idx = prices.index if not prices.empty else weights.index
        return (
            pd.Series(0.0, index=empty_idx),
            pd.Series(1.0, index=empty_idx),
            pd.DataFrame(),
        )

    # Reindex weights to align with all price dates and columns
    daily_weights = weights.reindex(index=prices.index, columns=prices.columns)

    # Rebalance weights forward-fill (holding positions until next rebalance)
    # Fill pre-first-rebalance NaNs with 0.0
    daily_weights = daily_weights.ffill().fillna(0.0)

    # Compute daily asset percentage changes (or use pre-computed)
    if asset_returns is not None:
        asset_returns_window = asset_returns.loc[prices.index]
    else:
        asset_returns_window = prices.pct_change().fillna(0.0)

    # Shift weights by 1 day to ensure lookahead-bias protection:
    # Position at close of t-1 is held for the return of day t.
    shifted_weights = daily_weights.shift(1).fillna(0.0)

    # Calculate daily portfolio return (sum of asset weights * asset returns)
    # Weights sum <= 1.0; unallocated weight is held in 0% interest cash.
    portfolio_returns = (shifted_weights * asset_returns_window).sum(axis=1)

    # Calculate cumulative returns (equity curve starting from 1.0)
    equity_curve = (1.0 + portfolio_returns).cumprod()

    return portfolio_returns, equity_curve, daily_weights
