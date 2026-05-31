"""Vol-targeted risk parity strategy."""

from typing import Any

import numpy as np
import pandas as pd


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate volatility-targeted equal risk contribution weights.

    Args:
        prices: Daily prices DataFrame (DatetimeIndex).
        config: Strategy configuration with keys:
            - risky_assets: list of tickers for the risk portfolio
            - cash_asset: ticker of the cash/defensive asset
            - target_vol: annualized target volatility (e.g., 0.15)
            - vol_span: EWMA span for volatility estimation (e.g., 126)

    Returns:
        pd.DataFrame: Weights indexed by rebalance dates.
    """
    risky_assets = config.get("risky_assets", [])
    cash_asset = config.get("cash_asset", "BIL")
    target_vol = config.get("target_vol", 0.15)
    vol_span = config.get("vol_span", 126)

    # Ensure all assets exist in prices
    available = set(prices.columns)
    risky_assets = [a for a in risky_assets if a in available]
    if cash_asset not in available:
        raise ValueError(f"Cash asset {cash_asset} not in price data")

    # Compute daily returns (simple percentage)
    returns = prices.pct_change().dropna(how='all')

    # Rebalance schedule: last trading day of each month
    monthly_dates = prices.groupby(prices.index.to_period("M")).tail(1).index
    monthly_dates = monthly_dates[monthly_dates >= returns.index[0]]  # start after some data

    weights = pd.DataFrame(0.0, index=monthly_dates, columns=prices.columns)

    for date in monthly_dates:
        # Slice returns up to current date
        hist_ret = returns.loc[:date, risky_assets].copy()
        if hist_ret.empty:
            continue

        # Compute EWMA volatility for each risky asset (annualized by sqrt(252))
        daily_vol = hist_ret.ewm(span=vol_span, min_periods=10).std().iloc[-1]
        ann_vol = daily_vol * np.sqrt(252)

        # Replace NaN/inf with a large value to effectively set weight to zero
        ann_vol = ann_vol.replace(0, np.nan).fillna(1e6)

        # Inverse volatility weights (raw)
        inv_vol = 1.0 / ann_vol
        total_inv_vol = inv_vol.sum()

        if total_inv_vol > 1e-8:
            w_raw = inv_vol / total_inv_vol
        else:
            w_raw = pd.Series(0.0, index=risky_assets)

        # Estimate portfolio volatility under zero correlation assumption
        port_var = (w_raw**2 * ann_vol**2).sum()
        port_vol = np.sqrt(port_var) if port_var > 0 else 0.0

        # Scale to target volatility; cap at 1.0
        if port_vol > 1e-8:
            leverage = min(1.0, target_vol / port_vol)
        else:
            leverage = 0.0

        # Final risky weights
        w_risky = w_raw * leverage

        # Assign to output
        for asset in risky_assets:
            weights.loc[date, asset] = w_risky.get(asset, 0.0)
        weights.loc[date, cash_asset] = 1.0 - leverage

    return weights
