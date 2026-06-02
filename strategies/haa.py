from typing import Any

import numpy as np
import pandas as pd


def _rebalance_dates(prices: pd.DataFrame, cadence: str) -> pd.DatetimeIndex:
    """Get last trading day of each period. cadence: 'M' (monthly) or 'Q' (quarterly)."""
    if cadence not in {"M", "Q"}:
        cadence = "M"
    return prices.groupby(prices.index.to_period(cadence)).tail(1).index


def _momentum(prices: pd.DataFrame, period: int) -> pd.DataFrame:
    """Compute simple returns over specified period."""
    return prices.pct_change(period)


def _volatility(prices: pd.Series, period: int) -> pd.Series:
    """Annualized volatility of daily returns."""
    return prices.pct_change().rolling(period).std() * np.sqrt(252)


def _trend_filter(prices: pd.Series, period: int) -> pd.Series:
    """Trend: asset price above SMA."""
    sma = prices.rolling(period).mean()
    return prices > sma


def _canary_triggered(
    mom: pd.DataFrame,
    vol: pd.Series,
    trend: pd.Series,
    canary_assets: list,
    vol_threshold: float,
    trend_asset: str,
    date: pd.Timestamp,
) -> bool:
    """Check if any canary condition is violated."""
    # Momentum canary: any canary <= 0
    for c in canary_assets:
        if c in mom.columns:
            m = mom.loc[date, c]
            if np.isnan(m) or m <= 0:
                return True
    # Volatility canary
    if trend_asset in mom.columns and not vol.empty and date in vol.index:
        v = vol.loc[date]
        if not np.isnan(v) and v > vol_threshold:
            return True
    # Trend canary
    if trend_asset in mom.columns and not trend.empty and date in trend.index:
        t = trend.loc[date]
        if not t:  # asset below SMA
            return True
    return False


def _best_defensive(mom: pd.DataFrame, defensive_assets: list, date: pd.Timestamp) -> str:
    """Return defensive asset with highest momentum."""
    best = None
    best_mom = -np.inf
    for d in defensive_assets:
        if d in mom.columns:
            m = mom.loc[date, d]
            if not np.isnan(m) and m > best_mom:
                best_mom = m
                best = d
    return best if best is not None else defensive_assets[0]


def _top_offensive(mom: pd.DataFrame, offensive_assets: list, top_n: int, date: pd.Timestamp) -> list:
    """Return top N offensive assets by momentum, with positive momentum."""
    scores = []
    for a in offensive_assets:
        if a in mom.columns:
            m = mom.loc[date, a]
            if not np.isnan(m):
                scores.append((a, m))
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = scores[:top_n]
    return [a for a, m in selected if m > 0]


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate HAA weights with dual canary, trend filter, and top-N selection."""
    params = config.get("params", {})
    mom_lookback = params.get("mom_lookback", 252)
    canary_assets = params.get("canary_assets", ["TIP", "BND"])
    vol_period = params.get("vol_period", 21)
    vol_threshold = params.get("vol_threshold", 0.25)
    trend_asset = params.get("trend_asset", "SPY")
    trend_period = params.get("trend_period", 200)
    defensive_assets = params.get("defensive_assets", ["BIL", "BND"])
    offensive_assets = params.get("offensive_assets", ["SPY", "QQQ", "VGK", "VWO", "GLD", "TLT"])
    top_n = params.get("top_n", 4)
    rebalance_cadence = params.get("rebalance_cadence", "M")

    all_assets = list(prices.columns)
    if prices.empty:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex([]), columns=all_assets)

    available_offensive = [a for a in offensive_assets if a in all_assets]
    available_defensive = [a for a in defensive_assets if a in all_assets]
    available_canary = [a for a in canary_assets if a in all_assets]
    if not available_offensive:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex([]), columns=all_assets)

    mom_columns = list(dict.fromkeys(available_offensive + available_defensive + available_canary))
    if trend_asset not in mom_columns:
        mom_columns.append(trend_asset)
    mom = _momentum(prices.loc[:, mom_columns], mom_lookback)
    vol = _volatility(prices[trend_asset], vol_period) if trend_asset in all_assets else pd.Series()
    trend = _trend_filter(prices[trend_asset], trend_period) if trend_asset in all_assets else pd.Series()

    rebalance_dates = _rebalance_dates(prices, rebalance_cadence)
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=all_assets)
    slot_weight = 1.0 / top_n

    for date in rebalance_dates:
        if date not in mom.index:
            continue
        if _canary_triggered(mom, vol, trend, available_canary, vol_threshold, trend_asset, date):
            if available_defensive:
                best_def = _best_defensive(mom, available_defensive, date)
                weights.loc[date, best_def] = 1.0
            elif available_offensive:
                weights.loc[date, available_offensive[0]] = 1.0
        else:
            selected = _top_offensive(mom, available_offensive, top_n, date)
            for asset in selected:
                weights.loc[date, asset] = slot_weight
            remaining_slots = top_n - len(selected)
            if remaining_slots > 0:
                if available_defensive:
                    best_def = _best_defensive(mom, available_defensive, date)
                    weights.loc[date, best_def] = remaining_slots * slot_weight
                elif selected:
                    weights.loc[date, selected[0]] += remaining_slots * slot_weight

    return weights
