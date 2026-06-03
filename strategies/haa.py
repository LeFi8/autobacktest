from typing import Any

import numpy as np
import pandas as pd


def _rebalance_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Return business month end dates that exist in the price index."""
    start = prices.index.min()
    end = prices.index.max()
    possible = pd.date_range(start=start, end=end, freq="BME")
    return possible.intersection(prices.index)


def _min_momentum(prices: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Compute the minimum return across the given lag periods."""
    returns = [prices.pct_change(lag) for lag in lags]
    stacked = np.stack([r.values for r in returns], axis=2)
    # Use nanmin ignoring NaN slices; if all NaN, result is NaN
    min_mom = np.nanmin(stacked, axis=2)
    return pd.DataFrame(min_mom, index=returns[0].index, columns=returns[0].columns)


def _trailing_volatility(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized volatility of daily returns over a rolling window."""
    returns = prices.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)


def _sma(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Simple moving average of prices."""
    return prices.rolling(window).mean()


def _risk_adjusted_min_mom(min_mom: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Divide min momentum by annualized volatility."""
    epsilon = 1e-6
    return min_mom.div(vol.add(epsilon))


def _canary_triggered(min_mom: pd.DataFrame, canary_assets: list[str], date: pd.Timestamp) -> bool:
    """Return True if any canary asset's min momentum <= 0."""
    for asset in canary_assets:
        if asset in min_mom.columns and not pd.isna(min_mom.at[date, asset]) and min_mom.at[date, asset] <= 0:
            return True
    return False


def _eligible_offensive(
    prices: pd.DataFrame,
    date: pd.Timestamp,
    rmom: pd.DataFrame,
    sma: pd.DataFrame,
    offensive_assets: list[str],
    threshold: float,
) -> list[str]:
    """Return offensive assets with positive risk-adjusted momentum and above SMA."""
    eligible = []
    for a in offensive_assets:
        if a not in rmom.columns or a not in sma.columns or a not in prices.columns:
            continue
        if pd.isna(rmom.at[date, a]) or pd.isna(sma.at[date, a]):
            continue
        if rmom.at[date, a] > threshold and prices.at[date, a] > sma.at[date, a]:
            eligible.append(a)
    # Sort by rmom descending
    eligible.sort(key=lambda a: rmom.at[date, a], reverse=True)
    return eligible


def _best_defensive(min_mom: pd.DataFrame, defensive_assets: list[str], date: pd.Timestamp) -> str | None:
    """Return defensive asset with highest min momentum (safety preference)."""
    best = None
    best_mom = -np.inf
    for d in defensive_assets:
        if d in min_mom.columns and not pd.isna(min_mom.at[date, d]):
            m = min_mom.at[date, d]
            if m > best_mom:
                best_mom, best = m, d
    return best


def _portfolio_volatility(
    daily_returns: pd.DataFrame,
    selected_assets: list[str],
    date: pd.Timestamp,
    window: int,
    vol_df: pd.DataFrame,
) -> float:
    """Estimate annualized volatility of an equal-weighted portfolio of selected assets."""
    if not selected_assets:
        return np.inf
    slice_ret = daily_returns.loc[:date, selected_assets]
    valid_mask = ~slice_ret.isna().any(axis=1)
    port_ret = slice_ret[valid_mask].mean(axis=1)
    if len(port_ret) < window:
        return vol_df.loc[date, selected_assets].mean()
    rolling_std = port_ret.rolling(window).std()
    port_vol = rolling_std.iloc[-1] * np.sqrt(252)
    if pd.isna(port_vol) or port_vol == 0:
        port_vol = vol_df.loc[date, selected_assets].mean()
    return port_vol


def _target_weights(
    date: pd.Timestamp,
    prices: pd.DataFrame,
    daily_returns: pd.DataFrame,
    min_mom: pd.DataFrame,
    rmom: pd.DataFrame,
    sma: pd.DataFrame,
    vol_df: pd.DataFrame,
    config: dict[str, Any],
    prev_weights: pd.Series | None,
) -> pd.Series:
    """Compute target weights for one rebalance date."""
    params = config.get("params", {})
    defensive_assets = params.get("defensive_assets", ["BIL", "BND"])
    offensive_assets = params.get("offensive_assets", [])
    canary_assets = params.get("canary_assets", [])
    top_n = params.get("top_n", 6)
    target_vol = params.get("target_vol", 0.10)
    port_vol_window = params.get("port_vol_window", 21)
    smooth = params.get("smooth", 0.25)
    min_mom_threshold = params.get("min_mom_threshold", 0.0)

    all_cols = min_mom.columns
    target = pd.Series(0.0, index=all_cols)

    # Canary check
    if _canary_triggered(min_mom, canary_assets, date):
        best_def = _best_defensive(min_mom, defensive_assets, date)
        if best_def and best_def in all_cols:
            target[best_def] = 1.0
    else:
        eligible = _eligible_offensive(prices, date, rmom, sma, offensive_assets, min_mom_threshold)
        if not eligible:
            best_def = _best_defensive(min_mom, defensive_assets, date)
            if best_def and best_def in all_cols:
                target[best_def] = 1.0
        else:
            selected = eligible[:top_n]
            port_vol = _portfolio_volatility(daily_returns, selected, date, port_vol_window, vol_df)
            scale = min(1.0, target_vol / port_vol) if port_vol > 0 else 1.0
            slot_weight = scale / max(1, len(selected))
            for asset in selected:
                target[asset] = slot_weight
            defensive_exposure = 1.0 - target.sum()
            if defensive_exposure > 0:
                best_def = _best_defensive(min_mom, defensive_assets, date)
                if best_def and best_def in all_cols:
                    target[best_def] += defensive_exposure
                else:
                    # Redistribute among selected assets
                    if selected:
                        per_slot = defensive_exposure / len(selected)
                        for a in selected:
                            target[a] += per_slot

    if prev_weights is not None:
        target = smooth * target + (1.0 - smooth) * prev_weights

    target = target.clip(lower=0.0)
    total = target.sum()
    if total > 0:
        target = target / total
    return target


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate signals for a robust momentum strategy with binary canary, trend filter, and volatility targeting.

    Parameters (all under `params`):
        mom_lags: list[int]              momentum lookback periods for min momentum (default [21,63,126,252])
        vol_window: int                  volatility rolling window for asset vol (default 126)
        trend_window: int                SMA window for trend filter (default 200)
        port_vol_window: int             window for portfolio volatility (default 21)
        target_vol: float                annualized target volatility (default 0.10)
        top_n: int                       number of offensive slots (default 6)
        min_mom_threshold: float         minimum risk-adjusted min momentum to be eligible (default 0.0)
        smooth: float                    blending factor for weight smoothing (default 0.25)
        defensive_assets: list[str]      safe haven assets
        offensive_assets: list[str]      risky assets to select from
        canary_assets: list[str]         assets used for binary canary (e.g., ["TIP", "BND"])
    """
    params = config.get("params", {})
    mom_lags = params.get("mom_lags", [21, 63, 126, 252])
    vol_window = params.get("vol_window", 126)
    trend_window = params.get("trend_window", 200)
    # other params are used inside _target_weights via config pass

    all_assets = list(prices.columns)
    if prices.empty:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex([]), columns=all_assets)

    # Pre-compute all signals (strictly backward-looking)
    mm = _min_momentum(prices, mom_lags)
    vol = _trailing_volatility(prices, vol_window)
    sma = _sma(prices, trend_window)
    rmom = _risk_adjusted_min_mom(mm, vol)
    daily_returns = prices.pct_change()

    # Business month end dates (no lookahead)
    rebalance_dates = _rebalance_dates(prices)

    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=all_assets)
    prev_weights = None

    for date in rebalance_dates:
        if date not in rmom.index:
            continue
        new_w = _target_weights(
            date=date,
            prices=prices,
            daily_returns=daily_returns,
            min_mom=mm,
            rmom=rmom,
            sma=sma,
            vol_df=vol,
            config=config,
            prev_weights=prev_weights,
        )
        weights.loc[date] = new_w
        prev_weights = new_w.copy()

    # Forward-fill to the full price index
    weights = weights.reindex(prices.index).ffill().fillna(0.0)
    return weights
