from typing import Any

import numpy as np
import pandas as pd


def _rebalance_dates(prices: pd.DataFrame) -> pd.DatetimeIndex:
    """Return business month end dates that exist in the price index.

    Using a deterministic calendar avoids lookahead that occurs when future data
    within a month shifts the last-observed-day for that month."""
    start = prices.index.min()
    end = prices.index.max()
    possible = pd.date_range(start=start, end=end, freq="BME")
    return possible.intersection(prices.index)


def _composite_momentum(prices: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Equal-weight average of simple returns over the given lag periods."""
    moments = [prices.pct_change(lag) for lag in lags]
    stacked = np.stack([m.values for m in moments], axis=2)
    valid = ~np.isnan(stacked)
    count = valid.sum(axis=2)
    avg = np.where(count > 0, np.where(valid, stacked, 0.0).sum(axis=2) / np.maximum(count, 1.0), 0.0)
    return pd.DataFrame(avg, index=moments[0].index, columns=moments[0].columns)


def _trailing_volatility(prices: pd.DataFrame, window: int) -> pd.DataFrame:
    """Annualized volatility of daily returns over a rolling window."""
    returns = prices.pct_change()
    return returns.rolling(window).std() * np.sqrt(252)


def _risk_adjusted_momentum(mom: pd.DataFrame, vol: pd.DataFrame) -> pd.DataFrame:
    """Divide composite momentum by annualized volatility (plus tiny epsilon)."""
    epsilon = 1e-6
    return mom.div(vol.add(epsilon))


def _canary_scale(rmom: pd.DataFrame, canary_assets: list[str], z_window: int) -> pd.Series:
    """Continuous exposure scale in [0,1] using rolling z-score of canary assets' mean rmom."""
    if not canary_assets or not all(a in rmom.columns for a in canary_assets):
        return pd.Series(1.0, index=rmom.index)
    mean_rmom = rmom[canary_assets].mean(axis=1)
    rolling_mean = mean_rmom.rolling(z_window, min_periods=1).mean()
    rolling_std = mean_rmom.rolling(z_window, min_periods=1).std()
    rolling_std = rolling_std.replace(0.0, 1e-6).fillna(1e-6)
    z = (mean_rmom - rolling_mean) / rolling_std
    scale = (np.tanh(z) + 1.0) / 2.0
    return scale


def _select_offensive(rmom: pd.DataFrame, offensive_assets: list[str], top_n: int, date: pd.Timestamp) -> list[str]:
    """Return top_n offensive assets with positive risk-adjusted momentum on `date`."""
    available = [a for a in offensive_assets if a in rmom.columns and not pd.isna(rmom.at[date, a])]
    scores = [(a, rmom.at[date, a]) for a in available if rmom.at[date, a] > 0]
    scores.sort(key=lambda x: x[1], reverse=True)
    return [a for a, _ in scores[:top_n]]


def _best_defensive(rmom: pd.DataFrame, defensive_assets: list[str], date: pd.Timestamp) -> str | None:
    """Return defensive asset with highest risk-adjusted momentum; None if none available."""
    best, best_mom = None, -np.inf
    for d in defensive_assets:
        if d in rmom.columns and not pd.isna(rmom.at[date, d]):
            m = rmom.at[date, d]
            if m > best_mom:
                best_mom, best = m, d
    return best


def _target_weights(
    date: pd.Timestamp,
    rmom: pd.DataFrame,
    canary: pd.Series,
    offensive_assets: list[str],
    defensive_assets: list[str],
    top_n: int,
) -> pd.Series:
    """Compute target weights for one rebalance date."""
    all_cols = rmom.columns
    target = pd.Series(0.0, index=all_cols)
    slot_weight = 1.0 / top_n

    scale = canary.get(date, 1.0)
    if pd.isna(scale):
        scale = 1.0
    scale = np.clip(scale, 0.0, 1.0)

    selected = _select_offensive(rmom, offensive_assets, top_n, date)
    for asset in selected:
        if asset in all_cols:
            target[asset] = slot_weight * scale

    defensive_exposure = 1.0 - target.sum()
    if defensive_exposure > 0.0:
        best_def = _best_defensive(rmom, defensive_assets, date)
        if best_def is not None and best_def in all_cols:
            target[best_def] += defensive_exposure
        else:
            # Distribute remaining equally among selected offensive if no defensive
            if selected:
                per_slot = defensive_exposure / len(selected)
                for a in selected:
                    target[a] += per_slot
            else:
                # Edge case: assign to first available offensive
                for a in offensive_assets:
                    if a in all_cols and not pd.isna(rmom.at[date, a]):
                        target[a] = defensive_exposure
                        break

    target = target.clip(lower=0.0)
    total = target.sum()
    if total > 0:
        target = target / total
    return target


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate HAA variant with risk-adjusted momentum, continuous canary, and smoothing.

    Parameters (all under `params`):
        mom_lags: list[int]          momentum lookback periods (default [21,63,126,252])
        vol_window: int              volatility rolling window (default 63)
        canary_z_window: int         rolling z-score window for canary scale (default 252)
        top_n: int                   number of offensive slots (default 6)
        smooth: float                blending factor for new weights (0-1, default 0.3)
        defensive_assets: list[str]
        offensive_assets: list[str]
        canary_assets: list[str]
    """
    params = config.get("params", {})
    mom_lags = params.get("mom_lags", [21, 63, 126, 252])
    vol_window = params.get("vol_window", 63)
    canary_z_window = params.get("canary_z_window", 252)
    top_n = params.get("top_n", 6)
    smooth = params.get("smooth", 0.3)
    defensive_assets = params.get("defensive_assets", ["BIL", "BND"])
    offensive_assets = params.get("offensive_assets", ["SPY", "QQQ", "VGK", "VWO", "GLD", "TLT"])
    canary_assets = params.get("canary_assets", ["TIP", "BND", "GLD", "TLT"])

    all_assets = list(prices.columns)
    if prices.empty:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex([]), columns=all_assets)

    # Pre-compute all signals (strictly backward-looking)
    comp_mom = _composite_momentum(prices, mom_lags)
    vol = _trailing_volatility(prices, vol_window)
    rmom = _risk_adjusted_momentum(comp_mom, vol)
    canary = _canary_scale(rmom, canary_assets, canary_z_window)

    # Business month end dates (no lookahead)
    rebalance_dates = _rebalance_dates(prices)

    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=all_assets)
    prev_weights = None

    for date in rebalance_dates:
        if date not in rmom.index:
            continue
        new_w = _target_weights(date, rmom, canary, offensive_assets, defensive_assets, top_n)
        if prev_weights is not None:
            new_w = smooth * new_w + (1.0 - smooth) * prev_weights
        new_w = new_w.clip(lower=0.0)
        if new_w.sum() > 0:
            new_w = new_w / new_w.sum()
        weights.loc[date] = new_w
        prev_weights = new_w.copy()

    # Forward-fill to the full price index
    weights = weights.reindex(prices.index).ffill().fillna(0.0)
    return weights
