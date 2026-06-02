"""Hybrid Asset Allocation (HAA) Optimized (G12/T6) per Keller 2023.

Implements the 13612U momentum scoring, dual-canary (TIP/BND) check,
top-N offensive selection (half of offensive universe) with defensive
substitution, and config-driven defensive allocation (BIL/BND).
"""

from typing import Any

import numpy as np
import pandas as pd


def _momentum_13612u(prices: pd.Series, current_date: pd.Timestamp) -> float:
    """Compute the unweighted 13612U momentum score.

    Returns the simple average of 1-month, 3-month, 6-month, and 12-month
    simple returns ending at current_date:
        Momentum = (r_1m + r_3m + r_6m + r_12m) / 4

    If insufficient price history for any window, that window is omitted
    from the average.
    """
    lookbacks = [21, 63, 126, 252]
    returns = []
    for lb in lookbacks:
        try:
            idx = prices.index.get_loc(current_date)
            if idx >= lb:
                r = (prices.iloc[idx] / prices.iloc[idx - lb]) - 1.0
                returns.append(r)
        except (KeyError, IndexError):
            continue
    if not returns:
        return -1.0
    return float(np.mean(returns))


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate HAA Optimized portfolio weights.

    Args:
        prices: Daily prices DataFrame (DatetimeIndex, columns=tickers).
        config: Configuration dict with keys:
            - params.offensive_assets: list of offensive tickers
            - params.defensive_assets: list of defensive tickers (BIL, BND)
            - params.canary_asset: legacy single canary ticker (string)
            - params.canary_assets: multi-canary tickers (list)

    Returns:
        pd.DataFrame: Weights indexed by rebalance dates.
    """
    params = config.get("params", {})
    offensive_assets = params.get(
        "offensive_assets",
        [
            "SPY",
            "IWM",
            "QQQ",
            "VGK",
            "EWJ",
            "VWO",
            "VNQ",
            "GLD",
            "DBC",
            "HYG",
            "LQD",
            "TLT",
        ],
    )
    defensive_assets = params.get("defensive_assets", ["BIL", "BND"])
    canary_raw = params.get("canary_assets") or params.get("canary_asset", "TIP")
    canary_assets = [canary_raw] if isinstance(canary_raw, str) else list(canary_raw)

    all_assets = list(prices.columns)

    if prices.empty:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex([]), columns=all_assets)

    # Rebalance on last trading day of each month
    monthly_dates = prices.groupby(prices.index.to_period("M")).tail(1).index

    weights = pd.DataFrame(0.0, index=monthly_dates, columns=all_assets)

    top_n = max(1, len(offensive_assets) // 2)
    slot_weight = 1.0 / top_n

    for date in monthly_dates:
        # Compute momentum for all assets
        mom_scores = {}
        for asset in all_assets:
            if asset in prices.columns:
                mom_scores[asset] = _momentum_13612u(prices[asset], date)

        # Best defensive asset (shared across defensive and substitution paths)
        def_mom_vals = [(d, mom_scores.get(d, -1.0)) for d in defensive_assets if d in all_assets]
        best_def = max(def_mom_vals, key=lambda x: x[1])[0] if def_mom_vals else None

        # Dual-canary check: OR gate — any canary negative triggers full defensive
        if any(mom_scores.get(c, -1.0) <= 0 for c in canary_assets):
            if best_def is not None:
                weights.loc[date, best_def] = 1.0
            continue

        # Clear skies: rank offensive assets, pick top N
        off_mom_vals = [(o, mom_scores.get(o, -1.0)) for o in offensive_assets if o in all_assets]
        off_mom_vals.sort(key=lambda x: x[1], reverse=True)
        top_selected = off_mom_vals[:top_n]

        for asset, mom in top_selected:
            if mom > 0:
                weights.loc[date, asset] = slot_weight
            elif best_def is not None:
                weights.loc[date, best_def] = weights.loc[date, best_def] + slot_weight

    return weights
