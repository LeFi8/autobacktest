"""Hybrid Asset Allocation (HAA) Balanced (G8/T4) per Keller 2023.

Implements the 13612U momentum scoring, TIP canary check, top-4 offensive
selection with defensive substitution, and dual-defense (BIL/IEF) allocation.
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
                start = prices.index[idx - lb]
                r = (prices.loc[current_date] / prices.loc[start]) - 1.0
                returns.append(r)
        except (KeyError, IndexError):
            continue
    if not returns:
        return -1.0
    return float(np.mean(returns))


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate HAA Balanced portfolio weights.

    Args:
        prices: Daily prices DataFrame (DatetimeIndex, columns=tickers).
        config: Configuration dict with keys:
            - params.offensive_assets: list of 8 offensive tickers
            - params.defensive_assets: list of 2 defensive tickers (BIL, IEF)
            - params.canary_asset: ticker for the TIP canary

    Returns:
        pd.DataFrame: Weights indexed by rebalance dates.
    """
    params = config.get("params", {})
    offensive_assets = params.get("offensive_assets", ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"])
    defensive_assets = params.get("defensive_assets", ["BIL", "IEF"])
    canary_asset = params.get("canary_asset", "TIP")

    all_assets = list(prices.columns)

    if prices.empty:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex([]), columns=all_assets)

    # Rebalance on last trading day of each month
    monthly_dates = prices.groupby(prices.index.to_period("M")).tail(1).index

    weights = pd.DataFrame(0.0, index=monthly_dates, columns=all_assets)

    for date in monthly_dates:
        # Compute momentum for all assets
        mom_scores = {}
        for asset in all_assets:
            if asset in prices.columns:
                mom_scores[asset] = _momentum_13612u(prices[asset], date)

        # Canary check
        canary_mom = mom_scores.get(canary_asset, -1.0)

        if canary_mom <= 0:
            # Defensive: 100% to the better defensive asset
            def_mom_vals = [(d, mom_scores.get(d, -1.0)) for d in defensive_assets if d in all_assets]
            if not def_mom_vals:
                weights.loc[date, :] = 0.0
                continue
            best_def = max(def_mom_vals, key=lambda x: x[1])[0]
            weights.loc[date, best_def] = 1.0
        else:
            # Clear skies: rank offensive assets, pick top 4
            off_mom_vals = [(o, mom_scores.get(o, -1.0)) for o in offensive_assets if o in all_assets]
            off_mom_vals.sort(key=lambda x: x[1], reverse=True)
            top_4 = off_mom_vals[:4]

            # Best defensive asset (for substitution)
            def_mom_vals = [(d, mom_scores.get(d, -1.0)) for d in defensive_assets if d in all_assets]
            best_def = max(def_mom_vals, key=lambda x: x[1])[0] if def_mom_vals else defensive_assets[0]

            slot_weight = 1.0 / 4.0
            for asset, mom in top_4:
                if mom > 0:
                    weights.loc[date, asset] = slot_weight
                else:
                    weights.loc[date, best_def] = weights.loc[date, best_def] + slot_weight

    return weights
