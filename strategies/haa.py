"""Hybrid Asset Allocation (HAA) Trend strategy with composite canary.

Reference: Keller & Keuning (2023) — "Dual and Canary Momentum with Rising Yields/Inflation", SSRN 4346906.

Enhancements:
- Removes IEF from offensive universe to avoid dual role.
- Composite canary (TIP + DBC) smoothed with SMA(12) and hysteresis to reduce whipsaw.
- Rebalances only on canary state change, with optional periodic offensive refresh.
- Monthly rebalancing on last trading day, output aligned to daily index.
"""

from typing import Any

import numpy as np
import pandas as pd


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate HAA Trend weights with composite canary and state-change rebalancing.

    Args:
        prices: Daily prices DataFrame (DatetimeIndex).
        config: Configuration dictionary.

    Returns:
        Daily weights DataFrame matching the input index.
    """
    # --- Parameters from config ---
    variant = config.get("variant", "balanced")
    mom_lookback = int(config.get("momentum_lookback", 12))
    top_x = int(config.get("top_x", 4))

    offensive_universe = config.get("offensive_universe")
    defensive_universe = config.get("defensive_universe")
    canary_assets = config.get("canary_assets", ["TIP"])
    canary_smoothing_window = int(config.get("canary_smoothing_window", 12))
    canary_hysteresis = float(config.get("canary_hysteresis", 0.02))
    min_canary_period = int(config.get("min_canary_period", 12))
    offensive_rebalance_months = int(config.get("offensive_rebalance_months", 3))

    # Validate required assets are present in price data
    all_assets = canary_assets + offensive_universe + defensive_universe
    missing = [a for a in all_assets if a not in prices.columns]
    if missing:
        raise ValueError(f"Missing assets in price data: {missing}")

    # --- Extract last trading day of each month ---
    monthly_mask = prices.groupby(prices.index.to_period("M")).apply(lambda x: x.index[-1])
    monthly_prices = prices.loc[monthly_mask]

    # --- Monthly momentum scores (13612U) ---
    # r1m, r3m, r6m, r12m
    def calc_momentum(p: pd.DataFrame) -> pd.DataFrame:
        # Return momentum as average of 1,3,6,12-month total returns
        p0 = p
        p1 = p.shift(1)
        p3 = p.shift(3)
        p6 = p.shift(6)
        p12 = p.shift(mom_lookback)

        # Avoid division by zero; replace 0 with np.nan and fill after division
        r1 = (p0.div(p1.replace(0.0, np.nan)).fillna(1.0) - 1.0)
        r3 = (p0.div(p3.replace(0.0, np.nan)).fillna(1.0) - 1.0)
        r6 = (p0.div(p6.replace(0.0, np.nan)).fillna(1.0) - 1.0)
        r12 = (p0.div(p12.replace(0.0, np.nan)).fillna(1.0) - 1.0)
        return (r1 + r3 + r6 + r12) / 4.0

    mom_scores = calc_momentum(monthly_prices)
    # Start after enough history for longest lookback
    mom_scores = mom_scores.iloc[mom_lookback:]

    # --- Composite canary: average momentum of canary assets ---
    comp_canary = mom_scores[canary_assets].mean(axis=1)

    # --- Smooth canary with SMA ---
    smoothed = comp_canary.rolling(window=canary_smoothing_window,
                                   min_periods=min_canary_period).mean()
    # Drop periods where smoothed is NaN (insufficient history)
    smoothed = smoothed.dropna()

    # --- Determine canary state with hysteresis ---
    all_dates = mom_scores.index
    states = pd.Series("defensive", index=all_dates)
    prev_state = "defensive"
    for i, date in enumerate(all_dates):
        if date in smoothed.index:
            val = smoothed.loc[date]
            if val > canary_hysteresis:
                curr = "offensive"
            elif val < -canary_hysteresis:
                curr = "defensive"
            else:
                curr = prev_state  # keep previous if within band
        else:
            curr = "defensive"  # default before enough history
        states[date] = curr
        prev_state = curr

    # --- Generate monthly weights (only on state changes or periodic offense rebalance) ---
    w_monthly = pd.DataFrame(0.0, index=all_dates, columns=prices.columns)
    last_weight = pd.Series(0.0, index=prices.columns)
    last_off_rebalance_idx = -9999  # month index when last offensive rebalance occurred

    for i, date in enumerate(all_dates):
        state_now = states.loc[date]
        rebalance = False
        if i == 0:
            rebalance = True
        else:
            prev_date = all_dates[i-1]
            state_prev = states.loc[prev_date]
            if state_now != state_prev:
                rebalance = True
            elif state_now == "offensive":
                # Periodic rebalance inside offensive regime
                if (i - last_off_rebalance_idx) >= offensive_rebalance_months:
                    rebalance = True

        if rebalance:
            target_w = pd.Series(0.0, index=prices.columns)
            if state_now == "defensive":
                # Pick best defensive asset
                def_scores = mom_scores.loc[date, defensive_universe].dropna()
                best = def_scores.idxmax() if not def_scores.empty else defensive_universe[0]
                target_w[best] = 1.0
            else:  # offensive
                # Top-X selection from offensive universe
                off_scores = mom_scores.loc[date, offensive_universe].dropna()
                ranked = off_scores.sort_values(ascending=False)
                selected = ranked.head(top_x)

                # Best defensive substitute
                def_scores = mom_scores.loc[date, defensive_universe].dropna()
                best_def = def_scores.idxmax() if not def_scores.empty else defensive_universe[0]

                slot_weight = 1.0 / top_x
                for asset in selected.index:
                    if selected[asset] > 0.0:
                        target_w[asset] = slot_weight
                    else:
                        target_w[best_def] += slot_weight

            w_monthly.loc[date] = target_w
            last_weight = target_w
            if state_now == "offensive":
                last_off_rebalance_idx = i
        else:
            w_monthly.loc[date] = last_weight

    # --- Forward‑fill to daily index ---
    daily_weights = w_monthly.reindex(prices.index, method="ffill").fillna(0.0)
    daily_weights = daily_weights.clip(lower=0.0)

    return daily_weights
