"""Hybrid Asset Allocation (HAA) quant strategy.

Reference: Keller & Keuning (2023) — "Dual and Canary Momentum with Rising
Yields/Inflation", SSRN 4346906.

Variants:
  - HAA-Balanced (default): Top-4 offensive selection from 8-asset universe.
  - HAA-Simple: Single-asset SPY with TIPS canary gate.
"""

from typing import Any

import numpy as np
import pandas as pd


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate Hybrid Asset Allocation strategy weights.

    Args:
        prices: Daily prices DataFrame (DatetimeIndex).
        config: Strategy configuration dictionary.

    Returns:
        pd.DataFrame: Strategy weights DataFrame indexed by rebalance dates.
    """
    variant = config.get("variant", "balanced")
    lookback = int(config.get("momentum_lookback", 12))
    top_x = int(config.get("top_x", 4))

    offensive_universe = config.get(
        "offensive_universe",
        ["SPY", "IWM", "VEA", "VWO", "VNQ", "DBC", "IEF", "TLT"],
    )
    defensive_universe = config.get(
        "defensive_universe",
        ["BIL", "IEF"],
    )
    filter_ticker = config.get("filter_ticker", "TIP")

    # Extract daily prices of last trading days for each month
    last_trading_days = prices.groupby(prices.index.to_period("M")).apply(lambda x: x.index[-1])
    monthly_prices = prices.loc[last_trading_days]

    # Calculate HAA momentum score (13612U):
    #   Momentum = (r1m + r3m + r6m + r12m) / 4
    # Unweighted average of 1, 3, 6, and 12-month total returns.
    mom_scores = pd.DataFrame(index=monthly_prices.index, columns=prices.columns)

    start_idx = max(12, lookback)
    for i in range(start_idx, len(monthly_prices)):
        date = monthly_prices.index[i]
        p0 = monthly_prices.iloc[i]
        p1 = monthly_prices.iloc[i - 1]
        p3 = monthly_prices.iloc[i - 3]
        p6 = monthly_prices.iloc[i - 6]
        p12 = monthly_prices.iloc[i - lookback]

        r1m = p0.div(p1.replace(0.0, np.nan)).fillna(1.0) - 1.0
        r3m = p0.div(p3.replace(0.0, np.nan)).fillna(1.0) - 1.0
        r6m = p0.div(p6.replace(0.0, np.nan)).fillna(1.0) - 1.0
        r12m = p0.div(p12.replace(0.0, np.nan)).fillna(1.0) - 1.0

        score = (r1m + r3m + r6m + r12m) / 4.0
        mom_scores.loc[date] = score

    smoothing_factor = float(config.get("smoothing_factor", 1.0))
    hysteresis_threshold = float(config.get("hysteresis_threshold", 0.0))
    momentum_buffer = float(config.get("momentum_buffer", 0.0))

    # Drop the first lookback months since we need lookback to calculate scores
    mom_scores = mom_scores.dropna(how="all")

    # Generate weights DataFrame aligned with monthly dates
    weights = pd.DataFrame(0.0, index=mom_scores.index, columns=prices.columns)

    for idx, date in enumerate(mom_scores.index):
        prev_date = mom_scores.index[idx - 1] if idx > 0 else None
        scores_t = mom_scores.loc[date].copy()

        # Apply Momentum Buffer to boost scores of currently held assets
        if prev_date is not None and momentum_buffer > 0.0:
            prev_weights = weights.loc[prev_date]
            held_assets = prev_weights[prev_weights > 0.0].index
            for asset in held_assets:
                if asset in scores_t.index:
                    scores_t[asset] += momentum_buffer

        tip_score = scores_t.get(filter_ticker, -1.0)
        target_w = pd.Series(0.0, index=prices.columns)

        if tip_score > 0.0:
            # Canary clear — apply dual momentum
            if variant == "simple":
                # HAA-Simple: single-asset SPY with defensive fallback
                spy_score = scores_t.get("SPY", -1.0)
                if spy_score > 0.0:
                    target_w["SPY"] = 1.0
                else:
                    valid_def = [t for t in defensive_universe if t in scores_t.index]
                    def_scores = scores_t[valid_def].dropna()
                    if not def_scores.empty:
                        best_def = def_scores.idxmax()
                        target_w[best_def] = 1.0
            else:
                # HAA-Balanced (or default): top-X selection
                valid_off = [t for t in offensive_universe if t in scores_t.index]
                off_scores = scores_t[valid_off].dropna()
                ranked = off_scores.sort_values(ascending=False)
                selected = ranked.head(top_x)

                # Determine best defensive asset for replacements
                valid_def = [t for t in defensive_universe if t in scores_t.index]
                def_scores = scores_t[valid_def].dropna()
                best_def = def_scores.idxmax() if not def_scores.empty else defensive_universe[0]

                slot_weight = 1.0 / max(len(selected), 1)
                for asset in selected.index:
                    if selected[asset] > 0.0:
                        target_w[asset] = slot_weight
                    else:
                        target_w[best_def] += slot_weight
        else:
            # Canary triggered — full defensive
            valid_def = [t for t in defensive_universe if t in scores_t.index]
            def_scores = scores_t[valid_def].dropna()
            if not def_scores.empty:
                best_def = def_scores.idxmax()
                target_w[best_def] = 1.0

        # Apply Turnover Hysteresis
        if prev_date is not None and hysteresis_threshold > 0.0:
            prev_w = weights.loc[prev_date]
            adjusted_w = prev_w.copy()
            for ticker in target_w.index:
                diff = abs(target_w[ticker] - prev_w[ticker])
                if diff >= hysteresis_threshold:
                    adjusted_w[ticker] = target_w[ticker]
            target_sum = target_w.sum()
            adjusted_sum = adjusted_w.sum()
            if adjusted_sum > 0.0:
                adjusted_w = (adjusted_w / adjusted_sum) * target_sum
            target_w = adjusted_w

        # Apply Exponential Weight Smoothing
        if prev_date is not None and smoothing_factor < 1.0:
            prev_w = weights.loc[prev_date]
            target_w = smoothing_factor * target_w + (1.0 - smoothing_factor) * prev_w

        weights.loc[date] = target_w

    return weights
