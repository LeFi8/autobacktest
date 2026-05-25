"""Historical Asset Allocation (HAA) quant strategy."""

from typing import Any

import numpy as np
import pandas as pd


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate Historical Asset Allocation strategy weights.

    Args:
        prices: Daily prices DataFrame (DatetimeIndex).
        config: Strategy configuration dictionary.

    Returns:
        pd.DataFrame: Strategy weights DataFrame indexed by rebalance dates.
    """
    lookback = int(config.get("momentum_lookback", 12))
    offensive_universe = config.get(
        "offensive_universe",
        ["SPY", "IWM", "QQQ", "VGK", "EWJ", "VWO", "VNQ", "DBC"],
    )
    defensive_universe = config.get(
        "defensive_universe",
        ["IEF", "BIL"],
    )
    filter_ticker = config.get("filter_ticker", "TIP")

    # Extract daily prices of actual last trading days for each month
    last_trading_days = prices.groupby(prices.index.to_period("M")).apply(
        lambda x: x.index[-1]
    )
    monthly_prices = prices.loc[last_trading_days]

    # Calculate momentum score:
    # 12 * (p0/p1 - 1) + 4 * (p0/p3 - 1) + 2 * (p0/p6 - 1) + (p0/p12 - 1)
    mom_scores = pd.DataFrame(index=monthly_prices.index, columns=prices.columns)

    start_idx = max(12, lookback)
    for i in range(start_idx, len(monthly_prices)):
        date = monthly_prices.index[i]
        p0 = monthly_prices.iloc[i]
        p1 = monthly_prices.iloc[i - 1]
        p3 = monthly_prices.iloc[i - 3]
        p6 = monthly_prices.iloc[i - 6]
        p12 = monthly_prices.iloc[i - lookback]

        # Calculate scores safely preventing division by zero or NaN cascades
        score = (
            12.0 * (p0.div(p1.replace(0.0, np.nan)).fillna(1.0) - 1.0)
            + 4.0 * (p0.div(p3.replace(0.0, np.nan)).fillna(1.0) - 1.0)
            + 2.0 * (p0.div(p6.replace(0.0, np.nan)).fillna(1.0) - 1.0)
            + 1.0 * (p0.div(p12.replace(0.0, np.nan)).fillna(1.0) - 1.0)
        )
        mom_scores.loc[date] = score

    # Drop the first lookback months since we need lookback to calculate scores
    mom_scores = mom_scores.dropna(how="all")

    # Generate weights DataFrame aligned with monthly dates
    weights = pd.DataFrame(0.0, index=mom_scores.index, columns=prices.columns)

    for date in mom_scores.index:
        scores_t = mom_scores.loc[date]
        tip_score = scores_t.get(filter_ticker, -1.0)

        if tip_score > 0.0:
            # Risk-On: Invest in offensive asset with the highest positive score
            valid_off = [t for t in offensive_universe if t in scores_t.index]
            off_scores = scores_t[valid_off].dropna()
            if not off_scores.empty and off_scores.max() > 0.0:
                best_off = off_scores.idxmax()
                weights.loc[date, best_off] = 1.0
            else:
                # Fallback to defensive if no offensive score is positive
                valid_def = [t for t in defensive_universe if t in scores_t.index]
                def_scores = scores_t[valid_def].dropna()
                if not def_scores.empty:
                    best_def = def_scores.idxmax()
                    weights.loc[date, best_def] = 1.0
        else:
            # Risk-Off: Invest in defensive asset with the highest score
            valid_def = [t for t in defensive_universe if t in scores_t.index]
            def_scores = scores_t[valid_def].dropna()
            if not def_scores.empty:
                best_def = def_scores.idxmax()
                weights.loc[date, best_def] = 1.0

    # Return sparse weights DataFrame directly to allow the backtester's
    # reindex and ffill to work correctly
    return weights
