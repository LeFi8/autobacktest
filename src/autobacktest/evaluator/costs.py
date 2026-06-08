"""Transaction cost and turnover calculation for vectorized backtesting.

Computes net returns after accounting for commissions, bid-ask spreads,
market impact, and short borrowing costs.  Handles weight drift between
rebalance dates using a growth-adjusted drift model.
"""

import numpy as np
import pandas as pd


def calculate_turnover_and_costs(
    daily_returns: pd.Series,
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    commission_bps: float = 5.0,  # default 5bps
    spread_bps: float = 5.0,  # default 5bps
    impact_coef: float = 0.0,
    *,
    asset_returns: pd.DataFrame | None = None,
    borrow_cost_bps: float = 100.0,
    adaptive_slippage: bool = False,
    slippage_vol_window: int = 21,
    slippage_vol_cap: float = 3.0,
) -> tuple[pd.Series, pd.Series, float]:
    """Calculate net returns after transaction costs, borrowing costs, and annualized turnover.

    Accounts for weight drift due to price changes between rebalances.

    Args:
        daily_returns: Daily portfolio gross returns series.
        daily_weights: Daily portfolio weights DataFrame (index=DatetimeIndex).
        prices: Daily asset prices DataFrame.
        commission_bps: Commission fee in basis points (1 bp = 0.0001).
        spread_bps: Bid-ask spread in basis points.
        impact_coef: Market impact parameter (quadratic/linear cost).
        asset_returns: Pre-computed daily asset returns (skips pct_change()).
        borrow_cost_bps: Short borrowing cost in basis points annualized.
        adaptive_slippage: Use volatility-adaptive slippage model (default: False).
        slippage_vol_window: Rolling window for volatility estimation (default: 21).
        slippage_vol_cap: Cap multiplier for volatility-adaptive slippage (default: 3.0).

    Returns:
        tuple containing:
            - Net daily returns (pd.Series)
            - Net cumulative equity curve (pd.Series)
            - Annualized portfolio turnover rate (float)
    """
    if daily_returns.empty or daily_weights.empty:
        return daily_returns, (1.0 + daily_returns).cumprod(), 0.0

    # Calculate weight drift due to daily asset returns
    # drift_weights_t = weights_{t-1} * (1 + R_t) / (1 + Rp_t)
    asset_returns = asset_returns.loc[prices.index] if asset_returns is not None else prices.pct_change().fillna(0.0)
    shifted_weights = daily_weights.shift(1).fillna(0.0)

    # Calculate drift adjusted weights just before the end-of-day rebalance
    growth = shifted_weights * (1.0 + asset_returns)
    portfolio_growth = 1.0 + daily_returns

    # Avoid division by zero by replacing zero with NaN before dividing
    drift_weights = growth.div(portfolio_growth.replace(0.0, np.nan), axis=0).fillna(0.0)

    # The trade size is the absolute difference between new target weights
    # and the drift-adjusted weights from the previous period
    trades = (daily_weights - drift_weights).abs()

    # Sum of trades across assets at each day
    daily_trade_volume = trades.sum(axis=1)

    # Convert bps to decimals
    commission_rate = commission_bps / 10000.0
    spread_rate = spread_bps / 10000.0

    if adaptive_slippage:
        vol = asset_returns.rolling(slippage_vol_window, min_periods=slippage_vol_window).std()
        vol_median = vol.expanding(min_periods=slippage_vol_window).median()
        mult = vol.div(vol_median.replace(0.0, np.nan)).clip(lower=1.0, upper=slippage_vol_cap).fillna(1.0)
        spread_cost = (trades * spread_rate * mult).sum(axis=1)
        commission_cost = daily_trade_volume * commission_rate
        linear_costs = spread_cost + commission_cost
    else:
        cost_per_trade_value = commission_rate + spread_rate
        linear_costs = daily_trade_volume * cost_per_trade_value

    # Linear fee + market impact (quadratic term calculated per asset)
    transaction_costs = linear_costs + impact_coef * (trades**2).sum(axis=1)

    # Calculate short borrowing cost
    # w_{i,t} is the weight of asset i at day t.
    negative_weights = daily_weights.where(daily_weights < 0.0, 0.0).abs()
    daily_borrow_cost = negative_weights.sum(axis=1) * (borrow_cost_bps / 2520000.0)

    total_costs = transaction_costs + daily_borrow_cost

    # Deduct costs from gross daily returns
    net_returns = daily_returns - total_costs

    # Recalculate net equity curve
    net_equity = (1.0 + net_returns).cumprod()

    # Calculate annualized turnover
    # Total sum of absolute target weight changes divided by number of years.
    # Divide by 2 because each notional trade (A -> B) produces both a sell (A)
    # and a buy (B) in ``daily_trade_volume``, double-counting the notional.
    n_days = len(daily_returns)
    years = max(n_days / 252.0, 1.0 / 252.0)
    total_turnover = daily_trade_volume.sum() / 2.0
    annualized_turnover = float(total_turnover / years)

    return net_returns, net_equity, annualized_turnover
