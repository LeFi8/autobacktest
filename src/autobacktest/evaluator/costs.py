import numpy as np
import pandas as pd


def calculate_turnover_and_costs(
    daily_returns: pd.Series,
    daily_weights: pd.DataFrame,
    prices: pd.DataFrame,
    commission_bps: float = 5.0,  # default 5bps
    spread_bps: float = 5.0,  # default 5bps
    impact_coef: float = 0.0,
) -> tuple[pd.Series, pd.Series, float]:
    """Calculate net returns after transaction costs and annualized turnover.

    Accounts for weight drift due to price changes between rebalances.

    Args:
        daily_returns: Daily portfolio gross returns series.
        daily_weights: Daily portfolio weights DataFrame (index=DatetimeIndex).
        prices: Daily asset prices DataFrame.
        commission_bps: Commission fee in basis points (1 bp = 0.0001).
        spread_bps: Bid-ask spread in basis points.
        impact_coef: Market impact parameter (quadratic/linear cost).

    Returns:
        tuple containing:
            - Net daily returns (pd.Series)
            - Net cumulative equity curve (pd.Series)
            - Annualized portfolio turnover rate (float)
    """
    if daily_returns.empty or daily_weights.empty:
        return daily_returns, (1.0 + daily_returns).cumprod(), 0.0

    # Convert bps to decimals
    cost_per_trade_value = (commission_bps + spread_bps) / 10000.0

    # Calculate weight drift due to daily asset returns
    # drift_weights_t = weights_{t-1} * (1 + R_t) / (1 + Rp_t)
    asset_returns = prices.pct_change().fillna(0.0)
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

    # Linear fee + market impact (quadratic term)
    costs = daily_trade_volume * cost_per_trade_value + impact_coef * (daily_trade_volume**2)

    # Deduct transaction costs from gross daily returns
    net_returns = daily_returns - costs

    # Recalculate net equity curve
    net_equity = (1.0 + net_returns).cumprod()

    # Calculate annualized turnover
    # Total sum of absolute target weight changes divided by number of years
    n_days = len(daily_returns)
    years = max(n_days / 252.0, 1.0 / 252.0)
    total_turnover = daily_trade_volume.sum()
    annualized_turnover = float(total_turnover / years)

    return net_returns, net_equity, annualized_turnover
