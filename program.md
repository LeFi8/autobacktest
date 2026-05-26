# Objective

Improve the risk-adjusted performance of the HAA strategy on its historical universe.
Increase the Sharpe ratio of the holdout period while maintaining acceptable drawdown
and turnover characteristics.

# Constraints

- Maximum drawdown in the holdout period must not exceed 20%.
- Annualized portfolio turnover must remain below 2.0.
- The strategy must pass all regime stress tests (2000 dotcom crash, 2008 financial crisis, 2020 COVID crash).
- Only `pandas` and `numpy` imports are permitted in the strategy code.
- The `generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame` signature must be preserved.
