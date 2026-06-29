# Objective

Optimize the **momentum_rotation** strategy for the SPY / QQQ / BIL universe, benchmarked against SPY.

Primary target: reduce crisis drawdowns while preserving competitive risk-adjusted returns.

Specific goals:
- Keep in-sample and holdout max drawdown at or below 20%.
- Improve observed Sharpe above the current baseline of roughly 0.64.
- Preserve annualized turnover at or below 2.0x.
- Pass regime stress tests for 2008 GFC, 2020 COVID, and 2022 bear market.

# Constraints

- Maximum drawdown in the holdout period must not exceed 20%.
- Annualized portfolio turnover must remain below 2.0x.
- The strategy must pass all regime stress tests (2008 GFC, 2020 COVID, 2022 bear market).
- Monthly rebalancing on the last trading day must be preserved (current momentum lookback: 12 months).
- Only ``pandas`` and ``numpy`` imports are permitted in the strategy code.
- The ``generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame`` signature must be preserved.

## Strategy Details

The baseline ranks SPY and QQQ by trailing momentum and moves to BIL only when no risky asset has positive momentum. This has left crisis drawdowns above the 20% gate. Prefer structural risk controls over tiny parameter-only edits.

Useful directions:
- Add a risk-off regime filter using trend and/or realized volatility.
- Cap combined SPY/QQQ exposure when trend is weak or volatility is elevated.
- Move partially or fully to BIL during weak momentum regimes.
- Keep the implementation simple enough to pass AST complexity and sandbox validation.

**Asset universe:** SPY, QQQ, BIL
**Benchmark:** SPY
