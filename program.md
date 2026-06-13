# Objective

Describe the strategy you want to build or improve. Be specific about what you're
trying to achieve — the LLM optimizer uses this as its north star.

Examples of good objectives:
- "Improve the Keller/Keuning HAA-Balanced strategy: increase Sharpe > 1.25,
  reduce max drawdown < 10%, maintain CAGR > 15.8%"
- "Build a simple momentum strategy for SPY with a volatility overlay"
- "Create a gold-focused tactical allocation with inflation hedging"

# Constraints

List the hard constraints the strategy must satisfy. The optimizer will reject
any candidate that violates these.

Common constraints (edit or remove as needed):
- Maximum drawdown in the holdout period must not exceed 20%.
- Annualized portfolio turnover must remain below 2.0.
- The strategy must pass all regime stress tests (e.g., 2008 GFC, 2020 COVID,
  2022 bear market).
- Monthly rebalancing on the last trading day should be preserved.
- Only `pandas` and `numpy` imports are permitted in the strategy code.
- The `generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame`
  signature must be preserved.

## Strategy Details

Optional: Add any background, references, or design notes here.
Referenced papers, asset universe definitions, momentum formulas, decision rules,
and historical performance context all help the LLM understand what it's working with.

---

**Tip:** See `strategies/equal_weight/strategy.py` and `strategies/equal_weight/config.yaml` for a complete reference
strategy. Use `uv run autobacktest init-strategy --name my_strategy` to scaffold
a new one.
