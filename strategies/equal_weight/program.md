# Objective

Optimize the **__NAME__** strategy (__UNIVERSE__ universe, benchmark: __BENCHMARK__).

Describe your specific targets here — the LLM optimizer uses this as its north star.

Examples:
- "Increase Sharpe ratio above 1.2 while keeping drawdown below __DRAWDOWN_PCT__%"
- "Improve risk-adjusted returns; reduce turnover below __TURNOVER__"
- "Maintain CAGR above 10% with max drawdown under __DRAWDOWN_PCT__%"

# Constraints

- Maximum drawdown in the holdout period must not exceed __DRAWDOWN_PCT__%.
- Annualized portfolio turnover must remain below __TURNOVER__.
- The strategy must pass all regime stress tests (2008 GFC, 2020 COVID, 2022 bear market).
- Monthly rebalancing on the last trading day must be preserved (momentum lookback: __LOOKBACK__ months).
- Only ``pandas`` and ``numpy`` imports are permitted in the strategy code.
- The ``generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame`` signature must be preserved.

## Strategy Details

Optional: Add background, references, or design notes here.

**Asset universe:** __UNIVERSE__
**Benchmark:** __BENCHMARK__

---

**Tip:** Run ``uv run autobacktest run --program program-__NAME__.md --strategy __NAME__ --iterations 5`` to start optimizing.
