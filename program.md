# Objective

Improve the Keller/Keuning HAA-Balanced strategy (SSRN 4346906) by addressing its
known weaknesses while preserving its strengths. Specific targets:
- Increase the Sharpe ratio above the reported ~1.25
- Reduce maximum drawdown below the reported ~10%
- Maintain or improve the reported ~15.8% CAGR
- Reduce reliance on TIP as a single canary point of failure
- Address IEF's dual role in both offensive and defensive universes
- Improve diversification beyond the current 8-asset ceiling

# Constraints

- Maximum drawdown in the holdout period must not exceed 20%.
- Annualized portfolio turnover must remain below 2.0.
- The strategy must pass all regime stress tests (2000 dotcom crash, 2008 financial crisis, 2020 COVID crash).
- Monthly rebalancing on the last trading day must be preserved.
- Only `pandas` and `numpy` imports are permitted in the strategy code.
- The `generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame` signature must be preserved.

# HAA — Hybrid Asset Allocation

**Developer:** Dr. Wouter Keller and JW Keuning · **Year:** 2023 · **Type:** Dual + canary momentum, inflation-aware
**Paper:** *Dual and Canary Momentum with Rising Yields/Inflation* — SSRN 4346906

---

## Context

HAA is the sixth major strategy in the Keller/Keuning family, designed as a direct follow-up to BAA. Its key innovation: a single **TIPS-based canary (TIP)** that turns negative when inflation or rising yields threaten markets — the condition that broke VAA and DAA in 2022. It combines dual momentum (from GEM/VAA) with canary crash protection (from DAA) into a simpler, more robust hybrid.

Evolution: GEM → PAA → VAA → DAA → BAA → **HAA**

---

## Asset Universes

### HAA-Balanced (G8/T4)

| Universe | Assets | Purpose |
|---|---|---|
| **Offensive** | SPY, IWM, VEA, VWO, VNQ, DBC, IEF, TLT | 8 assets across 4 global classes; top 4 selected |
| **Canary** | TIP | Single canary; negative momentum → full defensive |
| **Defensive** | BIL, IEF | Capital preservation; best one chosen by momentum |

Offensive asset classes covered:
- US Equities: SPY (S&P 500), IWM (small-cap)
- Foreign Equities: VEA (developed ex-US), VWO (emerging markets)
- Alternatives: VNQ (US REITs), DBC (commodities)
- US Bonds: IEF (7–10yr Treasury), TLT (20yr+ Treasury)

### HAA-Simple

| Universe | Assets | Purpose |
|---|---|---|
| **Offensive** | SPY | Single risky asset |
| **Canary** | TIP | Same canary as Balanced |
| **Defensive** | BIL, IEF | Same defensive as Balanced |

---

## Momentum Formula

All assets scored using the **13612U** formula — unweighted average of 1, 3, 6, and 12-month total returns:

```
Momentum = (r1m + r3m + r6m + r12m) / 4
```

Where `r1m` = 1-month total return, `r3m` = 3-month, etc.
This is simpler than VAA's weighted composite (`12×m1 + 4×m3 + 2×m6 + m12`) and treats all lookbacks equally.

---

## Decision Rules

### HAA-Balanced (monthly, last trading day)

1. Compute momentum score for TIP, all 8 offensive assets, BIL, and IEF.
2. **Canary check:** If `TIP momentum ≤ 0`:
   - Go 100% to the better of BIL or IEF (whichever has higher momentum score).
3. **Canary clear** (`TIP momentum > 0`): Apply dual momentum to offensive universe:
   - Rank all 8 offensive assets by momentum score (descending).
   - Select the **top 4** (TopX = T4; half of 8).
   - For each of the 4 selected slots (25% weight each):
     - If the asset's momentum score > 0 → **hold that asset**
     - If the asset's momentum score ≤ 0 → **replace with best defensive** (BIL or IEF)
4. Hold all positions until end of next month. Rebalance regardless of changes.

### HAA-Simple (monthly, last trading day)

1. Compute momentum for TIP, SPY, BIL, IEF.
2. If `TIP momentum ≤ 0` → hold best defensive (BIL or IEF).
3. If `TIP momentum > 0` AND `SPY momentum > 0` → hold 100% SPY.
4. If `TIP momentum > 0` BUT `SPY momentum ≤ 0` → hold best defensive (BIL or IEF).

---

## Reported Performance

| Variant | CAGR | Max Drawdown | Sharpe | Source |
|---|---|---|---|---|
| **HAA-Balanced** | ~15.8% | ~10.0% | ~1.25 | Allocate Smartly |
| **HAA-Simple** | ~13.0% | ~17.2% | ~0.79 | Allocate Smartly |

*Allocate Smartly strongly prefers HAA-Balanced — it outperforms HAA-Simple on every metric. HAA-Simple exists as a rules-verification tool, not as a recommended implementation.*

---

## Strengths
- **Best Sharpe in the Keller/Keuning family** (~1.25 for Balanced) — better than HAA's predecessors (DAA, BAA, VAA)
- **Very low drawdown** (~10%) relative to ~15.8% CAGR — the best CAGR/MDD ratio of any Keller strategy
- TIPS canary explicitly detects inflation and rising-yield regimes — addresses the 2022 failure mode
- Simpler momentum formula (unweighted 13612U) vs the weighted VAA formula — easier to implement and verify
- Low average cash fraction compared to BAA (~half of BAA's cash exposure)

## Weaknesses
- TIP canary is a single point of failure — if TIPS behave unexpectedly, the whole portfolio is misled
- Offensive universe has only 8 assets → moderate diversification ceiling
- Backtest period not as long as earlier Keller strategies; overfitting risk from iterative strategy refinement
- IEF appears in both the offensive AND defensive universe — creates a subtle dependency between the two
