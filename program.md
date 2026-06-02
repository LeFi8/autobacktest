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
- The strategy must pass all regime stress tests (2008 GFC, 2020 COVID crash, 2022 bear market).
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

### HAA-Optimized (G12/T6)

| Universe | Assets | Purpose |
|---|---|---|
| **Offensive** | SPY, IWM, QQQ, VGK, EWJ, VWO, VNQ, GLD, DBC, HYG, LQD, TLT | 12 assets across diverse classes; top 6 selected |
| **Canary** | TIP, BND | Dual canary; if either shows negative momentum → full defensive |
| **Defensive** | BIL, BND | Capital preservation; best one chosen by momentum |

Offensive asset classes covered:
- US Equities: SPY (S&P 500), IWM (small-cap), QQQ (Nasdaq 100)
- Foreign Equities: VGK (Europe), EWJ (Japan), VWO (emerging markets)
- Alternatives: VNQ (US REITs), DBC (commodities), GLD (Gold)
- Bonds: TLT (20yr+ Treasury), LQD (Corp), HYG (High Yield)

### HAA-Simple

| Universe | Assets | Purpose |
|---|---|---|
| **Offensive** | SPY | Single risky asset |
| **Canary** | TIP, BND | Same canaries as Optimized |
| **Defensive** | BIL, BND | Same defensive as Optimized |

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

### HAA-Optimized (monthly, last trading day)

1. Compute momentum score for TIP, BND, all 12 offensive assets, and BIL.
2. **Canary check:** If `TIP momentum ≤ 0` OR `BND momentum ≤ 0`:
   - Go 100% to the better of BIL or BND (whichever has higher momentum score).
3. **Canary clear** (`TIP momentum > 0` AND `BND momentum > 0`): Apply dual momentum to offensive universe:
   - Rank all 12 offensive assets by momentum score (descending).
   - Select the **top 6** (TopX = T6; half of 12).
   - For each of the 6 selected slots (~16.6% weight each):
     - If the asset's momentum score > 0 → **hold that asset**
     - If the asset's momentum score ≤ 0 → **replace with best defensive** (BIL or BND)
4. Hold all positions until end of next month. Rebalance regardless of changes.

### HAA-Simple (monthly, last trading day)

1. Compute momentum for TIP, BND, SPY, BIL.
2. If `TIP momentum ≤ 0` OR `BND momentum ≤ 0` → hold best defensive (BIL or BND).
3. If both canaries > 0 AND `SPY momentum > 0` → hold 100% SPY.
4. If both canaries > 0 BUT `SPY momentum ≤ 0` → hold best defensive (BIL or BND).

---

## Reported Performance

| Variant | CAGR | Max Drawdown | Sharpe | Source |
|---|---|---|---|---|
| **HAA-Optimized** | Target: >15.8% | Target: <10.0% | Target: >1.25 | Autobacktest Simulation |
| **HAA-Simple** | ~13.0% | ~17.2% | ~0.79 | Allocate Smartly |

*The Optimized variant is designed to exceed baseline Keller metrics by structurally reducing idiosyncratic risk and increasing asset breadth.*

---

## Strengths
- **Superior Risk-Adjusted Returns** — Targets a Sharpe > 1.25 by smoothing volatility via a 12-asset/6-selection structure.
- **Robust Crash Protection** — The dual-canary system (TIP + BND) avoids a single point of failure and cross-verifies inflation risk with aggregate rate risk.
- **Clean Universe Boundaries** — Eliminates the IEF overlap; offensive bonds (TLT/LQD/HYG) are strictly separated from defensive safe havens (BIL/BND).
- **Lower Single-Asset Drag** — Holding 6 assets at ~16.6% rather than 4 at 25% reduces the portfolio-level impact of a single momentum misfire.

## Weaknesses
- Increased data dependency due to 12 offensive assets + 2 canaries + 2 defensive assets (requires clean data for 15 distinct tickers).
- Expanding to 6 offensive selections may slightly increase transaction costs compared to the 4-asset version, making the 2.0 turnover limit a tight boundary.
