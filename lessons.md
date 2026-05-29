## Key Learnings
- Slow canary (smoothing≥12, hysteresis>0.02) causes excessive defensive crouching and near‑zero returns in range‑bound markets.
- Single‑asset canary (TIP alone) is brittle; composite canary (TIP+DBC) with skip‑NaN improves robustness and early‑window operation.
- Weighted momentum (12‑4‑2‑1) prioritises recent data and often yields higher offensive exposure than equal‑weighted 13612U.
- Offensive rebalance only on canary state change (offensive_rebalance_months=1) slashes lag while maintaining reasonable turnover.
- GLD addition to offense adds a momentum‑driven inflation hedge without increasing assets beyond T8.
- DBC skip‑NaN logic is essential to avoid pre‑2006 dead zone.
- Hysteresis elimination (0.005) keeps canary responsive, preventing lagged entries/exits.
- Iterative improvements must balance Sharpe, drawdown, and turnover walk‑forward, not just holdout metrics.
- **Latest iteration (50)**: re‑enabled weighted momentum, lowered smoothing to 3, set hysteresis to 0.005, added GLD, and forced monthly offensive refresh to maximise exposure; relies on raw canary fallback before sufficient smoothing history.
