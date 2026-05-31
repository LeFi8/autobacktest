## Lessons Learned

### Iteration 1 (Original HAA)
* The 13612U momentum score with a TIP canary generated extreme turnover (17.76) and drawdown (28.8%).
* Frequent flipping between risky assets due to short‑term momentum caused whipsaws.
* A more robust, slower signal is essential.

### Iteration 2 (Simple 10‑month SMA crossover)
* A plain trend filter failed the 2020 COVID regime test because it held through the crash.
* Any viable strategy must exit at the onset of a crash, not at month‑end.

### Iteration 3 (Vol‑filtered SMA)
* Adding a 20‑day vol stop (threshold 0.30) to an 18‑month SMA caused a KeyError when aligning daily indicators to monthly dates.
* Always use `.reindex(method='ffill')` to align date mismatches.

### Iteration 4 (Dual Momentum with Vol Stop) – Rejected for diversity
* 6‑month momentum on SPY+ IEF with a 35% vol stop produced excellent metrics but was rejected because its return series correlated >90% with earlier momentum attempts.
* Avoid replicating signal logic on nearly the same asset set; structural change is needed.

### Iteration 5 (Quarterly Risk‑Adjusted Momentum + Hysteresis) – Smoke Test Failure
* Quarterly rebalancing with risk‑adjusted momentum (12‑month return/vol) and a vol crash stop failed a smoke test because `pd.Grouper(freq='M')` is deprecated (use 'ME').
* Always test for pandas offset compatibility; use the newer 'ME' or period‑based grouping.

### Iteration 6 (Monthly Trend + Vol + Hysteresis, SPY Only) – Gate Failure (Turnover 3.02)
* A single‑asset (SPY/BIL) binary signal with 12‑month SMA trend, 21‑day vol>25% crash exit, and 3‑month re‑entry hysteresis lowered drawdown to 10% but still generated too many switches (annualized turnover 3.02).
* The 25% vol threshold was too low, causing unnecessary exits in moderate volatility; a higher threshold (e.g., 40%) or a continuous weighting scheme is needed to push turnover below 2.0.
* Hysteresis alone cannot fully compensate for an overly sensitive crash filter.

### Iteration 7 (Current) – Volatility‑Targeted Allocation
* Replaced binary on/off signals with a continuous weight: w = min(1, target_vol / EWMA_vol).
* The EWMA (span=126) balances responsiveness and smoothness, drastically reducing turnover while still forcing exposure cuts during spikes.
* This is structurally different from all prior momentum/trend designs, satisfying the diversity gate.
* Key principle: Continuous, volatility‑driven sizing avoids the whipsaw penalties of binary switching and keeps both turnover and drawdown low.