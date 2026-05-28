## Lessons from HAA Optimization

### Iteration 1 – Excessive Turnover in Canonical HAA-Balanced
- The canonical HAA-Balanced rebalances the Top‑4 momentum selection every month and the TIP canary frequently crosses zero, leading to extreme turnover (~8.5 annual).
- Smoothing the canary with a moving average drastically reduces false signals and the number of defensive switches.
- Letting the offensive allocation drift (buy‑and‑hold) between canary state changes eliminates turnover from rebalancing within the same market regime.
- Removing IEF from the offensive universe removes the tactical ambiguity of a single asset serving in both offensive and defensive roles.

### Iteration 2 – Implementing HAA‑Trend
- Used a 12‑month SMA of TIP momentum as the smoothed canary – state changes become rare, cutting turnover to near‑zero after the initial allocation.
- Adopted strict drift: holdings only update when the canary flips between offensive and defensive.
- Removed IEF from the offensive universe; used BIL and IEF as defensive options.
- Returned daily weights by forward‑filling monthly signals to match the input index, satisfying the output specification.
- Expected result: annual turnover << 1.5, drawdown protected by the slow‑but‑reliable TIP canary, and Sharpe improved by avoiding costly whipsaws.

### Iteration 3 – Formalizing HAA‑Trend Implementation
- Confirmed that the canonical HAA code produced turnover >8.5 in holdout, far above the 1.5 limit.
- Implemented full HAA‑Trend: TIP momentum SMA(12) determines canary state; rebalance only on state change.
- Removed IEF from offensive_universe in config; offensive now only 7 assets (SPY, IWM, VEA, VWO, VNQ, DBC, TLT).
- Defaulted to defensive state before sufficient smoothing history, protecting early drawdowns.
- Kept top‑4 selection and BIL/IEF defense when in offensive state, following original logic.
- Expect turnover to drop below 1.5, Sharpe to improve, while maintaining or improving max drawdown.

### Iteration 4 – HAA‑Trend Delivered
- The current iteration fully realizes HAA‑Trend: 12‑month SMA of TIP momentum, rebalance only on canary state change, and IEF removed from offense.
- Turnover plummeted from 8.57 to below 1.5, satisfying the hard constraint.
- The holdout Sharpe ratio (0.695) is below the reported ~1.25; this may reflect the shorter holdout window or the cost of excessive de‑risking during the 2022–2025 period.
- Regime stress tests (dotcom, GFC, COVID) passed with drawdowns under 15%.
- Future improvements could consider: a dual canary (e.g. TIP + another macro asset) to reduce single‑point failure, dynamic SMA period selection, or a less aggressive de‑risking rule in strong trending markets.

### Iteration 5 – Fixing Turnover Gate with HAA‑Trend
- The canonical HAA code (as initially provided) computed only monthly weights, which did not match the input DataFrame shape and produced extreme turnover.
- The evaluation framework expects a weights DataFrame with the same index as the input prices; we now forward‑fill rebalance weights to all days.
- The smoothed canary and state‑change‑only rebalancing dramatically reduce turnover. The backtest confirms turnover fell well below the 1.5 limit.
- The strategy now passes all gate checks (max drawdown, turnover, regimes).
- Holdout Sharpe remains modest (~0.7), but this is acceptable given the conservative de‑risking in volatile inflation periods; the primary objective of passing the turnover gate is achieved.
- Next steps to boost Sharpe: explore a dual canary (e.g., add a gold/commodity trend signal) or a less defensive transition rule.
