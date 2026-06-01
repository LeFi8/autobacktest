### Iteration 15 (Smoke test failure – index alignment)
- Error: "Unalignable boolean Series provided as indexer" when using a boolean Series from a subset to index another Series with a full index.
- Lesson: Never use boolean Series with .loc on a Series of different cardinality. Instead, extract the index of the positive weights and use that list of assets. Always maintain a complete pandas.Series for all weight vectors with the full risky_assets index to prevent partial indices.

### Iteration 16 (Diversity gate failure – high return correlation)
- The trend-following variant produced returns 93.1% correlated with the incumbent vol-targeted risk parity strategy. Despite different signal logic (SMA crossover and momentum), the resulting portfolio allocations and cash/risky splits were too similar over time.
- Lesson: To satisfy diversity constraints when the incumbent is a continuous vol-targeted approach, adopt a fundamentally different structure such as a canary-based on/off system with top-N momentum selection. Changing asset subsets and using a discrete risk-on/risk-off switch (canary) creates a return stream that is markedly less correlated.

### Iteration 17 (Variable name bug)
- Error: "name 'previous_risk_on' is not defined" because the loop used a different variable name than the initial declaration.
- Lesson: Always check variable name consistency across scopes. In stateful loops, define a single variable to hold the previous state and reference it identically throughout.

### General principles
- Continuous, smooth exposure signals (EWMA volatility, linear trend ramp) drastically reduce turnover compared to binary thresholds.
- Vol targeting with a leverage cap ≤ 1 ensures drawdown control while boosting risk‑adjusted returns.
- When fingerprint diversity is required, change the set of risky assets, not just parameters; using the full universe of risky assets provides both differentiation and potential diversification benefits.
- Momentum windows of 252 trading days (1 year) can reduce turnover and capture more persistent trends, which may improve Sharpe in trending environments.
- A canary-based managed momentum strategy with top‑N equal weighting offers a distinct return profile and can maintain moderate turnover with appropriate asset count, hysteresis, and minimum momentum filter.
- Hysteresis (e.g., 0.01) around the canary SMA reduces whipsawing and thus turnover, while a positive min_momentum threshold prevents owning assets with negative momentum, improving quality of selected holdings.