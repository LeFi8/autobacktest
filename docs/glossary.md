# Glossary

Quantitative finance and AutoBacktest terminology.

---

## Performance Metrics

### Sharpe Ratio
Annualised excess return per unit of total volatility. Measures risk-adjusted return. Higher is better.

$$\text{Sharpe} = \frac{\bar{r} - r_f}{\sigma_r} \times \sqrt{252}$$

### Sortino Ratio
Like Sharpe, but penalises only downside volatility (returns below zero or a target). More appropriate for strategies with asymmetric return distributions.

$$\text{Sortino} = \frac{\bar{r} - r_f}{\sigma_{down}} \times \sqrt{252}$$

### Information Ratio
Annualised active return per unit of tracking error relative to a benchmark. Measures how consistently a strategy outperforms.

$$\text{IR} = \frac{\bar{r}_a}{\sigma_{r_a}} \times \sqrt{252}$$

### Deflated Sharpe Ratio (DSR)
Sharpe Ratio adjusted for multiple testing bias. Accounts for the number of strategies tried, non-normality of returns, and serial correlation. A DSR below 1.0 suggests the Sharpe may be overfit.

Used in AutoBacktest as a **hard gate** — candidates must not degrade DSR.

### Maximum Drawdown
Largest peak-to-trough decline in portfolio value. A measure of worst-case loss.

### Turnover
Average daily change in portfolio weights. High turnover increases transaction costs.

---

## Overfitting Detection

### Probability of Backtest Overfitting (PBO)
Probability that a strategy's in-sample performance is due to overfitting rather than genuine alpha. Calculated via CSCV.

A PBO > 0.5 suggests the strategy is more likely overfit than not.

### Combinatorially Symmetric Cross-Validation (CSCV)
Method for estimating PBO. Splits backtest periods into *b* blocks, evaluates all C(b, b/2) train/test combinations, and checks whether in-sample ranking predicts out-of-sample ranking.

AutoBacktest uses 10 blocks → 252 train/test split combinations.

### Superior Predictive Ability (SPA) Test
Hansen's test for whether the best-performing strategy among *N* candidates is statistically better than chance. Accounts for selection bias.

Used in AutoBacktest via `autobacktest spa` command.

---

## Strategy Concepts

### Walk-Forward Analysis
滚动窗口 evaluation where the strategy is trained on a fixed window and tested on the subsequent period. Prevents lookahead bias and measures out-of-sample performance.

AutoBacktest default: 5-year train / 1-year test windows.

### Holdout Period
Final N years of data reserved exclusively for out-of-sample validation. Never used during optimization.

AutoBacktest default: 3 years.

### Regime Stress Testing
Evaluates strategy drawdowns during historical crash periods:
- **2008 GFC**: Global Financial Crisis
- **2020 COVID**: COVID-19 market crash
- **2022 Bear**: 2022 bear market

### Monte Carlo Block Bootstrap
Resamples daily returns using stationary or circular block bootstrap to estimate the distribution of performance metrics under resampling. Provides confidence intervals for Sharpe ratios.

### Lookahead Bias
Error where strategy logic accesses future data that wouldn't be available at decision time. AutoBacktest's vectorized backtest lags weights by 1 day to prevent this.

---

## AutoBacktest Pipeline

### Gate System
Two-phase validation:
1. **Select (in-sample)**: 9 checks including drawdown, turnover, metric improvement, DSR, regime stress
2. **Confirm (holdout)**: 3 checks on out-of-sample data — drawdown, turnover, DSR non-degradation

### Preflight Validation
AST-based static checks before evaluation:
- Import whitelist enforcement
- Cyclomatic complexity limits
- Function line limits
- Undefined name detection
- Signature validation

### Config Diversity Gate
Prevents redundant evaluations by rejecting candidates too similar to recently tested configurations. Uses cosine similarity on config vectors and Pearson correlation on return series.

### Parameter Importance Analysis
Tracks which strategy parameters most influence performance across iterations. Helps focus future LLM edits on high-impact parameters.

---

## Technical Terms

### Parquet Cache
Apache Parquet files storing downloaded price data locally to avoid repeated API calls.

### SQLite Ledger
Persistent store tracking all optimization runs, candidates, and metrics.

### Lesson Store
SQLite database capturing LLM "lessons learned" across iterations to inform future edits.

### Codemod
Automatic AST-based repair of deprecated pandas API calls (e.g., `DataFrame.append` → `pd.concat`).
