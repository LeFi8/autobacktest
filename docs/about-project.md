# About AutoBacktest

AutoBacktest is an autonomous, AI-driven quantitative trading strategy optimization system. It connects large language model (LLM) agents with deterministic backtesting and statistical evaluation pipelines to iteratively refine and validate quant trading strategies without human intervention.

## Business Goal
Automate the design, evaluation, and tuning of quant strategies. By pairing LLM reasoning with mathematical validation (DSR, bootstrapping, stress testing, transaction costs), the platform establishes a self-correcting strategy development loop that prevents overfitting and guarantees statistical significance before deploying strategy iterations.

## Target User Persona
- **Quantitative Researchers & Developers**: Who want to automate parameter searching and signal engineering.
- **Algotraders & Fund Managers**: Seeking a self-documenting, risk-mitigating pipeline for strategy lifecycle tracking.
- **Autonomous AI Agents**: Systems designed to run non-interactive loops optimizing performance metrics under target risk criteria.

## Primary Interaction Flows

```mermaid
graph TD
    A[Start Optimization Loop] --> B[Generate Strategy Baseline]
    B --> C[Generate 3 Candidates in Parallel]
    C --> D[Preflight Validation]
    D -- Fail --> F[Rollback & Feedback to LLM]
    D -- Pass --> E[Config Diversity Gate (explore mode)]
    E -- Fail --> F
    E -- Pass --> G[Vectorized Backtest Engine]
    G --> H[Returns Correlation Gate (explore mode)]
    H -- Fail --> F
    H -- Pass --> I[select Gate (in-sample)]
    I -- Fail --> F
    I -- Pass --> J[confirm Gate (holdout)]
    J -- Fail --> F
    J -- Pass --> K[Git Commit & Update Incumbent]
    K --> L[End Iterations / Report Leaders]
```

### Detailed Flow Steps
1. **Initiate Loop**: User defines constraints (e.g. max drawdown limit, maximum turnover, benchmark) in a configuration file and target strategy (`strategies/haa.py`).
2. **Multi-Candidate Generation**: The orchestrator requests **3 candidate edits in parallel** from the LLM, each containing proposed code changes, a YAML config, and updated lessons text. Candidates are deduplicated by the LessonStore (SQLite-backed, keyed by `(strategy, type, body_hash)`).
3. **Pre-Flight Verification** (per candidate, 8 checks):
    - Path traversal security check.
    - AST whitelist scan blocking unsafe imports, I/O operations, dunder escapes, cyclomatic complexity, and function size limits.
    - Pydantic config validation via `StrategyConfig`.
    - Dynamic compilation and isolated import.
    - Signature verification (`generate_signals(prices, config)` contract).
    - Smoke testing with 756 days of synthetic prices.
    - Lookahead sniff test: compares signals with vs. without future price data appended.
    - Lookahead shift test: shifts price history by 1 day and verifies consistent signal shift (for frequently-rebalancing strategies).
4. **Tier 1 Diversity Gate (Config Similarity)** — active only in **explore mode**:
   - Compares the candidate's proposed configuration fingerprint with all historical configs using min-max normalized parameters.
    - If similarity exceeds `DIVERSITY_CONFIG_THRESHOLD = 0.95` (configurable via `AUTOBACKTEST_DIVERSITY_CONFIG_THRESHOLD`), the candidate is rejected with a bounded retry (up to 2).
5. **Deterministic Evaluation**:
   - Fetches historical prices from cache or online (Yahoo Finance).
   - Generates daily signal weights from the active strategy file.
   - Runs a vectorized daily return computation (shifted by 1 day to prevent lookahead-bias).
   - Penalizes returns using dynamic rebalancing turnover costs, commission rates, bid-ask spreads, and market impact models.
6. **Tier 2 Diversity Gate (Returns Correlation)** — active only in **explore mode**:
   - Measures the Pearson correlation coefficient between the candidate's daily net returns and those of all previously recorded attempts in the SQLite ledger.
   - Rejects the candidate if return correlation `> DIVERSITY_RETURNS_THRESHOLD = 0.95` (configurable via `AUTOBACKTEST_DIVERSITY_RETURNS_THRESHOLD`) with any past attempt.
7. **Two-Phase Gate System**:
   - **Phase 1 — `select` (in-sample)**: Checks hard gates on the walk-forward aggregate: max drawdown, regime stress tests, turnover. If all pass, applies a target metric improvement tie-breaker and a DSR non-degradation check against the incumbent.
   - **Phase 2 — `confirm` (holdout)**: Only reached when `select` passes. Checks holdout max drawdown, turnover, and holdout DSR non-degradation. Each call counts as one **holdout peek** (budgeted — default limit of 20 peeks).
8. **Parameter Importance Tracking**: After each iteration, a Spearman rank correlation is computed between numeric config parameters and the target metric across all historical attempts, surfacing which parameters most influence performance.
9. **Ledger Commit / Rollback**:
   - If all gate criteria are met, the orchestrator commits the updated files to Git and registers the performance metrics, parameters, and net returns in the SQLite ledger database.
   - Otherwise, it rolls back strategy changes and sends diagnostic feedback to the LLM agent.
   - The orchestrator tracks an explore/exploit mode: accepted candidates switch to **exploit mode** (low temperature, focused refinement), while `EXPLOIT_PATIENCE = 3` consecutive failures without improvement force a return to **explore mode** (higher temperature).



