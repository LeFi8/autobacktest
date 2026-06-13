# Strategy Authoring Guide

> [Documentation Hub](index.md) | [About Project](about-project.md) | [Architecture](architecture.md) | [API Reference](api-reference.md)

This guide covers everything you need to know to create, configure, and optimize trading strategies with AutoBacktest.

---

## Overview

A strategy in AutoBacktest is a Python module that generates portfolio allocation weights from price data. The LLM optimizer iteratively mutates your strategy code and configuration to improve performance while respecting your constraints.

Each strategy consists of two files inside a **subdirectory**:

```
strategies/
└── my_strategy/
    ├── strategy.py      # Signal generation code
    ├── config.yaml       # Pydantic-validated parameters
    └── program.md        # LLM objective + constraints (optional, can live at root)
```

---

## File Structure

| File | Required | Purpose |
|------|----------|---------|
| `strategies/<name>/strategy.py` | Yes | Exports `generate_signals()` — the core signal logic |
| `strategies/<name>/config.yaml` | Yes | Strategy parameters validated by `StrategyConfig` Pydantic model |
| `strategies/<name>/program.md` | No | Strategy-specific LLM objective (can also use root `program.md`) |

**Legacy layout**: Flat files `strategies/<name>.py` + `configs/<name>.yaml` are still supported as a fallback, but the subdirectory layout is preferred for new strategies.

---

## Strategy Code (`strategy.py`)

### Required Signature

Every strategy must export a single function with this exact signature:

```python
def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `prices` | `pd.DataFrame` | Daily close prices with a `DatetimeIndex` and ticker columns |
| `config` | `dict[str, Any]` | Flattened strategy configuration (all YAML fields + custom `params`) |

### Return Value

A `pd.DataFrame` of portfolio weights where:
- **Index**: Rebalance dates (subset of the price index)
- **Columns**: Ticker symbols (must match `prices.columns`)
- **Values**: Allocation weights (must sum to 1.0 on each rebalance date)
- **Zero weights**: Assets with 0 weight are excluded from the portfolio on that date

### Example: Equal Weight

```python
def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    universe = config.get("universe", [])
    cash_asset = config.get("params", {}).get("cash_asset", "BIL")

    if prices.empty:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex([]), columns=prices.columns)

    available = set(prices.columns)
    start = prices.index.min()
    end = prices.index.max()
    rebalance_dates = pd.date_range(start=start, end=end, freq="BME", tz=prices.index.tz).intersection(prices.index)
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)

    for date in rebalance_dates:
        valid_assets = [t for t in universe if t in available]
        if valid_assets:
            w = 1.0 / len(valid_assets)
            for asset in valid_assets:
                weights.loc[date, asset] = w

    return weights
```

### Key Rules

1. **No lookahead bias**: Never access future price data. The backtester uses a 1-day weight lag, but your code should also be strictly causal.
2. **Monthly rebalancing**: Most strategies rebalance on the last business day of each month (`freq="BME"`). The optimizer preserves this frequency.
3. **Cash asset**: Always include a cash/risk-free asset (e.g., `BIL`) in the universe and use it as a fallback when no assets meet your criteria.
4. **Empty prices**: Handle the empty `prices` DataFrame edge case gracefully.

---

## Configuration (`config.yaml`)

### Core Fields

These fields are defined by the `StrategyConfig` Pydantic model and have validation constraints:

| Field | Type | Default | Constraints | Description |
|-------|------|---------|-------------|-------------|
| `universe` | `list[str]` | _required_ | `min_length=1` | Asset tickers to trade |
| `benchmark` | `str` | `SPY` | non-empty | Benchmark index ticker |
| `momentum_lookback` | `int` | `12` | `>= 1` | Momentum lookback window (months) |
| `max_drawdown_limit` | `float` | `0.20` | `0.0–1.0` | Maximum permitted drawdown |
| `turnover_limit` | `float` | `2.0` | `> 0` | Maximum annualized turnover |
| `borrow_cost_bps` | `float` | `100.0` | `>= 0` | Annualized short borrowing cost (bps) |
| `params` | `dict` | `{}` | — | Strategy-specific custom parameters |

### Gate & Evaluation Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `min_improvement` | `float` | `0.0` | Minimum target-metric improvement for select gate |
| `select_min_return_ratio` | `float` | `0.5` | Min fraction of baseline return for select gate |
| `require_dsr_non_degradation` | `bool` | `True` | Enforce DSR non-degradation in select gate |
| `enable_holdout_confirmation` | `bool` | `True` | Enable holdout confirmation phase |
| `holdout_min_improvement` | `float` | `0.0` | Tolerance for holdout DSR non-degradation |
| `metric_floor` | `float \| None` | `None` | Absolute target-metric floor |
| `metric_return_tradeoff` | `float` | `0.0` | Acceptable metric reduction per 1pp return increase |
| `select_compare_metric` | `str` | `"deflated"` | `"deflated"` or `"raw"` for select gate comparison |
| `select_improvement_tol` | `float` | `0.02` | Near-tie tolerance in select gate |
| `pbo_limit` | `float \| None` | `None` | PBO ceiling for select gate |

### Custom Parameters

Strategy-specific parameters go in the `params` dict to avoid collisions with schema fields:

```yaml
params:
  cash_asset: BIL
  top_n: 3
  lookback_window: 60
```

Access them in code via `config.get("params", {}).get("cash_asset", "BIL")`.

---

## Program Spec (`program.md`)

The program file tells the LLM what to optimize and what constraints to respect. It must have two H1 headers:

```markdown
# Objective

Describe your optimization target. Examples:
- "Increase Sharpe ratio above 1.2 while keeping drawdown below 15%"
- "Improve risk-adjusted returns; reduce turnover below 1.5"

# Constraints

List hard constraints:
- Maximum drawdown must not exceed 20%.
- Annualized turnover must remain below 2.0.
- Only pandas and numpy imports are permitted.
- The generate_signals signature must be preserved.
```

**Required sections**:
- `# Objective` — What the LLM should optimize toward
- `# Constraints` — Hard limits the LLM must never violate

**Optional sections**: Add a `## Strategy Details` section for background, references, or design notes.

---

## Scaffolding a New Strategy

Use the `init-strategy` command to generate boilerplate:

```bash
uv run autobacktest init-strategy --name my_strategy
```

### Interactive Mode

The command prompts for:
- Universe (comma-separated tickers)
- Benchmark ticker
- Drawdown limit
- Turnover limit
- Momentum lookback
- Template selection
- Cash asset

### Silent (Non-Interactive) Mode

Pass `--universe` to skip all prompts:

```bash
uv run autobacktest init-strategy \
    --name GTAA13 \
    --universe SPY,TLT,GLD,BIL \
    --benchmark SPY \
    --drawdown 0.20 \
    --turnover 2.0 \
    --lookback 12 \
    --template momentum-rotation \
    --cash-asset BIL
```

### Templates

| Template | Description |
|----------|-------------|
| `equal-weight` (default) | Allocates `1/N` to each asset on rebalance dates |
| `momentum-rotation` | Ranks by trailing return, selects top N, allocates equally |

### Generated Files

| File | Purpose |
|------|---------|
| `strategies/<name>/strategy.py` | Runnable signal-generation code |
| `strategies/<name>/config.yaml` | Pydantic-validated configuration |
| `program-<name>.md` | LLM objective/constraints template |

---

## Import Whitelist

Strategy code is restricted to these modules only (enforced by AST linter):

```
pandas, numpy, math, typing, scipy, dataclasses, collections,
itertools, functools, decimal, statistics, numbers, json
```

**Blocked patterns**:
- Wildcard imports (`from x import *`)
- Dangerous builtins (`open`, `exec`, `eval`, `compile`)
- Dunder attribute access (`__import__`, `__builtins__`)
- Any module not in the whitelist

The whitelist is configurable via `AUTOBACKTEST_SAFE_IMPORTS_WHITELIST` in `.env`.

---

## Validation Pipeline

Every candidate mutation goes through 8 validation checks before evaluation:

### 1. Path Traversal Security
Verifies no path escape attacks in strategy/config file names.

### 2. AST Scan
Static analysis blocks:
- Non-whitelisted imports
- Dangerous I/O operations (`open`, `exec`, `eval`, `compile`, pandas I/O)
- Dunder escapes in format strings
- Cyclomatic complexity exceeding `AUTOBACKTEST_MAX_CYCLOMATIC_COMPLEXITY` (default: 25)
- Function line count exceeding `AUTOBACKTEST_MAX_FUNCTION_LINES` (default: 100)

### 3. Undefined Name Detection
Catches LLM hallucinations — misspelled identifiers, out-of-scope variables, and references to undefined names. Handles tuple unpacking, comprehension targets, lambda parameters, and closure scopes.

### 4. Pydantic Config Validation
Validates YAML against `StrategyConfig` with `extra="forbid"` — the LLM cannot inject arbitrary top-level keys.

### 5. Dynamic Compilation & Import
Runs within isolated execution blocks to catch syntax and runtime errors.

### 6. Signature Verification
Verifies `generate_signals` accepts the required `(prices, config)` positional parameters.

### 7. Smoke Test
Generates 756 days of synthetic prices and verifies the strategy executes without errors.

### 8. Lookahead Bias Detection
- **Sniff test**: Evaluates with and without future price data appended. If signals change when future data is visible, the strategy is flagged.
- **Shift test**: Shifts price history by 1 trading day and verifies signals shift consistently (applies to strategies with >15% daily rebalancing activity).

---

## Running the Optimization Loop

```bash
uv run autobacktest run \
    --program program.md \
    --strategy my_strategy \
    --iterations 10
```

### What Happens

1. **Baseline evaluation**: Your current strategy is evaluated to establish a performance baseline.
2. **LLM mutation**: The LLM generates N candidate mutations (code + config edits) in parallel.
3. **Preflight validation**: Each candidate passes through the 8-check validation pipeline.
4. **Config diversity gate** (Tier 1): Rejects candidates with >95% config similarity to past attempts.
5. **Backtesting**: Walk-forward (in-sample) + holdout (out-of-sample) evaluation.
6. **Returns correlation gate** (Tier 2): Rejects candidates with >95% return correlation to past attempts.
7. **Two-phase gate**: `select` (in-sample metrics, DSR non-degradation) → `confirm` (holdout confirmation).
8. **Commit or rollback**: Passing candidates are committed to git; failures are rolled back with structured feedback.

### Useful Flags

| Flag | Description |
|------|-------------|
| `--iterations N` | Number of optimization iterations |
| `--quiet` | Suppress verbose output |
| `--json` | Output raw JSON instead of Rich dashboard |
| `--early-stop-patience N` | Stop after N consecutive failed iterations (default: 10) |

---

## Common Patterns

### Equal Weight Allocation

```python
for date in rebalance_dates:
    valid_assets = [t for t in universe if t in available]
    w = 1.0 / len(valid_assets)
    for asset in valid_assets:
        weights.loc[date, asset] = w
```

### Momentum Rotation

```python
for date in rebalance_dates:
    hist = prices.loc[:date]
    returns = hist.pct_change(trading_days).iloc[-1]
    ranked = returns.dropna().sort_values(ascending=False)
    selected = [t for t in ranked.index if t != cash_asset and ranked[t] > 0][:top_n]

    if not selected:
        weights.loc[date, cash_asset] = 1.0
    else:
        w = 1.0 / len(selected)
        for asset in selected:
            weights.loc[date, asset] = w
```

### Volatility-Weighted Allocation

```python
for date in rebalance_dates:
    hist = prices.loc[:date].pct_change().iloc[-lookback_window:]
    vols = hist.std()
    inv_vol = 1.0 / vols.replace(0, float("nan"))
    alloc = inv_vol / inv_vol.sum()
    for asset in alloc.index:
        if asset in available:
            weights.loc[date, asset] = alloc[asset]
```

---

## Debugging

### Common Failures

| Error | Cause | Fix |
|-------|-------|-----|
| `ast_blocked_import` | Used a non-whitelisted import | Remove the import or use an allowed module |
| `undefined_name` | LLM introduced a typo or out-of-scope variable | Check variable names in the strategy code |
| `config_schema_invalid` | YAML violates `StrategyConfig` constraints | Check field types, ranges, and `extra="forbid"` |
| `signature_mismatch` | `generate_signals` signature changed | Preserve `def generate_signals(prices, config)` |
| `smoke_test_failed` | Strategy crashes on synthetic data | Test locally with `autobacktest llm-test` |
| `lookahead_detected` | Strategy accesses future data | Ensure all logic is strictly causal |
| `ast_cyclomatic_complexity_exceeded` | Too many nested conditionals | Simplify logic or extract helper functions |

### Inspecting Gate Outputs

Run a standalone evaluation to see detailed metrics:

```bash
uv run autobacktest evaluate --strategy my_strategy
```

### Testing LLM Edits

Test a specific prompt against preflight checks without running the full loop:

```bash
uv run autobacktest llm-test "Add momentum filter" --strategy my_strategy
```

### Viewing Historical Performance

```bash
uv run autobacktest report
```

---

## Reference Strategies

See the following working examples in the repository:

| Strategy | Location | Description |
|----------|----------|-------------|
| Equal Weight | `strategies/equal_weight/` | Simplest starting point — 1/N allocation |
| Momentum Rotation | `strategies/momentum_rotation/` | Ranks by trailing return, selects top N |
