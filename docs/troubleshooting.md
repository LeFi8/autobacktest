# Troubleshooting

Common issues, error messages, and their resolutions.

---

## LLM Provider Issues

### `AuthenticationError` / `InvalidRequestError`

**Symptom:** LLM calls fail with auth errors.

**Fix:** Ensure your API key is set in `.env`:
```bash
# For OpenAI
OPENAI_API_KEY=sk-...

# For Anthropic
ANTHROPIC_API_KEY=sk-ant-...
```

Verify the key is loaded:
```bash
uv run python -c "from autobacktest.config import settings; print(settings.llm_model)"
```

### `RateLimitError`

**Symptom:** LLM calls fail intermittently with 429 errors.

**Fix:** Increase the request timeout in `.env`:
```bash
AUTOBACKTEST_LLM_REQUEST_TIMEOUT=900.0
```

Or reduce the number of parallel candidates:
```bash
AUTOBACKTEST_N_CANDIDATES=2
```

### `ContextWindowExceeded`

**Symptom:** LLM rejects input as too long.

**Fix:** Reduce `llm_max_tokens` or simplify the program.md constraints. The prompt includes strategy code, config, lessons, and exploration history — keep these concise.

---

## Sandbox / Preflight Failures

### `ast_blocked_import`

**Cause:** Strategy code imports a module not in the whitelist.

**Fix:** Use only allowed imports:
```
pandas, numpy, math, typing, scipy, dataclasses, collections,
itertools, functools, decimal, statistics, numbers, json
```

### `undefined_name`

**Cause:** LLM introduced a typo or referenced an out-of-scope variable.

**Fix:** Run the linter locally to catch issues:
```bash
uv run autobacktest llm-test "describe your change" --strategy <name>
```

### `ast_cyclomatic_complexity_exceeded`

**Cause:** Strategy has too many nested conditionals (default limit: 25).

**Fix:** Simplify logic, extract helper functions, or reduce branching depth.

### `smoke_test_failed`

**Cause:** Strategy crashes when executed on synthetic data.

**Fix:** Test locally:
```bash
uv run autobacktest llm-test "test prompt" --strategy <name>
```

Check the sandbox error detail in the output for the specific exception.

### `config_schema_invalid`

**Cause:** Config YAML violates `StrategyConfig` constraints (types, ranges, `extra="forbid"`).

**Fix:** Validate the YAML against the schema:
```bash
uv run python -c "
from autobacktest.strategy.config_schema import StrategyConfig
import yaml
cfg = yaml.safe_load(open('strategies/<name>/config.yaml'))
StrategyConfig(**cfg)
"
```

---

## Gate Rejections

### `target_metric_non_improving`

**Cause:** Candidate metric didn't improve over baseline.

**Fix:** The gate requires meaningful improvement. Try more aggressive strategy modifications or adjust the target metric.

### `max_drawdown_exceeded`

**Cause:** In-sample max drawdown exceeded 20% threshold.

**Fix:** Add risk management logic (stop-loss, position limits, volatility scaling).

### `turnover_exceeded`

**Cause:** Portfolio turnover exceeded 2.0 threshold.

**Fix:** Reduce rebalancing frequency or add turnover penalty in the strategy logic.

### `dsr_non_degradation_failed`

**Cause:** Deflated Sharpe Ratio degraded compared to baseline.

**Fix:** This is a hard gate — the candidate must not degrade risk-adjusted returns. Ensure improvements are statistically meaningful, not just noise.

### `diversity_config_rejected` / `diversity_returns_rejected`

**Cause:** Candidate is too similar to recently evaluated strategies.

**Fix:** The diversity gate prevents redundant evaluations. Try a meaningfully different approach (different indicators, parameters, or logic).

---

## Data Issues

### `yfinance` returns empty / NaN data

**Cause:** Ticker may be delisted or the date range has no data.

**Fix:**
- Verify tickers exist on Yahoo Finance
- Check date range is valid (not in the future)
- Enable quiet mode to suppress warnings: `AUTOBACKTEST_QUIET=true`

### Cache corruption

**Symptom:** Stale or inconsistent price data.

**Fix:** Clear the Parquet cache:
```bash
rm -rf data/cache/
```

Data will be re-downloaded on next evaluation.

---

## SQLite / Ledger Issues

### `database is locked`

**Cause:** Multiple processes accessing the ledger simultaneously.

**Fix:** Increase the SQLite timeout:
```bash
AUTOBACKTEST_DB_TIMEOUT=30.0
```

Or ensure only one optimization loop runs at a time.

### Corrupt ledger database

**Fix:** Delete the ledger and start fresh:
```bash
rm runs/ledger.db
uv run autobacktest report  # Will create a new empty ledger
```

---

## Configuration Issues

### Settings not loading from `.env`

**Fix:** Ensure `.env` is in the project root (same directory as `pyproject.toml`). The config is loaded via `python-dotenv` on module import.

### `n_candidates` test failures

**Cause:** Two tests assert `len(provider.calls) == 9` (3 candidates × 3 iterations). If `.env` sets a different `n_candidates`, these fail.

**Fix:** Run tests with the expected value:
```bash
AUTOBACKTEST_N_CANDIDATES=3 uv run pytest
```

---

## Debugging Commands

```bash
# Full evaluation with detailed metrics
uv run autobacktest evaluate --strategy <name>

# Test LLM edit without full loop
uv run autobacktest llm-test "prompt" --strategy <name>

# View leaderboard
uv run autobacktest report

# Run single test
uv run pytest tests/test_gate.py -x -k "test_name"

# Run fast tests (skip slow E2E)
uv run pytest -m "not slow"

# Check strategy AST
uv run python -c "
import ast, pathlib
code = pathlib.Path('strategies/<name>/strategy.py').read_text()
ast.parse(code)
print('AST OK')
"
```
