# AGENTS.md — AutoBacktest

## Commands (all via `uv run`)

```bash
uv sync                          # install deps
uv run autobacktest --help       # list subcommands
uv run pytest                    # full test suite
uv run pytest tests/test_gate.py -x -k "test_name"  # single test
uv run ruff check .              # lint (line-length 120, target py312)
uv run mypy src/                 # typecheck (--strict)
uv run autobacktest run --program program.md --strategy haa --iterations 5
uv run autobacktest report       # leaderboard
uv run autobacktest evaluate --strategy strategies/haa.py
uv run autobacktest llm-test "Add momentum filter" --strategy haa
uv run autobacktest init-strategy --name my_strategy
```

No `.pre-commit-config.yaml` exists — skip `pre-commit install`.

## Architecture

- **Entrypoint**: `src/autobacktest/cli.py:app` (typer). 7 subcommands: `run`, `report`, `reset`, `evaluate`, `llm-test`, `init-strategy`. `run` has `--quiet` and `--json` flags.
- **Strategy files**: `strategies/<name>.py` + `configs/<name>.yaml` — matched by stem. Strategy exports `generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame`.
- **Allowed imports** in strategy code: pandas, numpy, math, typing, scipy, dataclasses, collections, itertools, functools, decimal, statistics, numbers, json only. Blocked by AST whitelist.
- **Optimization loop** (`orchestrator.py`): LLM edits code → `repair_strategy_code()` (3 AST passes: pandas deprecation fix, `typing.Any` import injection, weight renormalization) → preflight validation (8 checks incl. undefined-name AST scan) → config diversity gate → evaluation (walk-forward + holdout) → returns diversity gate → two-phase select/confirm gate → git commit or rollback.
- **Undefined-name validator** (`_check_undefined_names` in `validator.py`): Catches LLM hallucinations (misspelled identifiers, out-of-scope variables). Handles tuple unpacking and top-level constants via `_extract_names` helper.
- **Gate** (`gate.py`): Two-phase: `select` (in-sample — max_drawdown ≤ 20%, regime stress tests pass, turnover ≤ 2.0, target metric improvement, min_return_ratio, DSR non-degradation) then `confirm` (holdout — drawdown, turnover, DSR non-degradation). DSR is a hard gate for both phases.
- **Holdout**: 3 years default (`AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS`). Walk-forward: 5y train / 1y test. All statistical simulations use seed=42.
- **Config**: `.env` via `python-dotenv`. Copy `.env.dist` → `.env`. `src/autobacktest/config.py` → global `settings` singleton. All env vars prefixed `AUTOBACKTEST_`.
- **Cache**: Parquet-backed (`data/cache/`), uses `pandas.tseries.holiday.USFederalHolidayCalendar` for US trading day detection. Suffix fetches skip weekends/holidays but must not skip the last trading day (uses `>` not `>=` in trading-day guard).

## Key files

| File | Purpose |
|---|---|
| `program.md` | LLM objective + constraints (must have `# Objective` and `# Constraints` H1) |
| `lessons.md` | LLM-curated memory (4096 token cap ~16k chars) |
| `strategies/<name>.py` | Strategy signal code |
| `configs/<name>.yaml` | Strategy parameters (Pydantic-validated) |
| `.antigravity/` | Pre-existing agent instruction files (AGENT_GUIDELINES.md, TECH_STACK.md, etc.) |

## Env var quirks

- `AUTOBACKTEST_LITELLM_DEBUG=True` enables verbose LiteLLM logging.
- `AUTOBACKTEST_LLM_REQUEST_TIMEOUT` defaults to 600s — LLM calls on large strategies can be slow.
- `AUTOBACKTEST_N_CANDIDATES` controls parallel candidate count per iteration (default 3). **Two tests expect this to be exactly 3** — if `.env` sets a different value, those tests fail with `assert len(provider.calls) == 9` or similar. Run with `AUTOBACKTEST_N_CANDIDATES=3 uv run pytest ...` to avoid.
- `AUTOBACKTEST_QUIET=true` / `--quiet` suppresses numpy all-NaN, yfinance "possibly delisted", and urllib3 warnings.
- Cache dir is `data/cache/` (parquet files). Run dir is `runs/` (SQLite ledger + event logs).

## Testing quirks

- **344 tests** total (45 files).
- Tests use synthetic prices — no API keys or network needed.
- `conftest.py` sets `settings.sandbox_timeout = 2` for fast test failures (session-scoped autouse).
- E2E orchestrator tests use `MockProvider` / `FakeProvider` — exercises full loop without an LLM.
- VCR cassettes may exist for yfinance tests.
- One pre-existing test failure: `test_count_node_lines_large` expects `max_function_lines` < 151 but the default is 100 (fails on some configs). Can be ignored.
