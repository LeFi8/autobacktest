# AGENTS.md — AutoBacktest

## Commands (all via `uv run`)

```bash
uv sync                          # install deps
uv run autobacktest --help       # list subcommands
uv run pytest                    # full test suite (all 447 tests)
uv run pytest -m "not slow"      # fast feedback (skip slow E2E tests, ~3-5s)
uv run pytest -m "not sandbox"   # skip sandbox subprocess tests
uv run pytest -x --cov=src/autobacktest  # with coverage
uv run pytest tests/test_gate.py -x -k "test_name"  # single test
uv run pytest -n auto            # parallel execution via pytest-xdist
uv run ruff check .              # lint (line-length 120, target py312)
uv run ruff format . --check     # formatter check
uv run mypy src/                 # typecheck (--strict)
uv run autobacktest run --program program.md --strategy equal_weight --iterations 5
uv run autobacktest report       # leaderboard
uv run autobacktest evaluate --strategy equal_weight
uv run autobacktest spa          # Hansen's SPA test
uv run autobacktest llm-test "Add momentum filter" --strategy equal_weight
uv run autobacktest init-strategy --name my_strategy
```

No `.pre-commit-config.yaml` exists — skip `pre-commit install`.

## Architecture

- **Entrypoint**: `src/autobacktest/cli.py:app` (typer) delegating to `commands/` package. 7 subcommands: `run`, `report`, `reset`, `evaluate`, `spa`, `llm-test`, `init-strategy`. `run` has `--quiet` and `--json` flags.
- **Strategy files**: `strategies/<name>/strategy.py` + `strategies/<name>/config.yaml` (subdirectory layout). Legacy flat layout `strategies/<name>.py` + `configs/<name>.yaml` still supported as fallback. Strategy exports `generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame`.
- **Allowed imports** in strategy code: pandas, numpy, math, typing, scipy, dataclasses, collections, itertools, functools, decimal, statistics, numbers, json only. Blocked by AST whitelist.
- **Optimization loop** (`orchestrator.py` with `optimization/` sub-package): LLM edits code → `repair_strategy_code()` (3 AST passes: pandas deprecation fix, `typing.Any` import injection, weight renormalization) → preflight validation (8 checks incl. AST linter in `ast_linter.py`, sandboxed smoke test in `sandbox_runner.py`) → config diversity gate (with optional jitter salvage via `config_jitter.py`) → evaluation (walk-forward + holdout) → returns diversity gate → two-phase select/confirm gate → git commit or rollback. Config jitter and LLM repair loop are optional salvages when candidates fail diversity or preflight.
- **Undefined-name validator** (`_check_undefined_names` in `ast_linter.py`): Catches LLM hallucinations (misspelled identifiers, out-of-scope variables). Handles tuple unpacking and top-level constants via `_extract_names` helper.
- **PBO via CSCV** (`evaluator/cscv.py`): Combinatorially Symmetric Cross-Validation calculates Probability of Backtest Overfitting. Default 10 blocks → 252 train/test split combinations. Stored in `EvaluationReport.pbo`.
- **Gate** (`gate.py`): Two-phase: `select` (in-sample — max_drawdown ≤ 20%, regime stress tests pass, turnover ≤ 2.0, target metric improvement, min_return_ratio, DSR non-degradation) then `confirm` (holdout — drawdown, turnover, DSR non-degradation). DSR is a hard gate for both phases.
- **Holdout**: 3 years default (`AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS`). Walk-forward: 5y train / 1y test. All statistical simulations use seed=42.
- **Config**: `.env` via `python-dotenv`. Copy `.env.dist` → `.env`. `src/autobacktest/config.py` → global `settings` singleton. All env vars prefixed `AUTOBACKTEST_`.
- **Cache**: Parquet-backed (`data/cache/`). Run dir is `runs/` (SQLite ledger + event logs). All statistical simulations use seed=42.
- **Ledger Comparability**: Changing `adaptive_slippage` settings (e.g. enabling it or changing its caps) invalidates comparability of historical metrics stored in the SQLite ledger. Start a fresh ledger (`ledger.db`) if changing these configurations to ensure consistent comparison baselines.

## Key files

| File | Purpose |
|---|---|
| `program.md` | LLM objective + constraints (must have `# Objective` and `# Constraints` H1) |
| `runs/lessons.db` | SQLite-backed LLM lesson store (replaces flat `lessons.md`) |
| `strategies/<name>/strategy.py` | Strategy signal code |
| `strategies/<name>/config.yaml` | Strategy parameters (Pydantic-validated) |
| `docs/` | Comprehensive docs: architecture, API reference, setup |
| `src/autobacktest/optimization/` | Candidate gen, eval mgmt, ledger persistence helpers |
| `src/autobacktest/commands/` | CLI subcommand implementations (7 subcommands, 8 files) |
| `src/autobacktest/strategy/ast_linter.py` | AST-based static validation (imports, complexity, undefined names) |
| `src/autobacktest/strategy/sandbox_runner.py` | Sandboxed subprocess execution for smoke tests |
| `src/autobacktest/evaluator/engine.py` | Vectorized window execution, caching, dataset hashing |
| `src/autobacktest/evaluator/metrics.py` | Sortino, Information Ratio, walk-forward aggregation |
| `src/autobacktest/evaluator/stress_testing.py` | Regime stress + Monte Carlo bootstrap wrapper |

## Env var quirks

- `AUTOBACKTEST_LITELLM_DEBUG=True` enables verbose LiteLLM logging.
- `AUTOBACKTEST_LLM_REQUEST_TIMEOUT` defaults to 600s — LLM calls on large strategies can be slow.
- `AUTOBACKTEST_N_CANDIDATES` controls parallel candidate count per iteration (code default 10, `.env.dist` sets to 3). **Two tests expect this to be exactly 3** — if `.env` sets a different value, those tests fail with `assert len(provider.calls) == 9` or similar. Run with `AUTOBACKTEST_N_CANDIDATES=3 uv run pytest ...` to avoid.
- `AUTOBACKTEST_QUIET=true` / `--quiet` suppresses numpy all-NaN, yfinance "possibly delisted", and urllib3 warnings.
- `AUTOBACKTEST_MAX_CYCLOMATIC_COMPLEXITY` defaults to **25** in code (not 20).
- `AUTOBACKTEST_ENABLE_LLM_REPAIR`, `AUTOBACKTEST_ENABLE_CONFIG_JITTER`, `AUTOBACKTEST_ENABLE_JSON_SALVAGE` are all `true` by default — these are salvage/retry mechanisms for failed candidates.

## Testing quirks

- **447 tests** total (54 files).
- Tests use synthetic prices — no API keys or network needed.
- `conftest.py` sets `settings.sandbox_timeout = 2` for fast test failures (session-scoped autouse).
- E2E orchestrator tests use `MockProvider` / `FakeProvider` — exercises full loop without an LLM.
- VCR cassettes may exist for yfinance tests.
- One pre-existing test failure: `test_count_node_lines_large` expects `max_function_lines` < 151 but the default is 100 (fails on some configs). Can be ignored.
