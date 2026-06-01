# AGENTS.md — AutoBacktest

## Commands (all via `uv run`)

```bash
uv sync                          # install deps
uv run autobacktest --help       # list subcommands
uv run pytest                    # full test suite
uv run pytest tests/test_gate.py  # single file
uv run ruff check .              # lint (line-length 120, target py312)
uv run mypy src/                 # typecheck (--strict)
uv run autobacktest run --program program.md --strategy haa --iterations 5
uv run autobacktest report       # leaderboard
uv run autobacktest evaluate --strategy strategies/haa.py
```

No `.pre-commit-config.yaml` exists — skip `pre-commit install`.

## Architecture

- **Entrypoint**: `src/autobacktest/cli.py` → `autobacktest.cli:app` (typer). 5 subcommands: `run`, `report`, `reset`, `evaluate`, `llm-test`.
- **Strategy files**: `strategies/<name>.py` + `configs/<name>.yaml` — matched by stem. Strategy exports `generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame`.
- **Allowed imports** in strategy code: pandas, numpy, math, typing, scipy, dataclasses, collections, itertools, functools, decimal, statistics, numbers, json only. Blocked by AST whitelist.
- **Optimization loop** (`orchestrator.py`): LLM edits code → preflight validation (6 checks in sandboxed subprocess) → config diversity gate → evaluation (walk-forward + holdout) → returns diversity gate → lexicographic gate → git commit or rollback.
- **Gate** (`gate.py`): hard gates in order — max_drawdown ≤ 20%, regime stress tests pass, turnover ≤ 2.0, then target metric improvement over baseline. DSR is computed for insight but is NOT a hard gate.
- **Holdout**: 3 years default (`AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS`). Walk-forward: 5y train / 1y test. All statistical simulations use seed=42.
- **Config**: `.env` via `python-dotenv`. Copy `.env.dist` → `.env`. `src/autobacktest/config.py` → global `settings` singleton.

## Key files

| File | Purpose |
|---|---|
| `program.md` | LLM objective + constraints (must have `# Objective` and `# Constraints` H1) |
| `lessons.md` | LLM-curated memory (4096 token cap ~16k chars) |
| `strategies/<name>.py` | Strategy signal code |
| `configs/<name>.yaml` | Strategy parameters (Pydantic-validated) |
| `.antigravity/` | Pre-existing agent instruction files (AGENT_GUIDELINES.md, TECH_STACK.md, etc.) |

## Testing quirks

- Tests use synthetic prices — no API keys or network needed.
- `conftest.py` sets `settings.sandbox_timeout = 2` for fast test failures.
- E2E orchestrator tests use `MockProvider` / `FakeProvider` — exercises full loop without an LLM.
- VCR cassettes may exist for yfinance tests.
- ~195 tests total.

## Environment quirks

- `AUTOBACKTEST_LITELLM_DEBUG=True` enables verbose LiteLLM logging.
- `AUTOBACKTEST_LLM_REQUEST_TIMEOUT` defaults to 600s — LLM calls on large strategies can be slow.
- Cache dir is `data/cache/` (parquet files). Run dir is `runs/` (SQLite ledger + event logs).
