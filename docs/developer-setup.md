# Developer Setup & Installation Guide

> [Documentation Hub](index.md) | [About Project](about-project.md) | [Strategy Guide](strategy-guide.md) | [Architecture](architecture.md) | [API Reference](api-reference.md)

This document describes how to set up the development environment, execute the test suite, run static checks, and configure pre-commit hooks.

## Prerequisites
- **Python**: `>=3.12` installed on your machine.
- **uv**: Astral's high-performance Python package installer and resolver.
  - Install using curl (macOS/Linux):
    ```bash
    curl -LsSf https://astral.sh/uv/install.sh | sh
    ```
  - Or via Homebrew:
    ```bash
    brew install uv
    ```

## Local Installation

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/LeFi8/autobacktest.git
   cd autobacktest
   ```

2. **Sync Virtual Environment & Dependencies**:
   `uv sync` will automatically create a `.venv` directory and download all dependencies defined in `pyproject.toml` and `uv.lock`.
   ```bash
   uv sync
   ```


> **Note:** This project does not include a `.pre-commit-config.yaml`. Pre-commit hooks are not configured — all linting/type-checking is run manually via the commands below.

## Development Commands

### 1. Static Typing Verification
Strict static type checking is enforced via `mypy` with `--strict`. All code in `src/` must compile with no errors:
```bash
uv run mypy --strict src/
```

### 2. Code Quality Audits (Linting & Formatting)
We use `ruff` to enforce code quality, style guidelines, and sorted imports.

- **Check Code Quality**:
   ```bash
   uv run ruff check .
   ```
- **Apply Automatic Fixes**:
   ```bash
   uv run ruff check . --fix
   ```
- **Reformat Code Style**:
   ```bash
   uv run ruff format .
   ```

### 3. Executing Unit Tests
We use `pytest` with `hypothesis` (property-based testing) and `vcrpy` (HTTP connection recording).

- **Run all tests**:
  ```bash
  uv run pytest
  ```
- **Run tests with coverage statistics**:
  ```bash
  uv run pytest -x --cov=src/autobacktest
  ```
- **Run a single test file or test**:
  ```bash
  uv run pytest tests/test_gate.py -x -k "test_name"
  ```

### 4. Running the CLI application
Verify installation by executing the CLI helper command:
```bash
uv run autobacktest --help
```

### 5. Testing LLM Edits (Without Full Loop)
Test whether a prompt produces a valid strategy edit against preflight checks:
```bash
uv run autobacktest llm-test \
  "Add a momentum filter that only selects assets with positive 6-month return" \
  --strategy equal_weight
```

### 6. Interactive Strategy Scaffolding
Generate a new strategy with Pydantic-validated boilerplate. By default the command
runs an interactive wizard that asks for universe, benchmark, risk limits, and template:

```bash
uv run autobacktest init-strategy --name my_strategy
```

#### Silent (Non-Interactive) Mode
When ``--universe`` is provided, all prompts are skipped — unspecified values use their
defaults.  This is ideal for scripting or CI:

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

#### CLI Flags

| Flag | Default | Description |
|---|---|---|
| ``--name`` / ``-n`` | _prompts_ | Strategy name (snake\_case). Required. |
| ``--universe`` / ``-u`` | _prompts_ | Comma-separated tickers (e.g. ``SPY,TLT,GLD,BIL``). **Triggers silent mode.** |
| ``--benchmark`` / ``--bench`` | ``SPY`` | Benchmark index ticker. |
| ``--drawdown`` / ``--mdd`` | ``0.20`` | Max drawdown limit (0.0–1.0). |
| ``--turnover`` | ``2.0`` | Annualized turnover limit. |
| ``--lookback`` / ``--mom-lookback`` | ``12`` | Momentum lookback in months. |
| ``--template`` | ``equal-weight`` | Strategy template: ``equal-weight`` or ``momentum-rotation``. |
| ``--cash-asset`` | ``BIL`` | Cash/risk-free asset ticker. |
| ``--overwrite`` | ``False`` | Overwrite existing files without prompting. |

#### Strategy Templates

The ``--template`` flag selects the generated boilerplate:

- **``equal-weight``** (default): Allocates ``1/N`` of capital to each universe asset on each
  rebalance date.  Simplest starting point.
- **``momentum-rotation``**: Ranks assets by trailing return over the momentum lookback
  period, selects the top ``N`` (configurable via ``params.top_n``), and allocates equally.
  Falls back to 100 % cash when no asset has positive momentum.

#### Generated Files

Three files are created:

| File | Purpose |
|---|---|
| ``strategies/{name}/config.yaml`` | Full Pydantic-validated strategy configuration |
| ``strategies/{name}/strategy.py`` | Runnable signal-generation code (edit to implement your logic) |
| ``program-{name}.md`` | LLM objective / constraints document for optimization |

The generated code is a **starting point** — edit ``strategies/{name}/strategy.py`` to implement
your custom signal logic before running the optimization loop.

#### Run the Optimization Loop

```bash
uv run autobacktest run --program program-{name}.md --strategy {name} --iterations 5
```

### 7. Hansen's SPA Test
Audit whether optimized strategies significantly outperform the baseline after
correcting for data-snooping bias:
```bash
uv run autobacktest spa --run-id <run_id> --accepted-only
```

**Flags:**

| Flag | Default | Description |
|---|---|---|
| `--run-id` | _required_ | Run identifier to audit (from `report` output) |
| `--accepted-only` | `False` | Only include gate-accepted candidates (excludes rejected attempts) |

The SPA test returns three p-value bounds (consistent, upper conservative,
lower liberal) and the observed test statistic. A p-value < 0.05 indicates
the best candidate significantly outperforms the baseline after correcting
for data-snooping bias.

### 8. Raw JSON Output
Use the `--json` flag with the `run` command to output raw JSON instead of
the Rich dashboard. Useful for scripting or piping to other tools:
```bash
uv run autobacktest run \
  --program program.md \
  --strategy equal_weight \
  --iterations 5 \
  --json
```


