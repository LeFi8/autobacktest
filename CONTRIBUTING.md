# Contributing to AutoBacktest

Welcome! Thank you for contributing to AutoBacktest. Please follow these guidelines for local development.

> **New to the project?** See [docs/developer-setup.md](docs/developer-setup.md) for detailed setup instructions, or [docs/index.md](docs/index.md) for the full documentation hub.

## Setup Instructions

This project uses `uv` for Python virtual environment and dependency management.

1. **Install `uv`**:
   Follow instructions at https://github.com/astral-sh/uv to install `uv` on your system.

2. **Clone the repository and sync dependencies**:
   ```bash
   git clone https://github.com/LeFi8/autobacktest.git
   cd autobacktest
   uv sync
   cp .env.dist .env   # Configure your LLM API key
   ```

## Development Workflow

### Running the Optimization Loop
```bash
uv run autobacktest run --program program.md --strategy equal_weight --iterations 5
```

### Running the Full Test Suite
```bash
uv run pytest                    # all tests
uv run pytest -x --cov=src/autobacktest  # with coverage
uv run pytest tests/test_gate.py -x -k "test_name"  # single test
```

### Static Analysis
```bash
uv run ruff check .              # linting (line-length 120, target py312)
uv run ruff format . --check    # formatter check
uv run mypy src/                 # strict type checking
```

### Pre-commit Checks (Manual)
There is no `.pre-commit-config.yaml` — run these manually before committing:
```bash
uv run ruff check . && uv run ruff format . && uv run mypy src/ && uv run pytest -x
```

## Code Style

- **Line length**: 120 characters max
- **Target Python**: 3.12+
- **Type annotations**: Required on all public functions (mypy `--strict`)
- **Import sorting**: Enforced by `ruff` isort
- **Docstrings**: All public functions must have docstrings with `Args:` and `Returns:` sections

## Testing Guidelines

- Tests use **synthetic prices** — no API keys or network access required
- `conftest.py` sets `settings.sandbox_timeout = 2` for fast test failures
- E2E orchestrator tests use `MockProvider` / `FakeProvider`
- Property-based testing via `hypothesis` is used for edge case coverage

## Commit Guidelines

- Write clear, concise commit messages
- Run the full test suite before committing
- Ensure `ruff check .` and `mypy src/` pass with zero errors

## Project Structure

```
autobacktest/
├── strategies/         # Signal code (<name>.py + <name>.yaml)
├── configs/            # Parameters per strategy
├── src/autobacktest/   # Core engine
│   ├── cli.py          # Typer entrypoint
│   ├── commands/       # Subcommand implementations
│   ├── orchestrator.py # Optimization loop orchestration
│   ├── gate.py         # Two-phase gate system
│   ├── evaluator/      # Backtest, metrics, stress testing
│   ├── strategy/       # Validator, AST linter, codemod
│   ├── llm/            # LLM provider abstraction
│   ├── ledger/         # SQLite ledger + git operations
│   └── lessons/        # SQLite-backed lesson store
├── docs/               # Architecture, API reference, setup guides
└── tests/              # Test suite (395+ tests)
```
