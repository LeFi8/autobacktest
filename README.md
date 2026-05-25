# AutoBacktest

AutoBacktest is an autonomous AI-driven strategy optimization loop. An LLM agent edits a structured strategy definition; a deterministic evaluator scores it against rigorous validation criteria; a git-backed ledger commits improvements or rolls back regressions.

## Repository Structure

- `src/autobacktest/`: Core library package
  - `cli.py`: Main entry point for the command-line interface
  - `orchestrator.py`: Top-level optimization loop
  - `gate.py`: Dictates the strategy improvement gate rules
  - `llm/`: LLM drivers (LiteLLM)
  - `data/`: Market data fetching and Parquet caching
  - `strategy/`: Strategy schemas, validators, and AST parsers
  - `evaluator/`: Deterministic backtesting engine, costs, regimes, and bootstrapping
  - `ledger/`: Git commits and SQLite-backed tracking ledger
- `strategies/`: Folder containing dynamic strategy files optimized by the agent
- `configs/`: Folder containing strategy parameter configurations
- `tests/`: Project test suite

## Quickstart

### Prerequisites
- Python 3.12+
- `uv` package manager

### Installation
1. Sync dependencies and create virtual environment:
   ```bash
   uv sync
   ```

2. Run the CLI tool:
   ```bash
   uv run autobacktest --help
   ```

### Execution subcommands
- **Run optimization loop**:
  ```bash
  uv run autobacktest run --strategy haa --iterations 10
  ```
- **Report leaderboard**:
  ```bash
  uv run autobacktest report
  ```
- **Reset repository state**:
  ```bash
  uv run autobacktest reset
  ```

## Development and Testing
See [CONTRIBUTING.md](CONTRIBUTING.md) for testing guidelines and pre-commit setup instructions.
