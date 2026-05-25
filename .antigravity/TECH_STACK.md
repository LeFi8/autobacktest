# Technical Stack: AutoBacktest

## Core Language
- **Python**: `>=3.12` (features: PEP 695 generic syntax, type parameter syntax, structural pattern matching)

## Primary Dependencies
- **Data & Computation**:
  - `pandas >=2.2.0` (DataFrames, resampling, indexing)
  - `numpy >=2.0.0` (Array manipulations, stats computation)
  - `scipy >=1.12.0` (Scientific/statistical distributions for PSR/DSR)
  - `pyarrow >=15.0.0` (Storage back-end for cached Parquet price files)
  - `pandas-market-calendars >=4.4.0` (Exchange schedule tracking)
- **AI & Integrations**:
  - `litellm >=1.35.0` (Unified interface for LLM code/parameter mutation)
  - `gitpython >=3.1.40` (Git index interactions, branch/commit history state)
- **CLI & Formatting**:
  - `typer >=0.12.0` (Console execution engine)
  - `rich >=13.7.0` (Formatted table printing, execution UI)
  - `structlog >=24.1.0` (JSON-formatted, high-performance structured logs)
- **Serialization & Environment**:
  - `pydantic >=2.0` (Data validation, schemas, and configurations)
  - `pyyaml >=6.0` (Configuration YAML parsing)
  - `python-dotenv >=1.0.1` (System secrets and API key handling)
  - `joblib >=1.4.0` (Threaded/process concurrency for bootsrap and walk-forward evaluations)

## Development and Verification Tools
- **Test Runner**: `pytest >=8.0.0` (with `pytest-cov >=4.1.0` for target coverage >=85%)
- **Robust testing**:
  - `hypothesis >=6.98.0` (Property-based generative test cases)
  - `vcrpy >=6.0.0` (HTTP response capture / caching)
- **Linter & Formatter**: `ruff >=0.3.0` (combining flake8, black, isort rules; line length 88)
- **Static Type Checking**: `mypy >=1.9.0` (strict mode enabled across project context)

## Infrastructure and Persistence
- **Database**: SQLite (built-in driver) for relational leaderboard run tracking.
- **Cache Store**: Parquet local system storage (`data/cache/` by default).

Generated: 2026-05-25
