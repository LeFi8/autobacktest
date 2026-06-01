# Contributing to AutoBacktest

Welcome! Thank you for contributing to AutoBacktest. Please follow these guidelines for local development.

## Setup Instructions

This project uses `uv` for Python virtual environment and dependency management.

1. **Install `uv`**:
   Follow instructions at https://github.com/astral-sh/uv to install `uv` on your system.

2. **Clone the repository and sync dependencies**:
   ```bash
   uv sync
   ```

## Development Guidelines

- **Strict Type Checking**: We use `mypy` with `--strict` verification. All source code in `src/` must pass strictly.
- **Linting and Formatting**: We use `ruff` for code linting and formatting. Run `uv run ruff check .` and `uv run ruff format .` before committing.
- **Testing**: We use `pytest` with `hypothesis` and `vcrpy`. Ensure coverage targets are met (target >= 85% on core modules). Run tests with:
   ```bash
   uv run pytest
   ```
