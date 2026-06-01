# Developer Setup & Installation Guide

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
   git clone <repository_url>
   cd autobacktest
   ```

2. **Sync Virtual Environment & Dependencies**:
   `uv sync` will automatically create a `.venv` directory and download all dependencies defined in `pyproject.toml` and `uv.lock`.
   ```bash
   uv sync
   ```


## Development Commands

### 1. Code Quality Audits (Linting & Formatting)
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

### 2. Static Typing Verification
We enforce strict static type checking via `mypy`. All code in `src/` must compile with no errors:
```bash
uv run mypy --strict src/
```

### 3. Executing Unit Tests
We use `pytest` with `hypothesis` (property-based testing) and `vcrpy` (HTTP connection recording).

- **Run all tests**:
  ```bash
  uv run pytest
  ```
- **Run tests with coverage statistics**:
  ```bash
  uv run pytest --cov=src/autobacktest
  ```

### 4. Running the CLI application
Verify installation by executing the CLI helper command:
```bash
uv run autobacktest --help
```


