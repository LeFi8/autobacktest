# AutoBacktest

AutoBacktest is an autonomous, AI-driven quantitative trading strategy optimization system. It connects large language model (LLM) agents with deterministic backtesting and statistical evaluation pipelines to iteratively refine and validate quant trading strategies without human intervention.

An LLM agent edits a structured strategy definition; a deterministic evaluator scores it against rigorous validation criteria; a git-backed ledger commits improvements or rolls back regressions.

---

## 📖 Table of Contents
- [Project Overview](docs/about-project.md) — Core business goals, target user personas, and primary system workflows.
- [System Architecture](docs/architecture.md) — Module designs, layout topography, and Mermaid component relationship diagrams.
- [Developer Setup & Installation](docs/developer-setup.md) — Virtual environment sync (`uv`), Ruff linting/formatting, MyPy strict validation, and test commands (`pytest`).
- [API Reference Manual](docs/api-reference.md) — Complete parameter specifications and typed signatures for evaluation and caching components.

---

## 🛠️ Technical Stack
- **Language**: Python `>=3.12`
- **Dependency Management**: `uv` package manager
- **Libraries**:
  - `pandas` & `numpy` (Fast vectorized computations)
  - `scipy` (PSR and Deflated Sharpe Ratio statistics)
  - `litellm` (Unified LLM interaction provider)
  - `yfinance` & `pyarrow` (Yahoo Finance downloads with Parquet cache support)
- **Code Standards**:
  - `mypy` (Strict type safety checks)
  - `ruff` (Line length 88 formatting & lint auditing)
  - `pytest` with `hypothesis` & `vcrpy` (Generative property and API-mocked testing)

---

## 🚀 Quickstart Guide

### 1. Installation & Environment Configuration
Ensure you have Python 3.12+ and `uv` installed. Sync dependencies and setup the virtual environment:
```bash
uv sync
```

Install pre-commit hooks to automate quality checks:
```bash
uv run pre-commit install
```

### 2. Execution Subcommands
The CLI tool provides command-line triggers:

- **Run Optimization Loop**:
  ```bash
  uv run autobacktest run --strategy haa --iterations 10
  ```
- **Evaluate Strategy Standalone**:
  ```bash
  uv run autobacktest evaluate --strategy strategies/haa.py
  ```
- **Leaderboard Performance Report**:
  ```bash
  uv run autobacktest report
  ```
- **Reset Repository and Code State**:
  ```bash
  uv run autobacktest reset
  ```

---

## 🧪 Testing & Verification
Execute the test suite to confirm complete module coverage and regression protection:
```bash
uv run pytest
```

Audit typing and quality standards:
```bash
uv run ruff check .
uv run mypy --strict src/
```

Generated: 2026-05-25
