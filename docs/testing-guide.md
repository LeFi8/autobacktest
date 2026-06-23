# Testing Guide

How to run, write, and understand the AutoBacktest test suite.

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `uv run pytest` | Run all 447 tests |
| `uv run pytest -m "not slow"` | Skip slow E2E tests (~3-5s) |
| `uv run pytest -m "not sandbox"` | Skip sandbox subprocess tests |
| `uv run pytest -x` | Stop on first failure |
| `uv run pytest -x --cov=src/autobacktest` | With coverage report |
| `uv run pytest tests/test_gate.py -x -k "test_name"` | Single test |
| `uv run pytest -n auto` | Parallel execution via pytest-xdist |

---

## Test Markers

Defined in `pyproject.toml`:

| Marker | Description | Deselect with |
|--------|-------------|---------------|
| `slow` | Slow orchestrator E2E tests | `-m "not slow"` |
| `e2e` | End-to-end orchestrator tests | `-m "not e2e"` |
| `sandbox` | Tests that spawn subprocesses | `-m "not sandbox"` |

---

## Test Architecture

### Fixtures (`conftest.py`)

| Fixture | Scope | Purpose |
|---------|-------|---------|
| `setup_test_environment` | session (autouse) | Sets `sandbox_timeout=2`, `n_candidates=3`, disables LLM repair and directives for fast test execution |
| `synthetic_prices` | session | 12-year daily price DataFrame (2 assets, seed=42) — no network required |
| `project_root_with_lessons` | function | Temp directory with git repo, strategy files, and seeded lesson DB |
| `mock_validate_candidate_pass` | function | Monkeypatches `_validate_candidate` to bypass sandbox |

### Synthetic Price Data

All tests use `synthetic_prices` — a deterministic DataFrame generated with `np.random.default_rng(42)`:
- Date range: 2013-01-01 to 2025-01-01 (business days)
- Assets: `HIGH` (μ=0.1%, σ=0.2%) and `LOW` (μ=0.01%, σ=0.2%)
- No API keys or network access required

### Mock/Fake Providers

E2E orchestrator tests use `MockProvider` / `FakeProvider` instead of real LLM calls:
- `MockProvider` (`llm/mock_provider.py`): Returns scripted responses for deterministic testing
- `FakeProvider`: Alternative mock with different response patterns
- Both exercise the full optimization loop without LLM API costs

---

## Test Organization

```
tests/
├── conftest.py                    # Shared fixtures
├── test_gate.py                   # Two-phase gate logic
├── test_orchestrator_e2e.py       # Full optimization loop (MockProvider)
├── test_evaluator_extras.py       # Extended evaluator tests
├── test_cscv.py                   # CSCV / PBO calculation
├── test_deflated_sharpe.py        # Deflated Sharpe Ratio
├── test_monte_carlo.py            # Monte Carlo bootstrap
├── test_spa.py                    # Hansen's SPA test
├── test_characterization_*.py     # Characterization/snapshot tests
├── test_cli_*.py                  # CLI command tests
├── test_config*.py                # Config/schema tests
└── test_*.py                      # ... (56 total test files)
```

---

## Writing New Tests

### Guidelines

1. **Use synthetic data** — never make real API calls in tests
2. **Set explicit seeds** — use `np.random.default_rng(42)` for reproducibility
3. **Mock external services** — use `MockProvider` for LLM, `unittest.mock` for filesystem
4. **Keep tests fast** — the session fixture sets `sandbox_timeout=2` for a reason
5. **Use markers** — mark slow tests with `@pytest.mark.slow`

### Example: Unit Test

```python
def test_sortino_ratio_calculation(synthetic_prices):
    """Sortino ratio penalises only downside volatility."""
    from autobacktest.evaluator.metrics import calculate_sortino_ratio

    returns = synthetic_prices.pct_change().dropna()
    sharpe = calculate_sortino_ratio(returns["HIGH"])
    assert isinstance(sharpe, float)
    assert -5.0 < sharpe < 5.0  # reasonable bounds
```

### Example: E2E Test with MockProvider

```python
@pytest.mark.slow
def test_optimization_improves_metric(mock_validate_candidate_pass):
    """Full loop should improve Sharpe over baseline."""
    from autobacktest.orchestrator import run_optimization
    from autobacktest.llm.mock_provider import MockProvider

    provider = MockProvider()
    result = run_optimization(
        program_path=Path("program.md"),
        strategy_name="equal_weight",
        provider=provider,
        iterations=3,
    )
    assert result.winner_report is not None
```

---

## Coverage

Run with coverage:
```bash
uv run pytest -x --cov=src/autobacktest --cov-report=term-missing
```

---

## CI Pipeline

Defined in `.github/workflows/ci.yml`:
1. Lint (`ruff check .`)
2. Format check (`ruff format . --check`)
3. Type check (`mypy src/`)
4. Full test suite (`pytest`)
