"""Unit tests for Hansen's Superior Predictive Ability (SPA) test."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autobacktest.evaluator.spa import calculate_hansen_spa


def _make_returns(seed: int = 0, n: int = 252, mean: float = 0.0) -> pd.Series:
    """Helper to generate a daily return series."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(mean, 0.01, size=n), index=dates, dtype=float)


def test_calculate_hansen_spa_validation_errors() -> None:
    """calculate_hansen_spa raises ValueError on invalid inputs."""
    bench = _make_returns(seed=0, n=100)
    alts = pd.DataFrame({"alt1": _make_returns(seed=1, n=100)})

    # Empty alternatives
    with pytest.raises(ValueError, match="No alternative returns provided"):
        calculate_hansen_spa(bench, pd.DataFrame())

    # Empty benchmark
    with pytest.raises(ValueError, match="Benchmark returns are empty"):
        calculate_hansen_spa(pd.Series(dtype=float), alts)

    # Too few days (less than 2)
    bench_short = _make_returns(seed=0, n=1)
    alts_short = pd.DataFrame({"alt1": _make_returns(seed=1, n=1)})
    with pytest.raises(ValueError, match="Insufficient aligned trading days"):
        calculate_hansen_spa(bench_short, alts_short)

    # Disjoint dates
    dates1 = pd.date_range("2024-01-01", periods=10)
    dates2 = pd.date_range("2024-02-01", periods=10)
    bench_disjoint = pd.Series(0.01, index=dates1)
    alts_disjoint = pd.DataFrame({"alt": pd.Series(0.01, index=dates2)})
    with pytest.raises(ValueError, match="Insufficient aligned trading days"):
        calculate_hansen_spa(bench_disjoint, alts_disjoint)


def test_calculate_hansen_spa_identical_returns() -> None:
    """When alternatives are identical to the benchmark, p-values are high and t_spa is 0."""
    bench = _make_returns(seed=42, n=100)
    # Alternative is identical to the benchmark
    alts = pd.DataFrame({"alt": bench})

    results = calculate_hansen_spa(bench, alts, n_paths=100, seed=123)

    assert results["t_spa"] == 0.0
    # Since all differences are exactly zero, all bootstrap statistics are 0.0,
    # so t_boot >= t_spa is always true, yielding p-values of 1.0.
    assert results["p_consistent"] == 1.0
    assert results["p_upper"] == 1.0
    assert results["p_lower"] == 1.0


def test_calculate_hansen_spa_bounds_relationship() -> None:
    """Hansen SPA p-values satisfy: p_lower <= p_consistent <= p_upper."""
    # Create a scenario where one alternative outperforms the benchmark
    bench = _make_returns(seed=10, n=200, mean=0.0)
    alts = pd.DataFrame(
        {
            "alt_poor": _make_returns(seed=11, n=200, mean=-0.002),
            "alt_good": _make_returns(seed=12, n=200, mean=0.001),
            "alt_great": _make_returns(seed=13, n=200, mean=0.003),
        }
    )

    results = calculate_hansen_spa(bench, alts, n_paths=500, seed=42)

    p_l = results["p_lower"]
    p_c = results["p_consistent"]
    p_u = results["p_upper"]

    assert 0.0 <= p_l <= 1.0
    assert 0.0 <= p_c <= 1.0
    assert 0.0 <= p_u <= 1.0

    # Verify the mathematical bounds
    assert p_l <= p_c
    assert p_c <= p_u

    # Observed t_spa should be positive because alt_great has a positive mean difference
    assert results["t_spa"] > 0.0


def test_calculate_hansen_spa_large_outperformance() -> None:
    """If alternatives significantly outperform benchmark, p-values should be low."""
    # Benchmark has low mean, alt has high mean
    bench = _make_returns(seed=20, n=200, mean=0.0)
    alts = pd.DataFrame(
        {
            "alt": _make_returns(seed=21, n=200, mean=0.05)  # huge daily mean difference (5% daily)
        }
    )

    results = calculate_hansen_spa(bench, alts, n_paths=500, seed=42)

    # Significant outperformance should result in a very low consistent p-value
    assert results["p_consistent"] < 0.05
    assert results["p_upper"] < 0.05


def test_cli_spa_no_db() -> None:
    """SPA command handles missing ledger.db by exiting with code 1."""
    from typer.testing import CliRunner

    from autobacktest.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["spa", "--run-dir", "nonexistent_dir"])
    assert result.exit_code == 1
    assert "Ledger store database not found" in result.output


def test_cli_spa_empty_db(tmp_path: Path) -> None:
    """SPA command handles empty database by exiting with code 1."""
    from typer.testing import CliRunner

    from autobacktest.cli import app
    from autobacktest.ledger.store import LedgerStore

    store = LedgerStore(tmp_path / "ledger.db")
    store.close()

    runner = CliRunner()
    result = runner.invoke(app, ["spa", "--run-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "No runs found" in result.output


def test_cli_spa_no_alternatives(tmp_path: Path) -> None:
    """SPA command warns when there are no alternatives to evaluate."""
    from typer.testing import CliRunner

    from autobacktest.cli import app
    from autobacktest.ledger.store import LedgerStore
    from tests.test_ledger_store import _record

    db_path = tmp_path / "ledger.db"
    store = LedgerStore(db_path)
    store.create_run(
        run_id="run-only-bench",
        strategy_name="haa",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-01T00:00:00",
    )
    _record(store, run_id="run-only-bench", iteration=0, strategy_name="haa", dataset_hash="hash-abc")
    store.close()

    runner = CliRunner()
    result = runner.invoke(app, ["spa", "--run-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "Warning: No alternative candidate strategies found" in result.output


def test_cli_spa_success(tmp_path: Path) -> None:
    """SPA command runs successfully and prints rich output."""
    from typer.testing import CliRunner

    from autobacktest.cli import app
    from autobacktest.ledger.store import LedgerStore
    from tests.test_ledger_store import _record

    db_path = tmp_path / "ledger.db"
    store = LedgerStore(db_path)
    store.create_run(
        run_id="run-full",
        strategy_name="haa",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-01T00:00:00",
    )
    # 0 = bench, 1 = alt1, 2 = alt2
    bench = _make_returns(seed=0)
    alt1 = _make_returns(seed=1)
    alt2 = _make_returns(seed=2)

    _record(store, run_id="run-full", iteration=0, strategy_name="haa", dataset_hash="hash-abc", returns=bench)
    _record(
        store, run_id="run-full", iteration=1, strategy_name="haa", dataset_hash="hash-abc", returns=alt1, accepted=True
    )
    _record(
        store,
        run_id="run-full",
        iteration=2,
        strategy_name="haa",
        dataset_hash="hash-abc",
        returns=alt2,
        accepted=False,
    )
    store.close()

    runner = CliRunner()

    # Test all alternatives (default)
    result = runner.invoke(app, ["spa", "--run-dir", str(tmp_path), "--paths", "100"])
    assert result.exit_code == 0
    assert "Hansen's Superior Predictive Ability" in result.output
    assert "Consistent P-value" in result.output
    assert "VERDICT" in result.output

    # Test accepted only
    result_acc = runner.invoke(app, ["spa", "--run-dir", str(tmp_path), "--accepted-only", "--paths", "100"])
    assert result_acc.exit_code == 0
    assert "Hansen's Superior Predictive Ability" in result_acc.output
