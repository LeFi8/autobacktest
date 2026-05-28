"""Tests for the CLI report subcommand."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from autobacktest.cli import app
from autobacktest.ledger.store import LedgerStore
from tests.test_ledger_store import _record

runner = CliRunner()


def test_report_no_db() -> None:
    """Report command handles missing ledger.db gracefully."""
    result = runner.invoke(app, ["report", "--run-dir", "nonexistent_dir"])
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_report_empty_db(tmp_path: Path) -> None:
    """Report command handles empty database gracefully."""
    store = LedgerStore(tmp_path / "ledger.db")
    store.close()

    result = runner.invoke(app, ["report", "--run-dir", str(tmp_path)])
    assert result.exit_code == 0
    assert "No runs found" in result.output


def test_report_with_runs(tmp_path: Path) -> None:
    """Report command prints table with seeded runs."""
    db_path = tmp_path / "ledger.db"
    store = LedgerStore(db_path)
    store.create_run(
        run_id="run-toy",
        strategy_name="toy",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-01T00:00:00",
    )
    _record(
        store,
        run_id="run-toy",
        iteration=1,
        strategy_name="toy",
        dataset_hash="hash-abc",
        observed_sharpe=1.234,
        accepted=True,
    )
    store.close()

    # Test single strategy filter
    result = runner.invoke(app, ["report", "--run-dir", str(tmp_path), "--strategy", "toy"])
    assert result.exit_code == 0
    assert "Leaderboard" in result.output
    assert "toy" in result.output
    assert "1.234" in result.output

    # Test compare-all option
    result_all = runner.invoke(app, ["report", "--run-dir", str(tmp_path), "--compare-all"])
    assert result_all.exit_code == 0
    assert "toy" in result_all.output
    assert "1.234" in result_all.output


def test_report_defaults_to_latest_run_not_all_time_best(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    store = LedgerStore(db_path)
    store.create_run(
        run_id="run-old",
        strategy_name="toy",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-01T00:00:00",
    )
    store.create_run(
        run_id="run-new",
        strategy_name="toy",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-02T00:00:00",
    )
    _record(
        store,
        run_id="run-old",
        iteration=1,
        strategy_name="toy",
        dataset_hash="hash-abc",
        observed_sharpe=2.0,
        accepted=True,
    )
    _record(
        store,
        run_id="run-new",
        iteration=1,
        strategy_name="toy",
        dataset_hash="hash-abc",
        observed_sharpe=0.5,
        accepted=True,
    )
    store.close()

    result = runner.invoke(app, ["report", "--run-dir", str(tmp_path)])

    assert result.exit_code == 0
    assert "run-new" in result.output
    assert "0.500" in result.output
    assert "run-old" not in result.output
    assert "2.000" not in result.output


def test_report_run_id_selects_non_latest_run(tmp_path: Path) -> None:
    db_path = tmp_path / "ledger.db"
    store = LedgerStore(db_path)
    store.create_run(
        run_id="run-old",
        strategy_name="toy",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-01T00:00:00",
    )
    store.create_run(
        run_id="run-new",
        strategy_name="toy",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-02T00:00:00",
    )
    _record(
        store,
        run_id="run-old",
        iteration=1,
        strategy_name="toy",
        dataset_hash="hash-abc",
        observed_sharpe=2.0,
        accepted=True,
    )
    _record(
        store,
        run_id="run-new",
        iteration=1,
        strategy_name="toy",
        dataset_hash="hash-abc",
        observed_sharpe=0.5,
        accepted=True,
    )
    store.close()

    result = runner.invoke(app, ["report", "--run-dir", str(tmp_path), "--run-id", "run-old"])

    assert result.exit_code == 0
    assert "run-old" in result.output
    assert "2.000" in result.output
    assert "run-new" not in result.output
    assert "0.500" not in result.output


def test_report_compare_all_only_includes_selected_run_strategies(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "ledger.db"
    store = LedgerStore(db_path)
    store.create_run(
        run_id="run-selected",
        strategy_name="toy",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-01T00:00:00",
    )
    store.create_run(
        run_id="run-other",
        strategy_name="outsider",
        program_path="/tmp/p.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=5,
        started_at="2026-01-02T00:00:00",
    )
    _record(store, run_id="run-selected", strategy_name="toy", observed_sharpe=1.0)
    _record(store, run_id="run-selected", strategy_name="alt", observed_sharpe=1.5)
    _record(store, run_id="run-other", strategy_name="outsider", observed_sharpe=9.9)
    store.close()

    result = runner.invoke(
        app,
        [
            "report",
            "--run-dir",
            str(tmp_path),
            "--run-id",
            "run-selected",
            "--compare-all",
        ],
    )

    assert result.exit_code == 0
    assert "toy" in result.output
    assert "alt" in result.output
    assert "outsider" not in result.output
    assert "9.900" not in result.output
