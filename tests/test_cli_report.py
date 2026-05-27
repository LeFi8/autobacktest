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
    result = runner.invoke(
        app, ["report", "--run-dir", str(tmp_path), "--strategy", "toy"]
    )
    assert result.exit_code == 0
    assert "Leaderboard" in result.output
    assert "toy" in result.output
    assert "1.234" in result.output

    # Test compare-all option
    result_all = runner.invoke(
        app, ["report", "--run-dir", str(tmp_path), "--compare-all"]
    )
    assert result_all.exit_code == 0
    assert "toy" in result_all.output
    assert "1.234" in result_all.output
