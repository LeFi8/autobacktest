"""Tests for cost tracking, token-usage persistence, migrations, and CLI reporting."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

from autobacktest.cli import app
from autobacktest.ledger.store import LedgerStore
from tests.test_ledger_store import _make_returns


def test_schema_migration_backward_compatibility(tmp_path: Path) -> None:
    """Verifies that an older ledger database is auto-migrated with new cost tracking columns."""
    db_file = tmp_path / "old_ledger.db"

    # 1. Create a database using the legacy schema lacking token/cost columns
    conn = sqlite3.connect(str(db_file))
    conn.execute(
        """
        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            strategy_name TEXT NOT NULL,
            program_path TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            branch TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            iterations INTEGER NOT NULL,
            started_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id TEXT NOT NULL,
            iteration INTEGER NOT NULL,
            strategy_name TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            config_yaml TEXT NOT NULL,
            observed_sharpe REAL NOT NULL,
            deflated_sharpe REAL NOT NULL,
            target_metric TEXT NOT NULL,
            target_metric_value REAL NOT NULL,
            holdout_max_drawdown REAL NOT NULL,
            holdout_turnover REAL NOT NULL,
            regime_passed INTEGER NOT NULL,
            accepted INTEGER NOT NULL,
            committed INTEGER NOT NULL,
            commit_sha TEXT,
            rejection_reason TEXT,
            report_json TEXT NOT NULL,
            returns_blob BLOB NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    conn.close()

    # 2. Instantiate LedgerStore to trigger migration logic
    store = LedgerStore(db_file)
    store.close()

    # 3. Verify columns exist now
    conn = sqlite3.connect(str(db_file))
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(attempts)")
    columns = [row[1] for row in cursor.fetchall()]
    conn.close()

    assert "prompt_tokens" in columns
    assert "completion_tokens" in columns
    assert "total_tokens" in columns
    assert "cost" in columns


def test_record_attempt_cost_round_trip(tmp_path: Path) -> None:
    """Verifies that record_attempt successfully persists cost and token metrics."""
    db_file = tmp_path / "ledger.db"
    store = LedgerStore(db_file)

    store.create_run(
        run_id="run-cost-test",
        strategy_name="haa",
        program_path="program.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-123",
        iterations=5,
        started_at="2026-05-28T00:00:00",
    )

    returns = _make_returns(seed=1, n=10)

    # Record first attempt with specific cost metrics
    store.record_attempt(
        run_id="run-cost-test",
        iteration=1,
        strategy_name="haa",
        dataset_hash="hash-123",
        config_yaml="momentum_lookback: 12",
        observed_sharpe=1.5,
        deflated_sharpe=1.4,
        target_metric="sharpe",
        target_metric_value=1.5,
        holdout_max_drawdown=0.08,
        holdout_turnover=0.25,
        regime_passed=True,
        accepted=True,
        committed=False,
        commit_sha=None,
        rejection_reason=None,
        report_json="{}",
        holdout_returns=returns,
        prompt_tokens=1500,
        completion_tokens=500,
        total_tokens=2000,
        cost=0.035,
    )

    attempts = store.attempts_for_run("run-cost-test")
    store.close()

    assert len(attempts) == 1
    attempt = attempts[0]
    assert attempt["prompt_tokens"] == 1500
    assert attempt["completion_tokens"] == 500
    assert attempt["total_tokens"] == 2000
    assert attempt["cost"] == pytest.approx(0.035)


def test_cli_report_displays_cost_summary(tmp_path: Path) -> None:
    """Verifies that `autobacktest report` displays correct cumulative tokens and cost summary."""
    db_file = tmp_path / "ledger.db"
    store = LedgerStore(db_file)

    store.create_run(
        run_id="run-cost-cli",
        strategy_name="haa",
        program_path="program.md",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-123",
        iterations=2,
        started_at="2026-05-28T00:00:00",
    )

    # Seed two attempts with different costs
    returns = _make_returns(seed=1, n=10)
    store.record_attempt(
        run_id="run-cost-cli",
        iteration=1,
        strategy_name="haa",
        dataset_hash="hash-123",
        config_yaml="momentum_lookback: 12",
        observed_sharpe=1.5,
        deflated_sharpe=1.4,
        target_metric="sharpe",
        target_metric_value=1.5,
        holdout_max_drawdown=0.08,
        holdout_turnover=0.25,
        regime_passed=True,
        accepted=True,
        committed=False,
        commit_sha=None,
        rejection_reason=None,
        report_json="{}",
        holdout_returns=returns,
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        cost=0.015,
    )

    store.record_attempt(
        run_id="run-cost-cli",
        iteration=2,
        strategy_name="haa",
        dataset_hash="hash-123",
        config_yaml="momentum_lookback: 12",
        observed_sharpe=1.8,
        deflated_sharpe=1.6,
        target_metric="sharpe",
        target_metric_value=1.8,
        holdout_max_drawdown=0.07,
        holdout_turnover=0.22,
        regime_passed=True,
        accepted=True,
        committed=False,
        commit_sha=None,
        rejection_reason=None,
        report_json="{}",
        holdout_returns=returns,
        prompt_tokens=2000,
        completion_tokens=400,
        total_tokens=2400,
        cost=0.030,
    )
    store.close()

    runner = CliRunner()
    result = runner.invoke(app, ["report", "--run-dir", str(tmp_path), "--run-id", "run-cost-cli"])

    assert result.exit_code == 0
    assert "Total Run Optimization Cost" in result.output
    # Cumulative prompt: 3000, completion: 600, total: 3600, cost: 0.0450
    assert "$0.0450" in result.output
    assert "3,000" in result.output
    assert "600" in result.output
    assert "3,600" in result.output
