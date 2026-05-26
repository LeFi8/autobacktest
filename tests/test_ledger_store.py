"""Tests for LedgerStore — SQLite persistence of optimization attempts."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from autobacktest.ledger.store import LedgerStore

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_returns(seed: int = 0, n: int = 50) -> pd.Series:  # type: ignore[type-arg]
    """Return a synthetic daily-return Series with a DatetimeIndex."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.Series(rng.normal(0.001, 0.01, size=n), index=dates, dtype=float)


def _record(
    store: LedgerStore,
    run_id: str = "run-1",
    iteration: int = 1,
    strategy_name: str = "strat_a",
    dataset_hash: str = "hash-abc",
    observed_sharpe: float = 1.2,
    accepted: bool = True,
    returns: pd.Series | None = None,  # type: ignore[type-arg]
) -> None:
    """Convenience wrapper to record a single attempt."""
    if returns is None:
        returns = _make_returns()
    store.record_attempt(
        run_id=run_id,
        iteration=iteration,
        strategy_name=strategy_name,
        dataset_hash=dataset_hash,
        config_yaml="param: 1",
        observed_sharpe=observed_sharpe,
        deflated_sharpe=observed_sharpe - 0.1,
        target_metric="sharpe",
        target_metric_value=observed_sharpe,
        holdout_max_drawdown=0.05,
        holdout_turnover=0.3,
        regime_passed=True,
        accepted=accepted,
        committed=False,
        commit_sha=None,
        rejection_reason=None,
        report_json="{}",
        holdout_returns=returns,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_create_run_and_retrieve(tmp_path: Path) -> None:
    """create_run stores a row that can be read back via raw sqlite3."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)
    store.create_run(
        run_id="run-1",
        strategy_name="strat_a",
        program_path="/tmp/strat.py",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=10,
        started_at="2024-01-01T00:00:00",
    )
    store.close()

    conn = sqlite3.connect(str(db))
    rows = conn.execute("SELECT * FROM runs WHERE run_id = 'run-1'").fetchall()
    conn.close()

    assert len(rows) == 1
    row = rows[0]
    assert row[0] == "run-1"
    assert row[1] == "strat_a"
    assert row[7] == 10
    assert row[8] == "2024-01-01T00:00:00"


def test_record_attempt_and_round_trip(tmp_path: Path) -> None:
    """record_attempt persists returns; fetch_historical_returns reconstructs them."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)
    original = _make_returns(seed=42, n=50)
    _record(store, observed_sharpe=1.5, returns=original)

    df, sharpes = store.fetch_historical_returns("hash-abc")
    store.close()

    assert df.shape[1] == 1
    assert len(sharpes) == 1
    assert sharpes[0] == pytest.approx(1.5)

    recovered = df.iloc[:, 0]
    pd.testing.assert_series_equal(
        recovered.reset_index(drop=True),
        original.reset_index(drop=True),
        check_names=False,
        rtol=1e-6,
    )


def test_fetch_historical_returns_multiple(tmp_path: Path) -> None:
    """Fetching by dataset_hash returns only attempts with that hash."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    _record(
        store, dataset_hash="hash-abc", observed_sharpe=1.0, returns=_make_returns(0)
    )
    _record(
        store, dataset_hash="hash-abc", observed_sharpe=1.2, returns=_make_returns(1)
    )
    _record(
        store, dataset_hash="hash-xyz", observed_sharpe=0.8, returns=_make_returns(2)
    )

    df_abc, sharpes_abc = store.fetch_historical_returns("hash-abc")
    df_xyz, sharpes_xyz = store.fetch_historical_returns("hash-xyz")
    store.close()

    assert df_abc.shape[1] == 2
    assert len(sharpes_abc) == 2
    assert sorted(sharpes_abc) == pytest.approx([1.0, 1.2])

    assert df_xyz.shape[1] == 1
    assert len(sharpes_xyz) == 1
    assert sharpes_xyz[0] == pytest.approx(0.8)


def test_fetch_historical_returns_empty(tmp_path: Path) -> None:
    """fetch_historical_returns returns (empty DataFrame, []) when no rows match."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)
    df, sharpes = store.fetch_historical_returns("nonexistent-hash")
    store.close()

    assert df.empty
    assert sharpes == []


def test_leaderboard_best_accepted(tmp_path: Path) -> None:
    """leaderboard returns the attempt with the highest observed_sharpe per strategy."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    _record(store, iteration=1, observed_sharpe=0.9, accepted=True)
    _record(store, iteration=2, observed_sharpe=1.8, accepted=True)

    board = store.leaderboard(strategy_name="strat_a")
    store.close()

    assert len(board) == 1
    assert board[0]["observed_sharpe"] == pytest.approx(1.8)
    assert board[0]["strategy_name"] == "strat_a"
    assert board[0]["iteration"] == 2


def test_exclude_id(tmp_path: Path) -> None:
    """fetch_historical_returns with exclude_id omits the specified attempt."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    _record(
        store, dataset_hash="hash-abc", observed_sharpe=1.0, returns=_make_returns(0)
    )
    _record(
        store, dataset_hash="hash-abc", observed_sharpe=1.5, returns=_make_returns(1)
    )

    # Get all to find the first id
    conn = sqlite3.connect(str(db))
    first_id = conn.execute(
        "SELECT id FROM attempts ORDER BY id ASC LIMIT 1"
    ).fetchone()[0]
    conn.close()

    df, sharpes = store.fetch_historical_returns("hash-abc", exclude_id=first_id)
    store.close()

    assert df.shape[1] == 1
    assert len(sharpes) == 1
    assert sharpes[0] == pytest.approx(1.5)
