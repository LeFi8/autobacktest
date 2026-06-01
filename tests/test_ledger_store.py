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


def _make_returns(seed: int = 0, n: int = 50) -> pd.Series:
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
    returns: pd.Series | None = None,
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
        selection_returns=returns,
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

    _record(store, dataset_hash="hash-abc", observed_sharpe=1.0, returns=_make_returns(0))
    _record(store, dataset_hash="hash-abc", observed_sharpe=1.2, returns=_make_returns(1))
    _record(store, dataset_hash="hash-xyz", observed_sharpe=0.8, returns=_make_returns(2))

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

    _record(store, dataset_hash="hash-abc", observed_sharpe=1.0, returns=_make_returns(0))
    _record(store, dataset_hash="hash-abc", observed_sharpe=1.5, returns=_make_returns(1))

    # Get all to find the first id
    conn = sqlite3.connect(str(db))
    first_id = conn.execute("SELECT id FROM attempts ORDER BY id ASC LIMIT 1").fetchone()[0]
    conn.close()

    df, sharpes = store.fetch_historical_returns("hash-abc", exclude_id=first_id)
    store.close()

    assert df.shape[1] == 1
    assert len(sharpes) == 1
    assert sharpes[0] == pytest.approx(1.5)


def test_list_runs_and_leaderboard_keys(tmp_path: Path) -> None:
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

    _record(store, iteration=1, observed_sharpe=1.2, accepted=True)

    runs = store.list_runs()
    board = store.leaderboard()
    store.close()

    assert len(runs) == 1
    assert runs[0]["run_id"] == "run-1"
    assert runs[0]["provider"] == "openai"

    assert len(board) == 1
    assert board[0]["target_metric"] == "sharpe"
    assert board[0]["target_metric_value"] == pytest.approx(1.2)


def test_run_scoped_attempts_and_leaderboard(tmp_path: Path) -> None:
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    store.create_run(
        run_id="run-old",
        strategy_name="strat_a",
        program_path="/tmp/strat.py",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=10,
        started_at="2024-01-01T00:00:00",
    )
    store.create_run(
        run_id="run-new",
        strategy_name="strat_a",
        program_path="/tmp/strat.py",
        provider="openai",
        model="gpt-4o",
        branch="main",
        dataset_hash="hash-abc",
        iterations=10,
        started_at="2024-01-02T00:00:00",
    )

    _record(store, run_id="run-old", strategy_name="strat_a", observed_sharpe=2.0)
    _record(store, run_id="run-new", strategy_name="strat_a", observed_sharpe=0.5)
    _record(store, run_id="run-new", strategy_name="strat_b", observed_sharpe=1.5)

    assert store.latest_run_id() == "run-new"

    attempts = store.attempts_for_run("run-new")
    run_strategies = {row["strategy_name"] for row in attempts}
    latest_board = store.leaderboard(strategy_name="strat_a", run_id="run-new")
    all_time_board = store.leaderboard(strategy_name="strat_a")
    store.close()

    assert run_strategies == {"strat_a", "strat_b"}
    assert len(latest_board) == 1
    assert latest_board[0]["run_id"] == "run-new"
    assert latest_board[0]["observed_sharpe"] == pytest.approx(0.5)
    assert all_time_board[0]["run_id"] == "run-old"
    assert all_time_board[0]["observed_sharpe"] == pytest.approx(2.0)


def test_fetch_configs_returns_committed_and_rejected(tmp_path: Path) -> None:
    """fetch_configs returns ALL attempts (committed and rejected), not just committed."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    # committed=True → simulates a successful commit
    store.record_attempt(
        run_id="run-1",
        iteration=1,
        strategy_name="strat_a",
        dataset_hash="hash-abc",
        config_yaml="param: committed_value",
        observed_sharpe=1.2,
        deflated_sharpe=1.1,
        target_metric="sharpe",
        target_metric_value=1.2,
        holdout_max_drawdown=0.05,
        holdout_turnover=0.3,
        regime_passed=True,
        accepted=True,
        committed=True,
        commit_sha="abc123",
        rejection_reason=None,
        report_json="{}",
        holdout_returns=_make_returns(seed=0),
        selection_returns=_make_returns(seed=0),
    )

    # committed=False → rejected attempt (diversity or gate rejection)
    store.record_attempt(
        run_id="run-1",
        iteration=2,
        strategy_name="strat_a",
        dataset_hash="hash-abc",
        config_yaml="param: rejected_value",
        observed_sharpe=0.5,
        deflated_sharpe=0.4,
        target_metric="sharpe",
        target_metric_value=0.5,
        holdout_max_drawdown=0.10,
        holdout_turnover=0.5,
        regime_passed=True,
        accepted=False,
        committed=False,
        commit_sha=None,
        rejection_reason="diversity_config",
        report_json="{}",
        holdout_returns=_make_returns(seed=1),
        selection_returns=_make_returns(seed=1),
    )

    configs = store.fetch_configs("hash-abc")
    store.close()

    # Both committed and rejected configs must be returned
    assert len(configs) == 2
    assert "committed_value" in configs[0]
    assert "rejected_value" in configs[1]


def test_fetch_configs_excludes_other_hashes(tmp_path: Path) -> None:
    """fetch_configs only returns configs for the requested dataset_hash."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    store.record_attempt(
        run_id="run-1",
        iteration=1,
        strategy_name="strat_a",
        dataset_hash="hash-abc",
        config_yaml="param: abc_value",
        observed_sharpe=1.0,
        deflated_sharpe=0.9,
        target_metric="sharpe",
        target_metric_value=1.0,
        holdout_max_drawdown=0.05,
        holdout_turnover=0.3,
        regime_passed=True,
        accepted=True,
        committed=False,
        commit_sha=None,
        rejection_reason=None,
        report_json="{}",
        holdout_returns=_make_returns(seed=0),
        selection_returns=_make_returns(seed=0),
    )
    store.record_attempt(
        run_id="run-1",
        iteration=2,
        strategy_name="strat_a",
        dataset_hash="hash-xyz",
        config_yaml="param: xyz_value",
        observed_sharpe=1.0,
        deflated_sharpe=0.9,
        target_metric="sharpe",
        target_metric_value=1.0,
        holdout_max_drawdown=0.05,
        holdout_turnover=0.3,
        regime_passed=True,
        accepted=True,
        committed=False,
        commit_sha=None,
        rejection_reason=None,
        report_json="{}",
        holdout_returns=_make_returns(seed=1),
        selection_returns=_make_returns(seed=1),
    )

    configs = store.fetch_configs("hash-abc")
    store.close()

    assert len(configs) == 1
    assert "abc_value" in configs[0]


# ---------------------------------------------------------------------------
# fetch_attempt_summaries tests
# ---------------------------------------------------------------------------


def test_fetch_attempt_summaries_empty(tmp_path: Path) -> None:
    """fetch_attempt_summaries returns [] when no rows match the hash."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)
    result = store.fetch_attempt_summaries("nonexistent-hash")
    store.close()

    assert result == []


def test_fetch_attempt_summaries_basic(tmp_path: Path) -> None:
    """fetch_attempt_summaries returns all expected fields in chronological order."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    _record(store, iteration=1, observed_sharpe=1.0, accepted=True)
    _record(store, iteration=2, observed_sharpe=1.5, accepted=False)

    results = store.fetch_attempt_summaries("hash-abc")
    store.close()

    assert len(results) == 2

    # chronological order
    assert results[0]["iteration"] == 1
    assert results[1]["iteration"] == 2

    expected_keys = {
        "iteration",
        "accepted",
        "committed",
        "target_metric_value",
        "observed_sharpe",
        "deflated_sharpe",
        "holdout_confirmed",
        "regime_passed",
        "rejection_reason",
        "config_fingerprint",
    }
    assert set(results[0].keys()) == expected_keys

    first = results[0]
    assert first["iteration"] == 1
    assert first["accepted"] is True
    assert first["committed"] is False
    assert first["observed_sharpe"] == pytest.approx(1.0)
    assert first["deflated_sharpe"] == pytest.approx(0.9)
    assert first["target_metric_value"] == pytest.approx(1.0)
    assert first["holdout_confirmed"] is False
    assert first["regime_passed"] is True
    assert first["rejection_reason"] is None

    second = results[1]
    assert second["iteration"] == 2
    assert second["accepted"] is False
    assert second["observed_sharpe"] == pytest.approx(1.5)


def test_fetch_attempt_summaries_filters_by_hash(tmp_path: Path) -> None:
    """fetch_attempt_summaries only returns rows matching the requested dataset_hash."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    _record(store, dataset_hash="hash-abc", iteration=1, observed_sharpe=1.0)
    _record(store, dataset_hash="hash-xyz", iteration=2, observed_sharpe=0.5)

    results_abc = store.fetch_attempt_summaries("hash-abc")
    results_xyz = store.fetch_attempt_summaries("hash-xyz")
    store.close()

    assert len(results_abc) == 1
    assert results_abc[0]["observed_sharpe"] == pytest.approx(1.0)

    assert len(results_xyz) == 1
    assert results_xyz[0]["observed_sharpe"] == pytest.approx(0.5)


def test_fetch_attempt_summaries_limit(tmp_path: Path) -> None:
    """fetch_attempt_summaries with limit=3 returns the 3 most recent rows."""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    for i in range(1, 6):
        _record(store, iteration=i, observed_sharpe=float(i))

    results = store.fetch_attempt_summaries("hash-abc", limit=3)
    store.close()

    assert len(results) == 3
    # most-recent 3 are iterations 3, 4, 5 — still in chronological (oldest-first) order
    assert [r["iteration"] for r in results] == [3, 4, 5]


def test_fetch_attempt_summaries_config_fingerprint(tmp_path: Path) -> None:
    """config_fingerprint contains only universe and params; invalid YAML yields {}."""
    full_yaml = """universe:
  - SPY
  - TIP
benchmark: SPY
momentum_lookback: 12
params:
  top_n: 3
  canary_threshold: 0.5
"""
    db = tmp_path / "ledger.db"
    store = LedgerStore(db)

    # record with full YAML
    store.record_attempt(
        run_id="run-1",
        iteration=1,
        strategy_name="strat_a",
        dataset_hash="hash-fp",
        config_yaml=full_yaml,
        observed_sharpe=1.2,
        deflated_sharpe=1.1,
        target_metric="sharpe",
        target_metric_value=1.2,
        holdout_max_drawdown=0.05,
        holdout_turnover=0.3,
        regime_passed=True,
        accepted=True,
        committed=False,
        commit_sha=None,
        rejection_reason=None,
        report_json="{}",
        holdout_returns=_make_returns(seed=0),
        selection_returns=_make_returns(seed=0),
    )

    # record with invalid YAML
    store.record_attempt(
        run_id="run-1",
        iteration=2,
        strategy_name="strat_a",
        dataset_hash="hash-fp",
        config_yaml=": invalid: yaml: [[[",
        observed_sharpe=0.5,
        deflated_sharpe=0.4,
        target_metric="sharpe",
        target_metric_value=0.5,
        holdout_max_drawdown=0.10,
        holdout_turnover=0.5,
        regime_passed=False,
        accepted=False,
        committed=False,
        commit_sha=None,
        rejection_reason="bad_config",
        report_json="{}",
        holdout_returns=_make_returns(seed=1),
        selection_returns=_make_returns(seed=1),
    )

    results = store.fetch_attempt_summaries("hash-fp")
    store.close()

    assert len(results) == 2

    fp = results[0]["config_fingerprint"]
    assert isinstance(fp, dict)
    assert set(fp.keys()) == {"universe", "params"}
    assert fp["universe"] == ["SPY", "TIP"]
    assert fp["params"] == {"top_n": 3, "canary_threshold": 0.5}
    # benchmark and momentum_lookback must NOT appear
    assert "benchmark" not in fp
    assert "momentum_lookback" not in fp

    # invalid YAML → empty fingerprint
    assert results[1]["config_fingerprint"] == {}
