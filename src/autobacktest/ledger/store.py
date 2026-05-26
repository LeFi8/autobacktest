"""SQLite-backed ledger store for tracking optimization attempts."""

from __future__ import annotations

import sqlite3
import zlib
from io import StringIO
from pathlib import Path

import pandas as pd


def _serialize_returns(series: pd.Series) -> bytes:
    json_str = series.to_json(orient="split", date_format="iso")
    return zlib.compress(json_str.encode("utf-8"))


def _deserialize_returns(blob: bytes) -> pd.Series:
    json_str = zlib.decompress(blob).decode("utf-8")
    return pd.read_json(StringIO(json_str), orient="split", typ="series")


_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
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

_CREATE_ATTEMPTS = """
CREATE TABLE IF NOT EXISTS attempts (
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


class LedgerStore:
    """Persist optimization attempts in a local SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_RUNS)
        self._conn.execute(_CREATE_ATTEMPTS)
        self._conn.commit()

    def create_run(
        self,
        run_id: str,
        strategy_name: str,
        program_path: str,
        provider: str,
        model: str,
        branch: str,
        dataset_hash: str,
        iterations: int,
        started_at: str,
    ) -> None:
        """Insert a new run record."""
        self._conn.execute(
            """
            INSERT INTO runs
                (run_id, strategy_name, program_path, provider, model,
                 branch, dataset_hash, iterations, started_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                strategy_name,
                program_path,
                provider,
                model,
                branch,
                dataset_hash,
                iterations,
                started_at,
            ),
        )
        self._conn.commit()

    def record_attempt(
        self,
        run_id: str,
        iteration: int,
        strategy_name: str,
        dataset_hash: str,
        config_yaml: str,
        observed_sharpe: float,
        deflated_sharpe: float,
        target_metric: str,
        target_metric_value: float,
        holdout_max_drawdown: float,
        holdout_turnover: float,
        regime_passed: bool,
        accepted: bool,
        committed: bool,
        commit_sha: str | None,
        rejection_reason: str | None,
        report_json: str,
        holdout_returns: pd.Series,
    ) -> None:
        """Serialize holdout returns and insert an attempt record."""
        blob = _serialize_returns(holdout_returns)
        self._conn.execute(
            """
            INSERT INTO attempts
                (run_id, iteration, strategy_name, dataset_hash, config_yaml,
                 observed_sharpe, deflated_sharpe, target_metric, target_metric_value,
                 holdout_max_drawdown, holdout_turnover, regime_passed, accepted,
                 committed, commit_sha, rejection_reason, report_json,
                 returns_blob, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    datetime('now'))
            """,
            (
                run_id,
                iteration,
                strategy_name,
                dataset_hash,
                config_yaml,
                observed_sharpe,
                deflated_sharpe,
                target_metric,
                target_metric_value,
                holdout_max_drawdown,
                holdout_turnover,
                int(regime_passed),
                int(accepted),
                int(committed),
                commit_sha,
                rejection_reason,
                report_json,
                blob,
            ),
        )
        self._conn.commit()

    def fetch_historical_returns(
        self,
        dataset_hash: str,
        exclude_id: int | None = None,
    ) -> tuple[pd.DataFrame, list[float]]:
        """Return a DataFrame of historical return series and observed Sharpe ratios.

        Each column in the returned DataFrame corresponds to one past attempt
        (indexed by attempt id). Returns an empty DataFrame and empty list when
        no matching attempts exist.
        """
        query = (
            "SELECT id, returns_blob, observed_sharpe FROM attempts "
            "WHERE dataset_hash = ?"
        )
        params: tuple[object, ...] = (dataset_hash,)
        if exclude_id is not None:
            query += " AND id != ?"
            params = (dataset_hash, exclude_id)

        rows = self._conn.execute(query, params).fetchall()
        if not rows:
            return pd.DataFrame(), []

        series_list = []
        sharpes: list[float] = []
        for row_id, blob, sharpe in rows:
            s = _deserialize_returns(bytes(blob))
            s.name = row_id
            series_list.append(s)
            sharpes.append(float(sharpe))

        matrix = pd.concat(series_list, axis=1)
        return matrix, sharpes

    def leaderboard(
        self,
        strategy_name: str | None = None,
    ) -> list[dict[str, object]]:
        """Return the best accepted attempt per strategy (highest observed Sharpe).

        Optionally filtered to a single strategy_name.
        """
        base = """
            SELECT
                a.strategy_name,
                a.run_id,
                a.iteration,
                a.observed_sharpe,
                a.deflated_sharpe,
                a.holdout_max_drawdown,
                a.holdout_turnover,
                a.created_at
            FROM attempts a
            INNER JOIN (
                SELECT strategy_name, MAX(observed_sharpe) AS best_sharpe
                FROM attempts
                WHERE accepted = 1
                {where}
                GROUP BY strategy_name
            ) best
            ON a.strategy_name = best.strategy_name
               AND a.observed_sharpe = best.best_sharpe
               AND a.accepted = 1
            ORDER BY a.observed_sharpe DESC
        """
        if strategy_name is not None:
            query = base.format(where="AND strategy_name = ?")
            rows = self._conn.execute(query, (strategy_name,)).fetchall()
        else:
            query = base.format(where="")
            rows = self._conn.execute(query).fetchall()

        results: list[dict[str, object]] = []
        for row in rows:
            results.append(
                {
                    "strategy_name": row[0],
                    "run_id": row[1],
                    "iteration": row[2],
                    "observed_sharpe": row[3],
                    "deflated_sharpe": row[4],
                    "holdout_max_drawdown": row[5],
                    "holdout_turnover": row[6],
                    "created_at": row[7],
                }
            )
        return results

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
