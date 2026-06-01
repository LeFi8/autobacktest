"""SQLite-backed ledger store for tracking optimization attempts."""

from __future__ import annotations

import sqlite3
import zlib
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from autobacktest.config import settings


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
    prompt_tokens INTEGER NOT NULL DEFAULT 0,
    completion_tokens INTEGER NOT NULL DEFAULT 0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    cost REAL NOT NULL DEFAULT 0.0,
    created_at TEXT NOT NULL
)
"""


class LedgerStore:
    """Persist optimization attempts in a local SQLite database."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path), timeout=settings.db_timeout)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(_CREATE_RUNS)
        self._conn.execute(_CREATE_ATTEMPTS)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_run_id ON attempts(run_id)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_strategy_name ON attempts(strategy_name)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_dataset_hash ON attempts(dataset_hash)")
        self._conn.commit()

        # Schema migration for older databases missing target_metric/value columns
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA table_info(attempts)")
        columns = [row[1] for row in cursor.fetchall()]
        if columns:
            migrated = False
            if "target_metric" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN target_metric TEXT NOT NULL DEFAULT 'sharpe'")
                migrated = True
            if "target_metric_value" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN target_metric_value REAL NOT NULL DEFAULT 0.0")
                migrated = True
            if "prompt_tokens" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN prompt_tokens INTEGER NOT NULL DEFAULT 0")
                migrated = True
            if "completion_tokens" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN completion_tokens INTEGER NOT NULL DEFAULT 0")
                migrated = True
            if "total_tokens" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN total_tokens INTEGER NOT NULL DEFAULT 0")
                migrated = True
            if "cost" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN cost REAL NOT NULL DEFAULT 0.0")
                migrated = True
            if migrated:
                # Backfill target_metric_value using observed_sharpe for older attempts
                self._conn.execute(
                    "UPDATE attempts SET target_metric_value = observed_sharpe WHERE target_metric = 'sharpe'"
                )
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
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost: float = 0.0,
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
                 returns_blob, prompt_tokens, completion_tokens, total_tokens, cost,
                 created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
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
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
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
        query = "SELECT id, returns_blob, observed_sharpe FROM attempts WHERE dataset_hash = ?"
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

    def fetch_configs(
        self,
        dataset_hash: str,
        exclude_id: int | None = None,
    ) -> list[str]:
        """Return all config_yaml strings for a given dataset_hash (chronological).

        Args:
            dataset_hash: Stable hash of the sorted universe tickers.
            exclude_id: Optional attempt id to exclude (e.g. the current candidate).

        Returns:
            List of YAML config strings, oldest first.
        """
        query = "SELECT config_yaml FROM attempts WHERE dataset_hash = ?"
        params: tuple[object, ...] = (dataset_hash,)
        if exclude_id is not None:
            query += " AND id != ?"
            params = (dataset_hash, exclude_id)
        query += " ORDER BY id ASC"

        rows = self._conn.execute(query, params).fetchall()
        return [str(row[0]) for row in rows]

    def fetch_attempt_summaries(
        self,
        dataset_hash: str,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Return a chronological list of attempt summary dicts for a dataset_hash.

        Each dict contains scalar metrics and a compact ``config_fingerprint``
        with only the ``universe`` and ``params`` keys parsed from ``config_yaml``.

        Args:
            dataset_hash: Stable hash of the sorted universe tickers.
            limit: When not None, return only the last *limit* rows (most recent).

        Returns:
            List of summary dicts, oldest first. Empty list when no rows match.
        """
        query = """
            SELECT
                iteration,
                accepted,
                committed,
                target_metric_value,
                observed_sharpe,
                deflated_sharpe,
                holdout_max_drawdown,
                holdout_turnover,
                regime_passed,
                rejection_reason,
                config_yaml
            FROM attempts
            WHERE dataset_hash = ?
            ORDER BY id ASC
        """
        rows = self._conn.execute(query, (dataset_hash,)).fetchall()

        if not rows:
            return []

        if limit is not None:
            rows = rows[-limit:]

        results: list[dict[str, Any]] = []
        for row in rows:
            (
                iteration,
                accepted,
                committed,
                target_metric_value,
                observed_sharpe,
                deflated_sharpe,
                holdout_max_drawdown,
                holdout_turnover,
                regime_passed,
                rejection_reason,
                config_yaml,
            ) = row

            try:
                parsed = yaml.safe_load(config_yaml)
                if isinstance(parsed, dict):
                    fingerprint: dict[str, Any] = {}
                    if "universe" in parsed:
                        fingerprint["universe"] = parsed["universe"]
                    if "params" in parsed:
                        fingerprint["params"] = parsed["params"]
                else:
                    fingerprint = {}
            except yaml.YAMLError:
                fingerprint = {}

            results.append(
                {
                    "iteration": int(iteration),
                    "accepted": bool(accepted),
                    "committed": bool(committed),
                    "target_metric_value": float(target_metric_value),
                    "observed_sharpe": float(observed_sharpe),
                    "deflated_sharpe": float(deflated_sharpe),
                    "holdout_max_drawdown": float(holdout_max_drawdown),
                    "holdout_turnover": float(holdout_turnover),
                    "regime_passed": bool(regime_passed),
                    "rejection_reason": rejection_reason,
                    "config_fingerprint": fingerprint,
                }
            )
        return results

    def latest_run_id(self) -> str | None:
        """Return the most recently started run id, if any runs exist."""
        row = self._conn.execute(
            """
            SELECT run_id
            FROM runs
            ORDER BY started_at DESC, run_id DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return None
        return str(row[0])

    def get_run(self, run_id: str) -> dict[str, object] | None:
        """Return metadata for one run id, or None when it is absent."""
        row = self._conn.execute(
            """
            SELECT
                run_id, strategy_name, program_path, provider, model,
                branch, dataset_hash, iterations, started_at
            FROM runs
            WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        if row is None:
            return None
        return {
            "run_id": row[0],
            "strategy_name": row[1],
            "program_path": row[2],
            "provider": row[3],
            "model": row[4],
            "branch": row[5],
            "dataset_hash": row[6],
            "iterations": row[7],
            "started_at": row[8],
        }

    def attempts_for_run(
        self,
        run_id: str,
        strategy_name: str | None = None,
    ) -> list[dict[str, object]]:
        """Return attempts recorded for one run, optionally scoped to a strategy."""
        filters = ["run_id = ?"]
        params: list[object] = [run_id]
        if strategy_name is not None:
            filters.append("strategy_name = ?")
            params.append(strategy_name)
        where_sql = " AND ".join(filters)

        rows = self._conn.execute(
            f"""
            SELECT
                strategy_name,
                run_id,
                iteration,
                observed_sharpe,
                deflated_sharpe,
                holdout_max_drawdown,
                holdout_turnover,
                created_at,
                target_metric,
                target_metric_value,
                accepted,
                committed,
                rejection_reason,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost
            FROM attempts
            WHERE {where_sql}
            ORDER BY strategy_name ASC, iteration ASC, id ASC
            """,
            tuple(params),
        ).fetchall()

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
                    "target_metric": row[8],
                    "target_metric_value": row[9],
                    "accepted": bool(row[10]),
                    "committed": bool(row[11]),
                    "rejection_reason": row[12],
                    "prompt_tokens": int(row[13] or 0),
                    "completion_tokens": int(row[14] or 0),
                    "total_tokens": int(row[15] or 0),
                    "cost": float(row[16] or 0.0),
                }
            )
        return results

    def leaderboard(
        self,
        strategy_name: str | None = None,
        run_id: str | None = None,
    ) -> list[dict[str, object]]:
        """Return the best accepted attempt per strategy (highest target metric value).

        Optionally filtered to a single strategy_name and/or run_id.
        """
        filters = ["accepted = 1"]
        params: list[object] = []
        if strategy_name is not None:
            filters.append("strategy_name = ?")
            params.append(strategy_name)
        if run_id is not None:
            filters.append("run_id = ?")
            params.append(run_id)
        where_sql = " AND ".join(filters)

        rows = self._conn.execute(
            f"""
            SELECT
                strategy_name,
                run_id,
                iteration,
                observed_sharpe,
                deflated_sharpe,
                holdout_max_drawdown,
                holdout_turnover,
                created_at,
                target_metric,
                target_metric_value
            FROM (
                SELECT
                    a.strategy_name,
                    a.run_id,
                    a.iteration,
                    a.observed_sharpe,
                    a.deflated_sharpe,
                    a.holdout_max_drawdown,
                    a.holdout_turnover,
                    a.created_at,
                    a.target_metric,
                    a.target_metric_value,
                    ROW_NUMBER() OVER (
                        PARTITION BY a.strategy_name
                        ORDER BY a.target_metric_value DESC, a.id ASC
                    ) AS row_num
                FROM attempts a
                WHERE {where_sql}
            ) ranked
            WHERE row_num = 1
            ORDER BY target_metric_value DESC, strategy_name ASC
            """,
            tuple(params),
        ).fetchall()

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
                    "target_metric": row[8],
                    "target_metric_value": row[9],
                }
            )
        return results

    def list_runs(self) -> list[dict[str, object]]:
        """Return metadata for all recorded runs."""
        query = """
            SELECT
                run_id, strategy_name, program_path, provider, model,
                branch, dataset_hash, iterations, started_at
            FROM runs
            ORDER BY started_at DESC
        """
        rows = self._conn.execute(query).fetchall()
        results: list[dict[str, object]] = []
        for row in rows:
            results.append(
                {
                    "run_id": row[0],
                    "strategy_name": row[1],
                    "program_path": row[2],
                    "provider": row[3],
                    "model": row[4],
                    "branch": row[5],
                    "dataset_hash": row[6],
                    "iterations": row[7],
                    "started_at": row[8],
                }
            )
        return results

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
