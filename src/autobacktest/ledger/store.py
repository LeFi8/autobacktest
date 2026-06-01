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
    in_sample_max_drawdown REAL NOT NULL,
    in_sample_turnover REAL NOT NULL,
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
    created_at TEXT NOT NULL,
    holdout_evaluated INTEGER NOT NULL DEFAULT 0,
    holdout_observed_sharpe REAL,
    holdout_returns_blob BLOB
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
            if "holdout_evaluated" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN holdout_evaluated INTEGER NOT NULL DEFAULT 0")
                migrated = True
            if "holdout_observed_sharpe" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN holdout_observed_sharpe REAL")
                migrated = True
            if "holdout_returns_blob" not in columns:
                self._conn.execute("ALTER TABLE attempts ADD COLUMN holdout_returns_blob BLOB")
                migrated = True
            if migrated:
                # Backfill target_metric_value using observed_sharpe for older attempts
                self._conn.execute(
                    "UPDATE attempts SET target_metric_value = observed_sharpe WHERE target_metric = 'sharpe'"
                )
                self._conn.commit()

            # Rename holdout_max_drawdown/turnover to in_sample_* (store in-sample values)
            if "holdout_max_drawdown" in columns and "in_sample_max_drawdown" not in columns:
                self._conn.execute("ALTER TABLE attempts RENAME COLUMN holdout_max_drawdown TO in_sample_max_drawdown")
            if "holdout_turnover" in columns and "in_sample_turnover" not in columns:
                self._conn.execute("ALTER TABLE attempts RENAME COLUMN holdout_turnover TO in_sample_turnover")

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
        in_sample_max_drawdown: float,
        in_sample_turnover: float,
        regime_passed: bool,
        accepted: bool,
        committed: bool,
        commit_sha: str | None,
        rejection_reason: str | None,
        report_json: str,
        selection_returns: pd.Series,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int = 0,
        cost: float = 0.0,
        holdout_evaluated: bool = False,
        holdout_observed_sharpe: float | None = None,
        holdout_returns: pd.Series | None = None,
    ) -> None:
        """Serialize in-sample selection returns and insert an attempt record.

        When ``holdout_evaluated`` is True the holdout returns are also
        persisted so that the confirmation gate's multiple-testing penalty
        (``_deflate_holdout``) can be computed later.
        """
        selection_blob = _serialize_returns(selection_returns)
        holdout_blob = _serialize_returns(holdout_returns) if holdout_returns is not None else None
        self._conn.execute(
            """
            INSERT INTO attempts
                (run_id, iteration, strategy_name, dataset_hash, config_yaml,
                 observed_sharpe, deflated_sharpe, target_metric, target_metric_value,
                 in_sample_max_drawdown, in_sample_turnover, regime_passed, accepted,
                 committed, commit_sha, rejection_reason, report_json,
                 returns_blob, prompt_tokens, completion_tokens, total_tokens, cost,
                 created_at,
                 holdout_evaluated, holdout_observed_sharpe, holdout_returns_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    datetime('now'),
                    ?, ?, ?)
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
                in_sample_max_drawdown,
                in_sample_turnover,
                int(regime_passed),
                int(accepted),
                int(committed),
                commit_sha,
                rejection_reason,
                report_json,
                selection_blob,
                prompt_tokens,
                completion_tokens,
                total_tokens,
                cost,
                int(holdout_evaluated),
                holdout_observed_sharpe,
                holdout_blob,
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

    def fetch_holdout_history(
        self,
        dataset_hash: str,
    ) -> tuple[pd.DataFrame, list[float]]:
        """Return holdout return series and observed Sharpes for peeked attempts.

        Only rows where ``holdout_evaluated = 1`` are included — these are
        the in-sample winners that were confirmed (or rejected) on the holdout.
        The returned matrix drives the holdout DSR multiple-testing deflation
        in ``_deflate_holdout``.

        Returns:
            Tuple of (DataFrame of holdout returns, list of holdout Sharpe ratios).
            Empty DataFrame and empty list when no holdout-peeked rows exist.
        """
        rows = self._conn.execute(
            """
            SELECT id, holdout_returns_blob, holdout_observed_sharpe
            FROM attempts
            WHERE dataset_hash = ? AND holdout_evaluated = 1
            """,
            (dataset_hash,),
        ).fetchall()

        if not rows:
            return pd.DataFrame(), []

        series_list: list[pd.Series] = []
        sharpes: list[float] = []
        for row_id, blob, sharpe in rows:
            if blob is None:
                continue
            s = _deserialize_returns(bytes(blob))
            s.name = row_id
            series_list.append(s)
            sharpes.append(float(sharpe))

        if not series_list:
            return pd.DataFrame(), []

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

        Each dict contains **in-sample** scalar metrics only — the holdout
        metric values are **not** exposed (they are hidden from the LLM to
        prevent overfitting).  A ``holdout_confirmed`` boolean indicates
        whether the attempt passed the final holdout confirmation gate.

        Also includes a compact ``config_fingerprint`` with only the
        ``universe`` and ``params`` keys parsed from ``config_yaml``.

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
                regime_passed,
                rejection_reason,
                config_yaml,
                holdout_evaluated
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
                regime_passed,
                rejection_reason,
                config_yaml,
                holdout_evaluated,
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
                    "regime_passed": bool(regime_passed),
                    "rejection_reason": rejection_reason,
                    "holdout_confirmed": bool(holdout_evaluated) and bool(committed),
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
                in_sample_max_drawdown,
                in_sample_turnover,
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
                    "in_sample_max_drawdown": row[5],
                    "in_sample_turnover": row[6],
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
                in_sample_max_drawdown,
                in_sample_turnover,
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
                    a.in_sample_max_drawdown,
                    a.in_sample_turnover,
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
                    "in_sample_max_drawdown": row[5],
                    "in_sample_turnover": row[6],
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
