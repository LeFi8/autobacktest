"""SQLite-backed ledger store for tracking optimization attempts."""

from __future__ import annotations

import contextlib
import sqlite3
import threading
import zlib
from io import StringIO
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from autobacktest.config import settings


def _serialize_returns(series: pd.Series) -> bytes:
    """Compress a pandas Series to a zlib-compressed JSON blob for SQLite storage."""
    json_str = series.to_json(orient="split", date_format="iso")
    return zlib.compress(json_str.encode("utf-8"))


def _deserialize_returns(blob: bytes) -> pd.Series:
    """Decompress and reconstruct a pandas Series from a zlib-compressed JSON blob."""
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
    holdout_returns_blob BLOB,
    optimization_applied INTEGER NOT NULL DEFAULT 0,
    optimization_gain REAL NOT NULL DEFAULT 0.0
)
"""


class LedgerStore:
    """Persist optimization attempts in a local SQLite database.

    Thread safety: each thread creates its own ``sqlite3.Connection`` so
    that concurrent ``ThreadPoolExecutor`` workers do not share a single
    connection.  WAL journal mode allows concurrent readers.
    """

    def __init__(self, db_path: Path) -> None:
        self._db_path = db_path
        self._local: dict[int, sqlite3.Connection] = {}
        self._lock = threading.Lock()
        self._schema_initialized = False
        # Trigger schema creation on the calling thread
        self._conn()

    def __enter__(self) -> LedgerStore:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    # ------------------------------------------------------------------
    # Connection management (one per thread)
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        tid = threading.get_ident()
        with self._lock:
            if tid not in self._local:
                c = sqlite3.connect(str(self._db_path), timeout=settings.db_timeout)
                c.execute("PRAGMA journal_mode=WAL")
                self._local[tid] = c
            conn = self._local[tid]

            if not self._schema_initialized:
                self._init_schema(conn)
                self._schema_initialized = True

        return conn

    def _init_schema(self, conn: sqlite3.Connection) -> None:
        """Initialize table schema and run migrations."""
        conn.execute(_CREATE_RUNS)
        conn.execute(_CREATE_ATTEMPTS)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_run_id ON attempts(run_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_strategy_name ON attempts(strategy_name)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_attempts_dataset_hash ON attempts(dataset_hash)")
        conn.commit()

        # Schema migration for older databases missing target_metric/value columns
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(attempts)")
        columns = [row[1] for row in cursor.fetchall()]
        if columns:
            column_specs = [
                ("target_metric", "TEXT NOT NULL DEFAULT 'sharpe'"),
                ("target_metric_value", "REAL NOT NULL DEFAULT 0.0"),
                ("prompt_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("completion_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("total_tokens", "INTEGER NOT NULL DEFAULT 0"),
                ("cost", "REAL NOT NULL DEFAULT 0.0"),
                ("holdout_evaluated", "INTEGER NOT NULL DEFAULT 0"),
                ("holdout_observed_sharpe", "REAL"),
                ("holdout_returns_blob", "BLOB"),
                ("optimization_applied", "INTEGER NOT NULL DEFAULT 0"),
                ("optimization_gain", "REAL NOT NULL DEFAULT 0.0"),
            ]
            if self._ensure_columns(conn, "attempts", column_specs):
                conn.execute("UPDATE attempts SET target_metric_value = observed_sharpe WHERE target_metric = 'sharpe'")
                conn.commit()

            if "holdout_max_drawdown" in columns and "in_sample_max_drawdown" not in columns:
                conn.execute("ALTER TABLE attempts RENAME COLUMN holdout_max_drawdown TO in_sample_max_drawdown")
            if "holdout_turnover" in columns and "in_sample_turnover" not in columns:
                conn.execute("ALTER TABLE attempts RENAME COLUMN holdout_turnover TO in_sample_turnover")

    def _ensure_columns(self, conn: sqlite3.Connection, table: str, column_specs: list[tuple[str, str]]) -> bool:
        """Ensure that the specified columns exist in the table, adding them if missing."""
        cursor = conn.cursor()
        cursor.execute(f"PRAGMA table_info({table})")
        columns = [row[1] for row in cursor.fetchall()]
        if not columns:
            return False

        migrated = False
        for col_name, col_def in column_specs:
            if col_name not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_def}")
                migrated = True
        return migrated

    def _marshal_row(self, row: tuple[Any, ...], schema: list[tuple[str, Any]]) -> dict[str, Any]:
        """Helper to marshal a database row tuple into a typed dict."""
        res: dict[str, Any] = {}
        for i, (name, cast_fn) in enumerate(schema):
            val = row[i]
            if val is None:
                res[name] = None
            else:
                res[name] = cast_fn(val)
        return res

    def _parse_fingerprint(self, config_yaml: str) -> dict[str, Any]:
        try:
            parsed = yaml.safe_load(config_yaml)
            if isinstance(parsed, dict):
                fingerprint: dict[str, Any] = {}
                if "universe" in parsed:
                    fingerprint["universe"] = parsed["universe"]
                if "params" in parsed:
                    fingerprint["params"] = parsed["params"]
                return fingerprint
        except Exception:
            pass
        return {}

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
        self._conn().execute(
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
        self._conn().commit()

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
        optimization_applied: bool = False,
        optimization_gain: float = 0.0,
    ) -> int:
        """Serialize in-sample selection returns and insert an attempt record.

        When ``holdout_evaluated`` is True the holdout returns are also
        persisted so that the confirmation gate's multiple-testing penalty
        (``_deflate_holdout``) can be computed later.

        Returns:
            The auto-generated ``id`` (primary key) of the new attempt row.
        """
        selection_blob = _serialize_returns(selection_returns)
        holdout_blob = _serialize_returns(holdout_returns) if holdout_returns is not None else None
        cursor = self._conn().execute(
            """
            INSERT INTO attempts
                (run_id, iteration, strategy_name, dataset_hash, config_yaml,
                 observed_sharpe, deflated_sharpe, target_metric, target_metric_value,
                 in_sample_max_drawdown, in_sample_turnover, regime_passed, accepted,
                 committed, commit_sha, rejection_reason, report_json,
                 returns_blob, prompt_tokens, completion_tokens, total_tokens, cost,
                 created_at,
                 holdout_evaluated, holdout_observed_sharpe, holdout_returns_blob,
                 optimization_applied, optimization_gain)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                    datetime('now'),
                    ?, ?, ?,
                    ?, ?)
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
                int(optimization_applied),
                optimization_gain,
            ),
        )
        self._conn().commit()
        row_id = cursor.lastrowid
        if row_id is None:
            raise RuntimeError("INSERT into attempts did not return a row id")
        return int(row_id)

    def mark_committed(self, attempt_id: int, commit_sha: str) -> None:
        """Mark a previously recorded attempt as committed with its git SHA.

        Called after the git commit succeeds in the two-phase atomic write
        (record → commit → mark_committed).
        """
        self._conn().execute(
            "UPDATE attempts SET committed = 1, commit_sha = ? WHERE id = ?",
            (commit_sha, attempt_id),
        )
        self._conn().commit()

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

        rows = self._conn().execute(query, params).fetchall()
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

    def fetch_run_returns(
        self,
        run_id: str,
        accepted_only: bool = False,
    ) -> tuple[pd.Series | None, pd.DataFrame]:
        """Fetch historical returns for a run.

        Returns:
            Tuple of (benchmark_returns, alternative_returns_df).
            benchmark_returns is iteration=0.
            alternative_returns_df has columns named by iteration/strategy.
        """
        query = "SELECT iteration, accepted, returns_blob, strategy_name, id FROM attempts WHERE run_id = ?"
        rows = self._conn().execute(query, (run_id,)).fetchall()
        if not rows:
            return None, pd.DataFrame()

        benchmark: pd.Series | None = None
        alternatives: list[pd.Series] = []

        for iteration, accepted, blob, strategy_name, attempt_id in rows:
            s = _deserialize_returns(bytes(blob))
            if iteration == 0:
                s.name = f"{strategy_name}_baseline_id{attempt_id}"
                benchmark = s
            else:
                if accepted_only and not accepted:
                    continue
                s.name = f"{strategy_name}_iter{iteration}_id{attempt_id}"
                alternatives.append(s)

        if not alternatives:
            return benchmark, pd.DataFrame()

        df_alt = pd.concat(alternatives, axis=1)
        return benchmark, df_alt

    def fetch_holdout_history(
        self,
        dataset_hash: str,
        exclude_id: int | None = None,
    ) -> tuple[pd.DataFrame, list[float]]:
        """Return holdout return series and observed Sharpes for peeked attempts.

        Only rows where ``holdout_evaluated = 1`` are included — these are
        the in-sample winners that were confirmed (or rejected) on the holdout.
        The returned matrix drives the holdout DSR multiple-testing deflation
        in ``_deflate_holdout``.

        Args:
            exclude_id: Optional attempt id to exclude (used to prevent the
                incumbent's own holdout from contaminating its null distribution).

        Returns:
            Tuple of (DataFrame of holdout returns, list of holdout Sharpe ratios).
            Empty DataFrame and empty list when no holdout-peeked rows exist.
        """
        query = (
            "SELECT id, holdout_returns_blob, holdout_observed_sharpe "
            "FROM attempts WHERE dataset_hash = ? AND holdout_evaluated = 1"
        )
        params: tuple[object, ...] = (dataset_hash,)
        if exclude_id is not None:
            query += " AND id != ?"
            params = (dataset_hash, exclude_id)

        rows = self._conn().execute(query, params).fetchall()

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

        rows = self._conn().execute(query, params).fetchall()
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
        rows = self._conn().execute(query, (dataset_hash,)).fetchall()

        if not rows:
            return []

        if limit is not None:
            rows = rows[-limit:]

        results: list[dict[str, Any]] = []
        schema = [
            ("iteration", int),
            ("accepted", bool),
            ("committed", bool),
            ("target_metric_value", float),
            ("observed_sharpe", float),
            ("deflated_sharpe", float),
            ("regime_passed", bool),
            ("rejection_reason", lambda x: str(x) if x is not None else None),
            ("config_yaml", str),
            ("holdout_evaluated", bool),
        ]
        for row in rows:
            m = self._marshal_row(row, schema)
            fingerprint = self._parse_fingerprint(m["config_yaml"])
            results.append(
                {
                    "iteration": m["iteration"],
                    "accepted": m["accepted"],
                    "committed": m["committed"],
                    "target_metric_value": m["target_metric_value"],
                    "observed_sharpe": m["observed_sharpe"],
                    "deflated_sharpe": m["deflated_sharpe"],
                    "regime_passed": m["regime_passed"],
                    "rejection_reason": m["rejection_reason"],
                    "holdout_confirmed": bool(m["holdout_evaluated"]) and bool(m["committed"]),
                    "config_fingerprint": fingerprint,
                }
            )
        return results

    def fetch_param_importance_data(
        self,
        dataset_hash: str,
    ) -> tuple[list[str], list[float]]:
        """Return config YAMLs and target metric values for all attempts.

        Only includes attempts that have been evaluated (have a report_json).
        Results are ordered chronologically by attempt id.

        Args:
            dataset_hash: Stable hash of the sorted universe tickers.

        Returns:
            Tuple of (config_yaml_strings, target_metric_values).
        """
        rows = (
            self._conn()
            .execute(
                """
            SELECT config_yaml, target_metric_value
            FROM attempts
            WHERE dataset_hash = ? AND report_json != ''
            ORDER BY id ASC
            """,
                (dataset_hash,),
            )
            .fetchall()
        )
        configs: list[str] = [str(row[0]) for row in rows]
        metrics: list[float] = [float(row[1]) for row in rows]
        return configs, metrics

    def latest_run_id(self) -> str | None:
        """Return the most recently started run id, if any runs exist."""
        row = (
            self._conn()
            .execute(
                """
            SELECT run_id
            FROM runs
            ORDER BY started_at DESC, run_id DESC
            LIMIT 1
            """
            )
            .fetchone()
        )
        if row is None:
            return None
        return str(row[0])

    def get_run(self, run_id: str) -> dict[str, object] | None:
        """Return metadata for one run id, or None when it is absent."""
        row = (
            self._conn()
            .execute(
                """
            SELECT
                run_id, strategy_name, program_path, provider, model,
                branch, dataset_hash, iterations, started_at
            FROM runs
            WHERE run_id = ?
            """,
                (run_id,),
            )
            .fetchone()
        )
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

        rows = (
            self._conn()
            .execute(
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
            )
            .fetchall()
        )

        schema = [
            ("strategy_name", str),
            ("run_id", str),
            ("iteration", int),
            ("observed_sharpe", float),
            ("deflated_sharpe", float),
            ("in_sample_max_drawdown", float),
            ("in_sample_turnover", float),
            ("created_at", str),
            ("target_metric", str),
            ("target_metric_value", float),
            ("accepted", bool),
            ("committed", bool),
            ("rejection_reason", lambda x: str(x) if x is not None else None),
            ("prompt_tokens", lambda x: int(x) if x is not None else 0),
            ("completion_tokens", lambda x: int(x) if x is not None else 0),
            ("total_tokens", lambda x: int(x) if x is not None else 0),
            ("cost", lambda x: float(x) if x is not None else 0.0),
        ]
        return [self._marshal_row(row, schema) for row in rows]

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

        rows = (
            self._conn()
            .execute(
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
            )
            .fetchall()
        )

        schema = [
            ("strategy_name", str),
            ("run_id", str),
            ("iteration", int),
            ("observed_sharpe", float),
            ("deflated_sharpe", float),
            ("in_sample_max_drawdown", float),
            ("in_sample_turnover", float),
            ("created_at", str),
            ("target_metric", str),
            ("target_metric_value", float),
        ]
        return [self._marshal_row(row, schema) for row in rows]

    def list_runs(self) -> list[dict[str, object]]:
        """Return metadata for all recorded runs."""
        query = """
            SELECT
                run_id, strategy_name, program_path, provider, model,
                branch, dataset_hash, iterations, started_at
            FROM runs
            ORDER BY started_at DESC
        """
        rows = self._conn().execute(query).fetchall()
        schema = [
            ("run_id", str),
            ("strategy_name", str),
            ("program_path", str),
            ("provider", str),
            ("model", str),
            ("branch", str),
            ("dataset_hash", str),
            ("iterations", int),
            ("started_at", str),
        ]
        return [self._marshal_row(row, schema) for row in rows]

    def close(self) -> None:
        """Close all per-thread database connections."""
        with self._lock:
            for c in self._local.values():
                c.close()
            self._local.clear()
