"""SQLite-backed lesson store with deduplication and type-aware filtering.

Replaces the flat ``lessons.md`` file with a structured database keyed by
``(strategy, type, body_hash)`` for cross-run learning and automatic
deduplication.
"""

from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from typing import Any

from autobacktest.config import settings
from autobacktest.llm.prompts import parse_lessons

_CREATE_LESSONS = """
CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    strategy TEXT NOT NULL,
    type TEXT NOT NULL,
    body_hash TEXT NOT NULL,
    title TEXT NOT NULL,
    body TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'llm',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(strategy, type, body_hash)
)
"""

_CREATE_STRATEGY_IDX = "CREATE INDEX IF NOT EXISTS idx_lessons_strategy ON lessons(strategy)"
_CREATE_TYPE_IDX = "CREATE INDEX IF NOT EXISTS idx_lessons_type ON lessons(type)"


class LessonStore:
    """Persist, deduplicate, and retrieve lessons in a local SQLite database.

    Lessons are parsed from the markdown format the LLM produces, then stored
    keyed by ``(strategy, type, body_hash)`` so that identical lessons are
    automatically skipped on re-insertion.

    Thread safety: each connection uses WAL mode.  The class creates a fresh
    connection per thread when the ``connection`` property is accessed.
    """

    def __init__(self, db_path: Path | str | None = None) -> None:
        self._db_path = Path(db_path) if db_path else settings.lessons_db_path
        self._local: dict[int, sqlite3.Connection] = {}

    # ------------------------------------------------------------------
    # Connection management (one per thread)
    # ------------------------------------------------------------------

    def _conn(self) -> sqlite3.Connection:
        tid = _current_thread_id()
        if tid not in self._local:
            c = sqlite3.connect(str(self._db_path), timeout=settings.db_timeout)
            c.execute("PRAGMA journal_mode=WAL")
            c.execute(_CREATE_LESSONS)
            c.execute(_CREATE_STRATEGY_IDX)
            c.execute(_CREATE_TYPE_IDX)
            c.commit()
            self._local[tid] = c
        return self._local[tid]

    def close(self) -> None:
        for c in self._local.values():
            c.close()
        self._local.clear()

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def store_lesson(
        self,
        strategy: str,
        title: str,
        body: str,
        lesson_type: str = "STRUCTURAL",
        source: str = "llm",
    ) -> bool:
        """Insert a single lesson, deduplicated by ``(strategy, type, body_hash)``.

        Returns:
            True if a new row was inserted, False if it already existed.
        """
        body_hash = _hash_body(body)
        try:
            self._conn().execute(
                """
                INSERT INTO lessons (strategy, type, body_hash, title, body, source)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (strategy, lesson_type, body_hash, title, body, source),
            )
            self._conn().commit()
            return True
        except sqlite3.IntegrityError:
            return False

    def ingest_markdown(self, markdown_text: str, strategy: str, source: str = "llm") -> int:
        """Parse markdown lessons and insert them with deduplication.

        Args:
            markdown_text: Raw markdown in the format the LLM produces
                (``### Title`` + ``**Type:** ENUM`` pattern).
            strategy: Strategy name to associate with the lessons.
            source: Source label (default ``"llm"``).

        Returns:
            Count of newly inserted lessons.
        """
        if not markdown_text or not markdown_text.strip():
            return 0

        parsed = parse_lessons(markdown_text)
        inserted = 0
        for lesson in parsed:
            if self.store_lesson(
                strategy=strategy,
                title=lesson["title"],
                body=lesson["body"],
                lesson_type=lesson["type"],
                source=source,
            ):
                inserted += 1
        return inserted

    def get_filtered_markdown(self, strategy: str, context_stage: str | None = None) -> str:
        """Retrieve lessons for *strategy*, optionally filtered by *context_stage*.

        When *context_stage* matches a known type (validation → BUG,
        diversity_config → DIVERSITY, gate → GATE_REJECTION) only lessons of
        that type plus STRUCTURAL are returned.  Otherwise all lessons are
        returned.

        Returns:
            Re-rendered markdown string, or ``"No lessons recorded yet."`` when
            the result is empty.
        """
        target_type = _stage_to_type(context_stage)

        if target_type:
            rows = self._conn().execute(
                """
                SELECT title, body FROM lessons
                WHERE strategy = ? AND (type = ? OR type = 'STRUCTURAL')
                ORDER BY id ASC
                """,
                (strategy, target_type),
            ).fetchall()
        else:
            rows = self._conn().execute(
                """
                SELECT title, body FROM lessons
                WHERE strategy = ?
                ORDER BY id ASC
                """,
                (strategy,),
            ).fetchall()

        if not rows:
            return "No lessons recorded yet."

        blocks = []
        for title, body in rows:
            blocks.append(f"### {title}\n{body}")
        return "\n\n".join(blocks)

    def count(self, strategy: str | None = None) -> int:
        """Return total number of lessons, optionally filtered by *strategy*."""
        if strategy:
            row = self._conn().execute(
                "SELECT COUNT(*) FROM lessons WHERE strategy = ?", (strategy,)
            ).fetchone()
        else:
            row = self._conn().execute("SELECT COUNT(*) FROM lessons").fetchone()
        return row[0] if row else 0

    def all_lessons(self, strategy: str | None = None) -> list[dict[str, Any]]:
        """Return all lessons as dicts, optionally filtered by *strategy*."""
        if strategy:
            rows = self._conn().execute(
                """
                SELECT strategy, type, title, body, source, created_at
                FROM lessons WHERE strategy = ?
                ORDER BY id ASC
                """,
                (strategy,),
            ).fetchall()
        else:
            rows = self._conn().execute(
                """
                SELECT strategy, type, title, body, source, created_at
                FROM lessons ORDER BY id ASC
                """
            ).fetchall()

        return [
            {
                "strategy": r[0],
                "type": r[1],
                "title": r[2],
                "body": r[3],
                "source": r[4],
                "created_at": r[5],
            }
            for r in rows
        ]

    def migrate_from_file(self, file_path: Path, strategy: str) -> int:
        """Migrate lessons from an existing ``lessons.md`` file into the store.

        Returns:
            Number of new lessons inserted.
        """
        if not file_path.exists():
            return 0
        text = file_path.read_text(encoding="utf-8")
        return self.ingest_markdown(text, strategy=strategy, source="migration")


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _hash_body(body: str) -> str:
    return hashlib.sha256(body.strip().encode("utf-8")).hexdigest()


def _current_thread_id() -> int:
    """Return a hashable id for the current thread."""
    import threading
    return threading.get_ident()


def _stage_to_type(context_stage: str | None) -> str | None:
    mapping = {
        "validation": "BUG",
        "eval_error": "BUG",
        "diversity_config": "DIVERSITY",
        "diversity_returns": "DIVERSITY",
        "gate": "GATE_REJECTION",
    }
    return mapping.get(context_stage) if context_stage else None
