"""Structured JSON events log for iteration-level audit trails.

Each optimisation iteration emits one JSON line to an ``events.jsonl`` file
inside the run directory.  The log is consumed by ``compile_failure_summary``
in the reporting module.
"""

from __future__ import annotations

import contextlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
    """Append-only structured JSON events log.

    Writes one JSON record per iteration to ``events.jsonl``.
    Each record captures the iteration number, mode, temperature,
    candidate outcomes (pass/fail reasons), gate verdicts, and
    commit SHAs.  Failure summaries are later derived from this log.

    Usage as context manager::

        with EventLog(path) as log:
            log.write({"iteration": 1, "mode": "explore", ...})
    """

    def __init__(self, path: Path) -> None:
        """Create parent dirs and open file for append."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._file = path.open("a", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        """Append one JSON line. Adds 'timestamp' key if not present."""
        if "timestamp" not in record:
            record["timestamp"] = datetime.now(tz=timezone.utc).isoformat()  # noqa: UP017
        self._file.write(json.dumps(record) + "\n")
        self._file.flush()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()

    def close(self) -> None:
        """Close the underlying file handle. Safe to call multiple times."""
        self._file.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
