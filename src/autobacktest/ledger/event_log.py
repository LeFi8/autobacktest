from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLog:
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

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> EventLog:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
