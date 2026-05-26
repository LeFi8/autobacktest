from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProgramSpec:
    objective: str  # text under # Objective
    constraints: str  # text under # Constraints
    raw_text: str  # full file content (passed to LLM as-is)


def parse_program(path: Path) -> ProgramSpec:
    """Parse a program.md file with required # Objective and # Constraints headers.

    Raises ValueError if either header is missing.
    """
    raw_text = path.read_text(encoding="utf-8")

    # Extract text under each H1 header (up to next H1 or EOF)
    sections: dict[str, str] = {}
    pattern = re.compile(r"^# (.+)$", re.MULTILINE)
    matches = list(pattern.finditer(raw_text))

    for i, match in enumerate(matches):
        header = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(raw_text)
        sections[header] = raw_text[start:end].strip()

    if "Objective" not in sections:
        raise ValueError(
            f"program.md at {path} is missing required '# Objective' header"
        )
    if "Constraints" not in sections:
        raise ValueError(
            f"program.md at {path} is missing required '# Constraints' header"
        )

    return ProgramSpec(
        objective=sections["Objective"],
        constraints=sections["Constraints"],
        raw_text=raw_text,
    )
