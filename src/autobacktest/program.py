"""Program specification parser for LLM objective/constraint files.

Parses ``program.md`` (or equivalent) extracting ``# Objective`` and
``# Constraints`` sections.  Fenced code blocks are skipped so that
example comments inside code fences are not mistaken for section headers.
Returns a ``ProgramSpec`` dataclass passed verbatim to the LLM provider.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ProgramSpec:
    """Dataclass holding structured objectives and constraints from a program file.

    Attributes:
        objective: The extracted objectives text under the '# Objective' header.
        constraints: The extracted constraints text under the '# Constraints' header.
        raw_text: The entire raw contents of the program file.
    """

    objective: str  # text under # Objective
    constraints: str  # text under # Constraints
    raw_text: str  # full file content (passed to LLM as-is)


def parse_program(path: Path) -> ProgramSpec:
    """Parse a program.md file with required # Objective and # Constraints headers.

    Skips fenced code blocks to prevent false header matching.

    Args:
        path: Path to the markdown program objective file.

    Returns:
        ProgramSpec: The structured and validated objectives/constraints.

    Raises:
        ValueError: If either '# Objective' or '# Constraints' headers are missing.
    """
    raw_text = path.read_text(encoding="utf-8")

    # Walk line by line to find H1 headers, skipping fenced code blocks so that
    # lines like `# comment` inside ``` fences are not treated as headers.
    header_re = re.compile(r"^# (.+)$")
    in_fence = False
    # Each entry: (name, line_start, body_start) where line_start is the offset
    # of the `# Header` line and body_start is right after its trailing newline.
    headers: list[tuple[str, int, int]] = []
    pos = 0
    for line in raw_text.splitlines(keepends=True):
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
        elif not in_fence:
            m = header_re.match(line.rstrip("\r\n"))
            if m:
                headers.append((m.group(1).strip(), pos, pos + len(line)))
        pos += len(line)

    sections: dict[str, str] = {}
    for i, (name, _line_start, body_start) in enumerate(headers):
        next_line_start = headers[i + 1][1] if i + 1 < len(headers) else len(raw_text)
        sections[name] = raw_text[body_start:next_line_start].strip()

    if "Objective" not in sections:
        raise ValueError(f"program.md at {path} is missing required '# Objective' header")
    if "Constraints" not in sections:
        raise ValueError(f"program.md at {path} is missing required '# Constraints' header")

    return ProgramSpec(
        objective=sections["Objective"],
        constraints=sections["Constraints"],
        raw_text=raw_text,
    )
