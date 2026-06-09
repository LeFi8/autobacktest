"""Strategy template registry and renderers.

Templates are stored as runnable example files in well-known project
locations (``strategies/``, ``configs/``, project root).  The renderers
copy those files and substitute sentinel values.

Template source files are resolved relative to this package so they work
regardless of ``strategies_dir`` / ``configs_dir`` settings.
"""

from __future__ import annotations

import os
from pathlib import Path


def _find_project_root(marker: str = "pyproject.toml") -> Path:
    candidates: list[Path] = []

    candidates.append(Path(__file__).resolve().parent)
    if "AUTOBACKTEST_PROJECT_ROOT" in os.environ:
        candidates.append(Path(os.environ["AUTOBACKTEST_PROJECT_ROOT"]))
    candidates.append(Path.cwd())

    seen: set[Path] = set()
    for start in candidates:
        current = start
        for _ in range(20):
            resolved = current.resolve()
            if resolved in seen:
                break
            seen.add(resolved)
            if (resolved / marker).exists():
                return resolved
            current = current.parent

    msg = f"Could not find project root (no {marker} found ascending from {__file__}, CWD={Path.cwd()})"
    raise RuntimeError(msg)


_PACKAGE_ROOT = _find_project_root()

TEMPLATE_REGISTRY: dict[str, dict[str, str]] = {
    "equal-weight": {
        "strategy": "equal_weight.py",
        "program": "program-equal_weight.md",
    },
    "momentum-rotation": {
        "strategy": "momentum_rotation.py",
        "program": "program-momentum_rotation.md",
    },
}

_FILE_TYPE_DIRS: dict[str, Path] = {
    "strategy": _PACKAGE_ROOT / "strategies",
    "program": _PACKAGE_ROOT,
}


def _lookup(template_name: str) -> dict[str, str]:
    """Resolve a template name to its file mapping, or raise."""
    entry = TEMPLATE_REGISTRY.get(template_name)
    if entry is None:
        valid = ", ".join(sorted(TEMPLATE_REGISTRY))
        raise ValueError(f"Unknown template '{template_name}'. Valid options: {valid}")
    return entry


def _source_path(template_name: str, file_type: str) -> Path:
    """Canonical path to a template source file."""
    entry = _lookup(template_name)
    return _FILE_TYPE_DIRS[file_type] / entry[file_type]


def render_strategy_source(
    strategy_name: str,
    cash_asset: str,
    template_name: str,
) -> str:
    """Load and render a strategy ``.py`` template.

    Replaces the template's hardcoded name in the docstring with
    *strategy_name*, and its ``"BIL"`` default with *cash_asset* when
    *cash_asset* differs from ``"BIL"``.

    Args:
        strategy_name: Normalised snake_case strategy name.
        cash_asset: Cash/risk-free asset ticker.
        template_name: Key into ``TEMPLATE_REGISTRY``.

    Returns:
        str: Rendered Python source code.
    """
    source = _source_path(template_name, "strategy").read_text(encoding="utf-8")

    # Replace the first occurrence of the template name in the docstring.
    entry = _lookup(template_name)
    template_stem = entry["strategy"].removesuffix(".py")
    source = source.replace(template_stem, strategy_name, 1)

    if cash_asset != "BIL":
        source = source.replace('"BIL"', f'"{cash_asset}"')

    return source


def _substitute(text: str, subs: dict[str, str]) -> str:
    """Replace all sentinel keys in *text* with their values."""
    for key, val in subs.items():
        text = text.replace(key, val)
    return text


def render_program_template(
    template_name: str,
    strategy_name: str,
    universe_str: str,
    benchmark: str,
    mdd: float,
    turnover: float,
    lookback: int,
) -> str:
    """Load and render a program ``.md`` template.

    Substitutes all sentinel values into the template.
    """
    source = _source_path(template_name, "program").read_text(encoding="utf-8")

    subs: dict[str, str] = {
        "__NAME__": strategy_name,
        "__UNIVERSE__": universe_str,
        "__BENCHMARK__": benchmark,
        "__DRAWDOWN_PCT__": str(round(mdd * 100)),
        "__TURNOVER__": str(turnover),
        "__LOOKBACK__": str(lookback),
    }
    return _substitute(source, subs)
