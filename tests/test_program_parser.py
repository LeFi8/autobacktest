import pytest
from pathlib import Path
from autobacktest.program import parse_program, ProgramSpec


def test_parse_valid_program(tmp_path: Path) -> None:
    p = tmp_path / "program.md"
    p.write_text(
        "# Objective\nMaximize risk-adjusted returns.\n\n"
        "# Constraints\nMax drawdown 15%. Turnover limit 2x.",
        encoding="utf-8",
    )
    spec = parse_program(p)
    assert isinstance(spec, ProgramSpec)
    assert "risk-adjusted" in spec.objective
    assert "15%" in spec.constraints
    assert spec.raw_text.startswith("# Objective")


def test_parse_missing_constraints_raises(tmp_path: Path) -> None:
    p = tmp_path / "program.md"
    p.write_text("# Objective\nSome objective.", encoding="utf-8")
    with pytest.raises(ValueError, match="Constraints"):
        parse_program(p)


def test_parse_missing_objective_raises(tmp_path: Path) -> None:
    p = tmp_path / "program.md"
    p.write_text("# Constraints\nSome constraints.", encoding="utf-8")
    with pytest.raises(ValueError, match="Objective"):
        parse_program(p)


def test_parse_extra_sections_ignored(tmp_path: Path) -> None:
    p = tmp_path / "program.md"
    p.write_text(
        "# Objective\nGoal.\n\n# Constraints\nLimits.\n\n# Notes\nExtra.",
        encoding="utf-8",
    )
    spec = parse_program(p)
    assert spec.objective == "Goal."
    assert spec.constraints == "Limits."


def test_raw_text_is_full_content(tmp_path: Path) -> None:
    content = "# Objective\nX.\n\n# Constraints\nY."
    p = tmp_path / "program.md"
    p.write_text(content, encoding="utf-8")
    assert parse_program(p).raw_text == content
