"""Tests for the CLI reset subcommand."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

from typer.testing import CliRunner

from autobacktest.cli import app

runner = CliRunner()


def test_reset_command_flow(tmp_path: Path) -> None:
    """Reset command restores files to baseline, clears lessons, and deletes runs/."""
    # Setup folders
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs"
    run_dir = tmp_path / "runs"
    strat_dir.mkdir()
    cfg_dir.mkdir()
    run_dir.mkdir()

    # Create lessons.md in temporary working directory
    lessons_file = tmp_path / "lessons.md"
    lessons_file.write_text("# Lessons\n\n- Some lessons here.\n", encoding="utf-8")

    # Patch GitLedger to mock repository interaction
    with patch("autobacktest.ledger.git_ops.GitLedger") as mock_git_ledger:
        ledger_instance = mock_git_ledger.return_value

        def mock_path(*args: Any) -> Path:
            if not args:
                return tmp_path
            if args[0] == "lessons.md":
                return tmp_path / "lessons.md"
            return Path(*args)

        with patch("autobacktest.cli.Path", side_effect=mock_path):
            result = runner.invoke(
                app,
                [
                    "reset",
                    "--strategy",
                    "haa",
                    "--run-dir",
                    str(run_dir),
                ],
            )

        # Assert correct methods were called on GitLedger
        mock_git_ledger.assert_called_once()
        ledger_instance.reset_to_main.assert_called_once_with("haa")

    # Assert CLI output
    assert result.exit_code == 0
    assert "Baseline strategy files restored successfully" in result.output
    assert "lessons.md cleared back to default template" in result.output
    assert f"Run directory '{run_dir}' deleted entirely" in result.output

    # Assert runs/ was deleted
    assert not run_dir.exists()

    # Assert lessons.md was restored to default template
    lessons_content = lessons_file.read_text(encoding="utf-8")
    assert "# Lessons" in lessons_content
    assert "Agent-curated memory" in lessons_content
    assert "Some lessons here" not in lessons_content


def test_reset_rmtree_failure_exits_nonzero(tmp_path: Path) -> None:
    """Reset reports failure when run directory deletion fails."""
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    lessons_file = tmp_path / "lessons.md"
    lessons_file.write_text("# Lessons\n\n- Existing.\n", encoding="utf-8")

    with (
        patch("autobacktest.ledger.git_ops.GitLedger") as mock_git_ledger,
        patch("shutil.rmtree", side_effect=OSError("cannot delete")),
    ):
        ledger_instance = mock_git_ledger.return_value
        ledger_instance.repo_root = tmp_path

        result = runner.invoke(
            app,
            ["reset", "--strategy", "haa", "--run-dir", str(run_dir)],
        )

    assert result.exit_code == 1
    assert "Error deleting run directory" in result.output
    assert "Reset failed." in result.output
    assert "Reset completed." not in result.output
    assert run_dir.exists()


def test_reset_lessons_write_failure_exits_nonzero(tmp_path: Path) -> None:
    """Reset reports failure when lessons.md cannot be cleared."""
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    lessons_file = tmp_path / "lessons.md"
    lessons_file.write_text("# Saved lessons\n", encoding="utf-8")

    with (
        patch("autobacktest.ledger.git_ops.GitLedger") as mock_git_ledger,
        patch.object(Path, "write_text", side_effect=OSError("cannot write")),
    ):
        ledger_instance = mock_git_ledger.return_value
        ledger_instance.repo_root = tmp_path

        result = runner.invoke(
            app,
            ["reset", "--strategy", "haa", "--run-dir", str(run_dir)],
        )

    assert result.exit_code == 1
    assert "Error clearing lessons.md" in result.output
    assert "Reset failed." in result.output
    assert "Reset completed." not in result.output
    assert not run_dir.exists()
    assert lessons_file.read_text(encoding="utf-8") == "# Saved lessons\n"
