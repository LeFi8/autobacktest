"""Tests for the CLI reset subcommand."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from autobacktest.cli import app

runner = CliRunner()


def test_reset_command_flow(tmp_path: Path) -> None:
    """Reset command restores files to baseline and deletes runs/."""
    # Setup folders
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs"
    run_dir = tmp_path / "runs"
    strat_dir.mkdir()
    cfg_dir.mkdir()
    run_dir.mkdir()

    # Patch GitLedger to mock repository interaction
    with patch("autobacktest.ledger.git_ops.GitLedger") as mock_git_ledger:
        ledger_instance = mock_git_ledger.return_value
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
    assert f"Run directory '{run_dir}' deleted entirely" in result.output

    # Assert runs/ was deleted
    assert not run_dir.exists()


def test_reset_rmtree_failure_exits_nonzero(tmp_path: Path) -> None:
    """Reset reports failure when run directory deletion fails."""
    run_dir = tmp_path / "runs"
    run_dir.mkdir()

    with (
        patch("autobacktest.ledger.git_ops.GitLedger") as mock_git_ledger,
        patch("shutil.rmtree", side_effect=OSError("cannot delete")),
    ):
        mock_git_ledger.return_value.repo_root = tmp_path

        result = runner.invoke(
            app,
            ["reset", "--strategy", "haa", "--run-dir", str(run_dir)],
        )

    assert result.exit_code == 1
    assert "Error deleting run directory" in result.output
    assert "Reset failed." in result.output
    assert "Reset completed." not in result.output
    assert run_dir.exists()
