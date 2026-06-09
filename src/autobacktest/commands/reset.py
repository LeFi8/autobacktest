"""CLI command 'reset' implementation."""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import typer

from autobacktest.ledger.git_ops import GitLedger as DefaultGitLedger

logger = logging.getLogger(__name__)


def _delete_run_dir(run_dir: str, path_class: type[Path], cleanup_failures: list[str]) -> None:
    """Delete the run artifacts directory."""
    run_path = path_class(run_dir)
    if run_path.exists():
        try:
            shutil.rmtree(run_path)
            typer.echo(f"Run directory '{run_dir}' deleted entirely.")
        except Exception as e:
            typer.echo(f"Error deleting run directory '{run_dir}': {e}")
            cleanup_failures.append(f"run directory '{run_dir}'")
    else:
        typer.echo(f"Run directory '{run_dir}' does not exist, skipping deletion.")


def reset_impl(
    strategy: str | None,
    run_dir: str,
    path_class: type[Path] = Path,
    git_ledger_class: type[DefaultGitLedger] = DefaultGitLedger,
) -> None:
    """Reset strategy baseline files and delete the runs directory.

    Args:
        strategy: Strategy name to reset (``None`` resets all).
        run_dir: Path to the run artifacts directory to delete.
        path_class: Path class (injected for testability).
        git_ledger_class: Git ledger class (injected for testability).
    """
    cleanup_failures: list[str] = []

    try:
        git_ledger = git_ledger_class(path_class())
        git_ledger.reset_to_main(strategy)
        typer.echo("Baseline strategy files restored successfully.")
    except Exception as e:
        typer.echo(f"Error: Failed to reset workspace via git: {e}")
        typer.echo("Abort: Reset could not be completed safely.")
        raise typer.Exit(code=1) from e

    _delete_run_dir(run_dir, path_class, cleanup_failures)

    if cleanup_failures:
        typer.echo("Reset failed.")
        raise typer.Exit(code=1)

    typer.echo("Reset completed.")
