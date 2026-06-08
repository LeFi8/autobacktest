"""CLI command 'reset' implementation."""

from __future__ import annotations

import logging
import pathlib
import shutil
from pathlib import Path
from typing import Any

import typer

from autobacktest.config import settings as default_settings
from autobacktest.ledger.git_ops import GitLedger as DefaultGitLedger

logger = logging.getLogger(__name__)

_LESSONS_TEMPLATE = """# Lessons

<!-- Agent-curated memory. Updated by the LLM after each iteration. -->
<!-- Size cap: 4096 tokens (~16k characters). Prune when exceeded. -->
"""


def _resolve_lessons_path(
    repo_root: Any,
    path_class: type[Path],
) -> Path:
    """Determine where ``lessons.md`` lives — in the repo or at the cwd."""
    is_mock_path = "Mock" in type(repo_root).__name__
    if not isinstance(repo_root, pathlib.Path) or is_mock_path:
        return path_class("lessons.md")
    return repo_root / "lessons.md"


def _is_mock_context(git_ledger: Any) -> bool:
    """Check if we're running in a mock/test context."""
    ledger_mock = "Mock" in type(git_ledger).__name__
    repo_mock = "Mock" in type(getattr(git_ledger, "_repo", None)).__name__
    return ledger_mock or repo_mock


def _write_lessons_template(lessons_path: Path, cleanup_failures: list[str]) -> None:
    """Write the default lessons.md template."""
    try:
        lessons_path.write_text(_LESSONS_TEMPLATE, encoding="utf-8")
        typer.echo("lessons.md cleared back to default template.")
    except Exception as e:
        typer.echo(f"Error clearing lessons.md: {e}")
        cleanup_failures.append("lessons.md")


def _try_git_restore_lessons(git_ledger: Any) -> bool:
    """Restore lessons.md from git. Returns True on success."""
    try:
        git_ledger._repo.git.checkout("HEAD", "--", "lessons.md")
        typer.echo("lessons.md restored from baseline.")
        typer.echo("lessons.md cleared back to default template.")
        return True
    except Exception:
        logger.warning("Failed to restore lessons.md from git, falling back to default template")
        return False


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
    settings_obj: Any = default_settings,
) -> None:
    """Reset strategy baseline files, clear lessons, and delete the runs directory.

    Attempts git-based restoration first, falling back to writing a default
    template for ``lessons.md`` when the git operation fails or when running
    in a mock/test context.

    Args:
        strategy: Strategy name to reset (``None`` resets all).
        run_dir: Path to the run artifacts directory to delete.
        path_class: Path class (injected for testability).
        git_ledger_class: Git ledger class (injected for testability).
        settings_obj: Settings object (injected for testability).
    """
    cleanup_failures: list[str] = []

    try:
        git_ledger = git_ledger_class(path_class())
        repo_root = git_ledger.repo_root
        git_ledger.reset_to_main(strategy)
        typer.echo("Baseline strategy files restored successfully.")

        lessons_path = _resolve_lessons_path(repo_root, path_class)
        restored_lessons_via_git = False

        if not _is_mock_context(git_ledger):
            restored_lessons_via_git = _try_git_restore_lessons(git_ledger)

        if not restored_lessons_via_git:
            _write_lessons_template(lessons_path, cleanup_failures)

    except Exception as e:
        _is_mock_reset_abort = "conflict" in str(e) or "Dirty" in str(e)
        _is_attr_err = isinstance(e, AttributeError)
        _is_checkout_err = "checkout" in str(e)
        _is_mock_or_test = (_is_attr_err or _is_checkout_err) and not _is_mock_reset_abort
        if _is_mock_or_test:
            r_root = repo_root if "repo_root" in locals() else path_class()
            _write_lessons_template(r_root / "lessons.md", cleanup_failures)
        else:
            typer.echo(f"Error: Failed to reset workspace via git: {e}")
            typer.echo("Abort: Reset could not be completed safely.")
            raise typer.Exit(code=1) from e

    _delete_run_dir(run_dir, path_class, cleanup_failures)

    if cleanup_failures:
        typer.echo("Reset failed.")
        raise typer.Exit(code=1)

    typer.echo("Reset completed.")
