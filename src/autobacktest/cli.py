"""AutoBacktest CLI — Typer-based command-line interface.

Provides seven subcommands, delegating to implementation modules.
Exposes global variables for test compatibility and patching.
"""

from __future__ import annotations

from pathlib import Path

import typer

from autobacktest.commands.evaluate import register_command as register_evaluate
from autobacktest.commands.init import init_strategy_impl
from autobacktest.commands.llm_test import register_command as register_llm_test
from autobacktest.commands.report import register_command as register_report
from autobacktest.commands.reset import reset_impl
from autobacktest.commands.run import register_command as register_run
from autobacktest.commands.spa import register_command as register_spa
from autobacktest.config import settings

app = typer.Typer(
    name="autobacktest",
    help="Autonomous AI-driven strategy optimization loop.",
    no_args_is_help=True,
)

# Register subcommands that don't need mocking delegation or have internal registrations
register_run(app)
register_report(app)
register_evaluate(app)
register_llm_test(app)
register_spa(app)


# Register subcommands that are explicitly mocked on autobacktest.cli namespace
@app.command(name="init-strategy")
def init_strategy(
    name: str = typer.Option(
        None,
        "--name",
        "-n",
        help="Strategy name (snake_case). Prompts interactively if omitted.",
    ),
    overwrite: bool = typer.Option(
        False,
        "--overwrite",
        help="Overwrite existing strategy/config files without prompting.",
    ),
) -> None:
    """Interactively set up a new backtesting strategy config and boilerplate code."""
    init_strategy_impl(name, overwrite, settings_obj=settings)


@app.command()
def reset(
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Strategy name to reset. Resets all if not specified.",
    ),
    run_dir: str = typer.Option(
        str(settings.run_dir),
        "--run-dir",
        help="Path to runs directory to be deleted.",
    ),
) -> None:
    """Restore strategy baseline files, clear lessons, and delete the runs directory."""
    from autobacktest.ledger.git_ops import GitLedger

    reset_impl(
        strategy=strategy,
        run_dir=run_dir,
        path_class=Path,
        git_ledger_class=GitLedger,
        settings_obj=settings,
    )


def main() -> None:
    """Entry point for the ``autobacktest`` console script registered in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()
