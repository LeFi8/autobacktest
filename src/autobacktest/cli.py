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
from autobacktest.templates import TEMPLATE_REGISTRY

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
    universe: str | None = typer.Option(
        None,
        "--universe",
        "-u",
        help="Comma-separated asset tickers (e.g. SPY,TLT,GLD,BIL). "
        "When provided, all prompts are skipped (silent mode).",
    ),
    benchmark: str = typer.Option(
        "SPY",
        "--benchmark",
        "--bench",
        help="Benchmark index ticker.",
    ),
    max_drawdown: float = typer.Option(
        0.20,
        "--drawdown",
        "--mdd",
        help="Maximum drawdown limit (0.0 to 1.0).",
    ),
    turnover: float = typer.Option(
        2.0,
        "--turnover",
        help="Annualized turnover limit.",
    ),
    lookback: int = typer.Option(
        12,
        "--lookback",
        "--mom-lookback",
        help="Momentum lookback window in months.",
    ),
    template: str = typer.Option(
        "equal-weight",
        "--template",
        help=f"Strategy template: {', '.join(sorted(TEMPLATE_REGISTRY))}.",
    ),
    cash_asset: str = typer.Option(
        "BIL",
        "--cash-asset",
        help="Cash/risk-free asset ticker.",
    ),
) -> None:
    """Scaffold a new strategy with Pydantic-validated boilerplate code.

    When --universe is provided, runs in silent (non-interactive) mode.
    All other options can be mixed: omitted values use their defaults.
    """
    init_strategy_impl(
        name=name,
        overwrite=overwrite,
        silent_universe=universe,
        silent_benchmark=benchmark,
        silent_max_drawdown=max_drawdown,
        silent_turnover=turnover,
        silent_lookback=lookback,
        silent_template=template,
        silent_cash_asset=cash_asset,
        settings_obj=settings,
    )


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
    """Restore strategy baseline files and delete the runs directory."""
    from autobacktest.ledger.git_ops import GitLedger

    reset_impl(
        strategy=strategy,
        run_dir=run_dir,
        path_class=Path,
        git_ledger_class=GitLedger,
    )


def main() -> None:
    """Entry point for the ``autobacktest`` console script registered in ``pyproject.toml``."""
    app()


if __name__ == "__main__":
    main()
