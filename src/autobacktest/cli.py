"""Command-line interface for AutoBacktest."""

import typer

app = typer.Typer(
    name="autobacktest",
    help="Autonomous AI-driven strategy optimization loop.",
    no_args_is_help=True,
)


@app.command()
def run(
    program: str = typer.Option(
        ...,
        "--program",
        "-p",
        help="Path to program.md objective and constraints.",
    ),
    strategy: str = typer.Option(
        ...,
        "--strategy",
        "-s",
        help="Name of the strategy in the registry.",
    ),
    iterations: int = typer.Option(
        5,
        "--iterations",
        "-i",
        help="Number of optimization iterations to run.",
    ),
    provider: str | None = typer.Option(
        None,
        "--provider",
        help="LLM provider (e.g. anthropic, openai, google).",
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        help="LLM model name to run.",
    ),
    run_dir: str | None = typer.Option(
        None,
        "--run-dir",
        help="Directory to store runs and SQLite ledger.",
    ),
) -> None:
    """Run the optimization loop on a strategy."""
    typer.echo(
        f"TODO: run subcommand (program={program}, strategy={strategy}, "
        f"iterations={iterations}, provider={provider}, model={model}, "
        f"run_dir={run_dir})"
    )


@app.command()
def report(
    compare_all: bool = typer.Option(
        False,
        "--compare-all",
        help="Compare all registry strategies side-by-side.",
    ),
) -> None:
    """Print the run leaderboard from the SQLite ledger."""
    typer.echo(f"TODO: report subcommand (compare_all={compare_all})")


@app.command()
def reset(
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Strategy name to reset. Resets all if not specified.",
    ),
) -> None:
    """Clean the run directory and reset target strategies to baseline."""
    typer.echo(f"TODO: reset subcommand (strategy={strategy})")


def main() -> None:
    """Entry point for the console script."""
    app()


if __name__ == "__main__":
    main()
