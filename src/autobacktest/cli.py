"""Command-line interface for AutoBacktest."""

import importlib.util
from pathlib import Path

import typer

from autobacktest.evaluator.evaluate import evaluate_strategy

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


@app.command()
def evaluate(
    strategy: str = typer.Option(
        ...,
        "--strategy",
        "-s",
        help="Path to strategy file (e.g., strategies/haa.py).",
    ),
    start_date: str = typer.Option(
        "2015-01-01",
        "--start-date",
        help="Start date YYYY-MM-DD for backtesting.",
    ),
    end_date: str = typer.Option(
        "2026-01-01",
        "--end-date",
        help="End date YYYY-MM-DD for backtesting.",
    ),
) -> None:
    """Run walk-forward and holdout evaluation on a target strategy file."""
    strategy_path = Path(strategy)
    if not strategy_path.exists():
        typer.echo(f"Error: Strategy file not found at {strategy_path}")
        raise typer.Exit(code=1)

    strategy_name = strategy_path.stem
    config_path = Path("configs") / f"{strategy_name}.yaml"
    if not config_path.exists():
        # Fallback to strategy_path's parent relative directory configs
        config_path = (
            strategy_path.resolve().parent.parent / "configs" / f"{strategy_name}.yaml"
        )

    if not config_path.exists():
        typer.echo(f"Error: Strategy config file not found at {config_path}")
        raise typer.Exit(code=1)

    # Load and validate YAML config via Pydantic StrategyConfig
    from autobacktest.strategy.config_schema import StrategyConfig

    try:
        strategy_config = StrategyConfig.from_yaml(config_path)
        config = strategy_config.model_dump()
    except Exception as e:
        typer.echo(f"Error: Strategy config file is invalid: {e}")
        raise typer.Exit(code=1) from e

    # Dynamically import strategy signals generator
    spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
    if spec is None or spec.loader is None:
        typer.echo(f"Error: Failed to construct loader for {strategy_path}")
        raise typer.Exit(code=1)

    module = importlib.util.module_from_spec(spec)
    try:
        # Security log warning for execution of dynamic third-party strategy files
        typer.echo(
            "WARNING: Dynamically executing external python module: "
            f"{strategy_path.resolve()}"
        )
        spec.loader.exec_module(module)
    except Exception as e:
        typer.echo(f"Error: Failed to execute strategy file module: {e}")
        raise typer.Exit(code=1) from e

    if not hasattr(module, "generate_signals"):
        typer.echo("Error: Strategy module must export a generate_signals function.")
        raise typer.Exit(code=1)

    typer.echo(f"Initializing evaluation of strategy: {strategy_name}...")
    report_data = evaluate_strategy(
        strategy_name,
        module.generate_signals,
        config,
        start_date=start_date,
        end_date=end_date,
    )

    typer.echo("--- Evaluation Report ---")
    typer.echo(report_data.to_json())


def main() -> None:
    """Entry point for the console script."""
    app()


if __name__ == "__main__":
    main()
