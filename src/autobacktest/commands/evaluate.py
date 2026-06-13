"""CLI command 'evaluate' implementation."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import typer

from autobacktest import configure_verbosity
from autobacktest.config import settings
from autobacktest.evaluator.evaluate import evaluate_strategy
from autobacktest.strategy.config_schema import StrategyConfig


def register_command(app: typer.Typer) -> None:
    @app.command()
    def evaluate(
        strategy: str = typer.Option(
            ...,
            "--strategy",
            "-s",
            help="Path to strategy file (e.g., strategies/haa.py).",
        ),
        start_date: str = typer.Option(
            settings.default_start_date,
            "--start-date",
            help="Start date YYYY-MM-DD for backtesting.",
        ),
        end_date: str = typer.Option(
            settings.default_end_date,
            "--end-date",
            help="End date YYYY-MM-DD for backtesting.",
        ),
        quiet: bool = typer.Option(
            settings.quiet,
            "--quiet",
            "-q",
            help="Suppress non-critical warnings and reduce terminal noise.",
        ),
    ) -> None:
        """Run walk-forward and holdout evaluation on a target strategy file."""
        configure_verbosity(quiet=quiet)
        strategy_path = Path(strategy)
        if not strategy_path.exists():
            typer.echo(f"Error: Strategy file not found at {strategy_path}")
            raise typer.Exit(code=1)

        resolved_path = strategy_path.resolve()
        strategy_name = resolved_path.parent.name if resolved_path.stem == "strategy" else resolved_path.stem
        config_path = resolved_path.parent / "config.yaml"
        if not config_path.exists():
            config_path = resolved_path.parent / f"{strategy_name}.yaml"
        if not config_path.exists():
            config_path = settings.configs_dir / f"{strategy_name}.yaml"
        if not config_path.exists():
            config_path = resolved_path.parent.parent / "configs" / f"{strategy_name}.yaml"

        if not config_path.exists():
            typer.echo(f"Error: Strategy config file not found at {config_path}")
            raise typer.Exit(code=1)

        try:
            strategy_config = StrategyConfig.from_yaml(config_path)
            config = strategy_config.model_dump()
        except Exception as e:
            typer.echo(f"Error: Strategy config file is invalid: {e}")
            raise typer.Exit(code=1) from e

        spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
        if spec is None or spec.loader is None:
            typer.echo(f"Error: Failed to construct loader for {strategy_path}")
            raise typer.Exit(code=1)

        module = importlib.util.module_from_spec(spec)
        try:
            typer.echo(f"WARNING: Dynamically executing external python module: {strategy_path.resolve()}")
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
