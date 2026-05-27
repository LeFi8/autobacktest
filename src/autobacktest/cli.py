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
    target_metric: str = typer.Option(
        "sharpe",
        "--target-metric",
        help="Target metric for optimization: sharpe, sortino, or information_ratio.",
    ),
) -> None:
    """Run the optimization loop on a strategy."""
    from autobacktest.gate import TargetMetric
    from autobacktest.llm.base import LLMProvider as _LLMProvider
    from autobacktest.llm.litellm_provider import LiteLLMProvider
    from autobacktest.llm.mock_provider import MockProvider
    from autobacktest.orchestrator import OrchestratorResult, run_optimization

    # Resolve target metric
    try:
        metric = TargetMetric(target_metric)
    except ValueError as err:
        error_msg = (
            f"Error: Unknown target metric '{target_metric}'. "
            "Use: sharpe, sortino, information_ratio."
        )
        typer.echo(error_msg)
        raise typer.Exit(code=1) from err

    # Resolve provider
    provider_impl: _LLMProvider
    if provider == "mock":
        provider_impl = MockProvider()
    else:
        # Prefix with provider unless the model already contains a slash.
        model_str = model or "gpt-4o"
        if provider and provider != "litellm" and "/" not in model_str:
            model_str = f"{provider}/{model_str}"
        provider_impl = LiteLLMProvider(model=model_str)

    # Resolve paths
    program_path = Path(program)
    run_dir_path = Path(run_dir) if run_dir else Path("runs")

    # Run optimization
    try:
        result: OrchestratorResult = run_optimization(
            program_path=program_path,
            strategy_name=strategy,
            iterations=iterations,
            provider=provider_impl,
            run_dir=run_dir_path,
            target_metric=metric,
        )
    except Exception as e:
        typer.echo(f"Error: {e}")
        raise typer.Exit(code=1) from e

    # Print results
    typer.echo("\nRun complete!")
    typer.echo(f"  Branch:    {result.branch}")
    typer.echo(f"  Committed: {result.n_committed} / {iterations}")
    typer.echo(f"  Run ID:    {result.run_id}")
    typer.echo("\n--- Final Report ---")
    typer.echo(result.final_report.to_json())


@app.command()
def report(
    run_dir: str = typer.Option(
        "runs",
        "--run-dir",
        help="Path to runs directory containing ledger.db.",
    ),
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Strategy name to filter by.",
    ),
    compare_all: bool = typer.Option(
        False,
        "--compare-all",
        help="Compare all registry strategies side-by-side.",
    ),
) -> None:
    """Print the run leaderboard from the SQLite ledger."""
    db_path = Path(run_dir) / "ledger.db"
    if not db_path.exists():
        typer.echo("No runs found (no ledger.db exists).")
        raise typer.Exit()

    from rich.console import Console
    from rich.table import Table

    from autobacktest.ledger.store import LedgerStore

    store = LedgerStore(db_path)
    try:
        if compare_all:
            rows = store.leaderboard(strategy_name=None)
        else:
            rows = store.leaderboard(strategy_name=strategy)

        if not rows:
            typer.echo("No runs found in ledger.")
            return

        console = Console()
        table = Table(title="AutoBacktest Optimization Leaderboard")
        table.add_column("Strategy", style="cyan", justify="left")
        table.add_column("Run ID", style="magenta", justify="left")
        table.add_column("Iter", style="blue", justify="center")
        table.add_column("Observed Sharpe", justify="right")
        table.add_column("Deflated Sharpe", style="yellow", justify="right")
        table.add_column("Max DD", style="red", justify="right")
        table.add_column("Turnover", style="cyan", justify="right")
        table.add_column("Date", style="dim", justify="center")

        from typing import cast

        for r in rows:
            sharpe_val = float(cast(float, r["observed_sharpe"]))
            sharpe_style = "green" if sharpe_val > 0 else "red"
            sharpe_str = f"[bold {sharpe_style}]{sharpe_val:.3f}[/]"

            deflated_val = float(cast(float, r["deflated_sharpe"]))
            deflated_str = f"{deflated_val:.3f}"

            max_dd_val = float(cast(float, r["holdout_max_drawdown"]))
            max_dd_str = f"{max_dd_val * 100:.2f}%"

            turnover_val = float(cast(float, r["holdout_turnover"]))
            turnover_str = f"{turnover_val:.2f}x"

            date_str = r["created_at"]
            if isinstance(date_str, str) and "T" in date_str:
                date_str = date_str.split("T")[0]

            table.add_row(
                str(r["strategy_name"]),
                str(r["run_id"]),
                str(r["iteration"]),
                sharpe_str,
                deflated_str,
                max_dd_str,
                turnover_str,
                str(date_str),
            )

        console.print(table)
    finally:
        store.close()


@app.command()
def reset(
    strategy: str | None = typer.Option(
        None,
        "--strategy",
        "-s",
        help="Strategy name to reset. Resets all if not specified.",
    ),
    run_dir: str = typer.Option(
        "runs",
        "--run-dir",
        help="Path to runs directory to be deleted.",
    ),
) -> None:
    """Restore strategy baseline files, clear lessons, and delete the runs directory."""
    import shutil

    from autobacktest.ledger.git_ops import GitLedger

    # 1. Reset strategy files to main baseline
    try:
        git_ledger = GitLedger(Path())
        git_ledger.reset_to_main(strategy)
        typer.echo("Baseline strategy files restored successfully.")
    except Exception as e:
        typer.echo(f"Error resetting strategy files via git: {e}")

    # 2. Restore lessons.md to empty template
    lessons_path = Path("lessons.md")
    template_content = """# Lessons

<!-- Agent-curated memory. Updated by the LLM after each iteration. -->
<!-- Size cap: 4096 tokens (~16k characters). Prune when exceeded. -->
"""
    try:
        lessons_path.write_text(template_content, encoding="utf-8")
        typer.echo("lessons.md cleared back to default template.")
    except Exception as e:
        typer.echo(f"Error clearing lessons.md: {e}")

    # 3. Delete the run directory entirely
    run_path = Path(run_dir)
    if run_path.exists():
        try:
            shutil.rmtree(run_path)
            typer.echo(f"Run directory '{run_dir}' deleted entirely.")
        except Exception as e:
            typer.echo(f"Error deleting run directory '{run_dir}': {e}")
    else:
        typer.echo(f"Run directory '{run_dir}' does not exist, skipping deletion.")

    typer.echo("Reset completed.")


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


@app.command("llm-test")
def llm_test(
    prompt: str = typer.Argument(
        ...,
        help="The prompt/instruction for strategy modification.",
    ),
    strategy: str = typer.Option(
        "haa",
        "--strategy",
        "-s",
        help="Strategy name in the registry.",
    ),
    model: str = typer.Option(
        "gpt-4o",
        "--model",
        "-m",
        help="LLM model name to run.",
    ),
    provider: str = typer.Option(
        "litellm",
        "--provider",
        "-p",
        help="LLM provider: 'litellm' or 'mock'.",
    ),
) -> None:
    """Test LLM-driven strategy edits against validation preflight checks."""
    strategies_dir = Path("strategies")
    configs_dir = Path("configs")

    strategy_path = strategies_dir / f"{strategy}.py"
    config_path = configs_dir / f"{strategy}.yaml"

    if not strategy_path.exists():
        typer.echo(f"Error: Strategy file not found at {strategy_path}")
        raise typer.Exit(code=1)

    if not config_path.exists():
        typer.echo(f"Error: Config file not found at {config_path}")
        raise typer.Exit(code=1)

    try:
        strategy_code = strategy_path.read_text(encoding="utf-8")
        config_yaml = config_path.read_text(encoding="utf-8")
    except Exception as e:
        typer.echo(f"Error reading files: {e}")
        raise typer.Exit(code=1) from e

    # 1. Construct AgentContext
    from autobacktest.llm.base import AgentContext

    context = AgentContext(
        strategy_name=strategy,
        strategy_code=strategy_code,
        config_yaml=config_yaml,
        program_text=prompt,
        evaluation_report=None,
        iteration=1,
    )

    # 2. Instantiate LLM Provider
    from autobacktest.llm.base import LLMProvider
    from autobacktest.llm.litellm_provider import LiteLLMProvider
    from autobacktest.llm.mock_provider import MockProvider

    provider_impl: LLMProvider
    if provider == "litellm":
        provider_impl = LiteLLMProvider(model=model)
    elif provider == "mock":
        provider_impl = MockProvider()
    else:
        typer.echo(f"Error: Unknown provider '{provider}'")
        raise typer.Exit(code=1)

    # 3. Call Provider
    typer.echo(f"Calling LLM provider '{provider}' with model '{model}'...")
    try:
        edit = provider_impl.generate_edit(context)
    except Exception as e:
        typer.echo(f"Error generating LLM edit: {e}")
        raise typer.Exit(code=1) from e

    typer.echo(f"Reasoning:\n{edit.reasoning}\n")

    # 4. Write Temporary Files for Validation
    import uuid

    candidate_py_path = strategies_dir / f"{strategy}.py.candidate"
    candidate_yaml_path = configs_dir / f"{strategy}.yaml.candidate"
    temp_name = f"{strategy}_candidate_{uuid.uuid4().hex}"
    temp_py_path = strategies_dir / f"{temp_name}.py"
    temp_yaml_path = configs_dir / f"{temp_name}.yaml"

    try:
        # Temporary files for validator preflight
        temp_py_path.write_text(edit.strategy_code, encoding="utf-8")
        temp_yaml_path.write_text(edit.config_yaml, encoding="utf-8")
    except Exception as e:
        typer.echo(f"Error writing temporary files for validation: {e}")
        raise typer.Exit(code=1) from e

    # 5. Run Preflight
    from autobacktest.strategy.validator import preflight

    typer.echo("Running pre-flight validation on generated candidate...")
    try:
        res = preflight(temp_name, strategies_dir, configs_dir)
    finally:
        # Clean up temporary validator files
        if temp_py_path.exists():
            temp_py_path.unlink()
        if temp_yaml_path.exists():
            temp_yaml_path.unlink()

    # 6. Print Results and Persist Candidate on Success
    if res.passed:
        try:
            candidate_py_path.write_text(edit.strategy_code, encoding="utf-8")
            candidate_yaml_path.write_text(edit.config_yaml, encoding="utf-8")
        except Exception as e:
            typer.echo(f"Error writing candidate files: {e}")
            raise typer.Exit(code=1) from e
        typer.echo("SUCCESS: Candidate passed all preflight validation checks!")
        typer.echo(f"Candidate Python: {candidate_py_path}")
        typer.echo(f"Candidate Config: {candidate_yaml_path}")
    else:
        typer.echo(f"FAILED: Validation failed with error code: {res.error_code}")
        typer.echo(f"Detail: {res.detail}")


def main() -> None:
    """Entry point for the console script."""
    app()


if __name__ == "__main__":
    main()
