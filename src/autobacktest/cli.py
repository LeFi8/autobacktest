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
    candidate_py_path = strategies_dir / f"{strategy}.py.candidate"
    candidate_yaml_path = configs_dir / f"{strategy}.yaml.candidate"
    temp_py_path = strategies_dir / f"{strategy}_candidate.py"
    temp_yaml_path = configs_dir / f"{strategy}_candidate.yaml"

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
        res = preflight(f"{strategy}_candidate", strategies_dir, configs_dir)
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
