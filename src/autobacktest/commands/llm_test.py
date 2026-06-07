"""CLI command 'llm-test' implementation."""

from __future__ import annotations

import uuid
from typing import Any

import typer

from autobacktest import configure_verbosity
from autobacktest.config import settings
from autobacktest.llm.base import AgentContext
from autobacktest.strategy.validator import preflight


def register_command(app: typer.Typer) -> None:
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
            settings.llm_model,
            "--model",
            "-m",
            help="LLM model name to run.",
        ),
        provider: str = typer.Option(
            settings.llm_provider,
            "--provider",
            "-p",
            help="LLM provider: 'litellm' or 'mock'.",
        ),
        quiet: bool = typer.Option(
            settings.quiet,
            "--quiet",
            "-q",
            help="Suppress non-critical warnings and reduce terminal noise.",
        ),
    ) -> None:
        """Test LLM-driven strategy edits against validation preflight checks."""
        configure_verbosity(quiet=quiet)
        strategies_dir = settings.strategies_dir
        configs_dir = settings.configs_dir

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

        context = AgentContext(
            strategy_name=strategy,
            strategy_code=strategy_code,
            config_yaml=config_yaml,
            program_text=prompt,
            evaluation_report=None,
            iteration=1,
        )

        from autobacktest.llm.litellm_provider import LiteLLMProvider
        from autobacktest.llm.mock_provider import MockProvider

        provider_impl: Any = None
        if provider == "litellm":
            provider_impl = LiteLLMProvider(
                model=model,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )
        elif provider == "mock":
            provider_impl = MockProvider()
        else:
            typer.echo(f"Error: Unknown provider '{provider}'")
            raise typer.Exit(code=1)

        typer.echo(f"Calling LLM provider '{provider}' with model '{model}'...")
        try:
            edit = provider_impl.generate_edit(context)
        except Exception as e:
            typer.echo(f"Error generating LLM edit: {e}")
            raise typer.Exit(code=1) from e

        typer.echo(f"Reasoning:\n{edit.reasoning}\n")

        candidate_py_path = strategies_dir / f"{strategy}.py.candidate"
        candidate_yaml_path = configs_dir / f"{strategy}.yaml.candidate"
        temp_name = f"{strategy}_candidate_{uuid.uuid4().hex}"
        temp_py_path = strategies_dir / f"{temp_name}.py"
        temp_yaml_path = configs_dir / f"{temp_name}.yaml"

        try:
            temp_py_path.write_text(edit.strategy_code, encoding="utf-8")
            temp_yaml_path.write_text(edit.config_yaml, encoding="utf-8")
        except Exception as e:
            typer.echo(f"Error writing temporary files for validation: {e}")
            raise typer.Exit(code=1) from e

        typer.echo("Running pre-flight validation on generated candidate...")
        try:
            res = preflight(temp_name, strategies_dir, configs_dir)
        finally:
            if temp_py_path.exists():
                temp_py_path.unlink()
            if temp_yaml_path.exists():
                temp_yaml_path.unlink()

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
