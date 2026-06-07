"""CLI command 'run' implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autobacktest import configure_verbosity
from autobacktest.config import settings
from autobacktest.gate import TargetMetric
from autobacktest.orchestrator import OrchestratorResult, run_optimization
from autobacktest.reports.generator import (
    compile_failure_summary,
    compile_strategy_report,
    plot_equity_curves,
    plot_mc_histogram,
    plot_walk_forward_bars,
)
from autobacktest.strategy.config_schema import StrategyConfig

logger = logging.getLogger(__name__)


def _render_rich_summary(
    result: OrchestratorResult,
    iterations: int,
    report_path: Path | None,
    config_path: Path | None = None,
) -> None:
    """Render a detailed Rich summary dashboard for the completed run."""
    console = Console()
    report = result.final_report

    header = Panel(
        Text(f"AutoBacktest Optimization Complete — {result.run_id}", style="bold cyan"),
        border_style="cyan",
    )
    console.print(header)

    info = Table.grid(padding=(0, 2))
    info.add_column(style="bold")
    info.add_column()
    info.add_row("Strategy", result.run_id.split("-")[0] if "-" in result.run_id else "?")
    info.add_row("Branch", result.branch)
    info.add_row("Committed", f"{result.n_committed} / {iterations}")
    console.print(info)
    console.print("")

    metrics = Table(title="Performance Comparison", show_header=True, header_style="bold magenta")
    metrics.add_column("Metric", style="cyan")
    metrics.add_column("Baseline", justify="right")
    metrics.add_column("Final", justify="right")
    metrics.add_column("", justify="center")

    bl = result.baseline_report

    def _chg(current: float, baseline: float, higher_is_better: bool = True) -> str:
        diff = current - baseline
        if abs(diff) < 1e-8:
            return ""
        if higher_is_better:
            return "[green]▲[/]" if diff > 0 else "[red]▼[/]"
        return "[red]▲[/]" if diff > 0 else "[green]▼[/]"

    metrics.add_row(
        "Observed Sharpe",
        f"{bl.observed_sharpe:.3f}" if bl else "—",
        f"{report.observed_sharpe:.3f}",
        _chg(report.observed_sharpe, bl.observed_sharpe) if bl else "",
    )
    metrics.add_row(
        "Deflated Sharpe",
        f"{bl.deflated_sharpe:.3f}" if bl else "—",
        f"{report.deflated_sharpe:.3f}",
        _chg(report.deflated_sharpe, bl.deflated_sharpe) if bl else "",
    )

    metrics.add_row(
        "Max Drawdown",
        f"{bl.in_sample_metrics.max_drawdown * 100:.2f}%" if bl else "—",
        f"{report.in_sample_metrics.max_drawdown * 100:.2f}%",
        _chg(report.in_sample_metrics.max_drawdown, bl.in_sample_metrics.max_drawdown, higher_is_better=False)
        if bl
        else "",
    )
    metrics.add_row(
        "Turnover",
        f"{bl.in_sample_metrics.turnover:.2f}x" if bl else "—",
        f"{report.in_sample_metrics.turnover:.2f}x",
        _chg(report.in_sample_metrics.turnover, bl.in_sample_metrics.turnover, higher_is_better=False) if bl else "",
    )
    metrics.add_row(
        "Regime Stress",
        ("[green]Pass[/]" if bl.regime_passed else "[red]Fail[/]") if bl else "—",
        "[green]Pass[/]" if report.regime_passed else "[red]Fail[/]",
    )
    console.print(metrics)
    console.print("")

    gates = Table(title="Gate Results", show_header=True, header_style="bold blue")
    gates.add_column("Gate", style="cyan")
    gates.add_column("Status", justify="center")

    def _gate_pass(passed: bool) -> str:
        return "[green]✓ PASS[/]" if passed else "[red]✗ FAIL[/]"

    display_dd_limit = 0.20
    display_to_limit = 2.0
    if config_path is not None and config_path.exists():
        try:
            _cfg = StrategyConfig.from_yaml(config_path)
            display_dd_limit = _cfg.max_drawdown_limit
            display_to_limit = _cfg.turnover_limit
        except Exception:
            logger.warning("Failed to parse config %s for display, using defaults", config_path)

    gates.add_row(
        f"Max Drawdown ≤ {display_dd_limit * 100:.0f}%",
        _gate_pass(report.in_sample_metrics.max_drawdown <= display_dd_limit),
    )
    gates.add_row("Regime Stress", _gate_pass(report.regime_passed))
    gates.add_row(
        f"Turnover ≤ {display_to_limit:.1f}x",
        _gate_pass(report.in_sample_metrics.turnover <= display_to_limit),
    )
    console.print(gates)
    console.print("")

    cost = Table(title="Cost Summary", show_header=True, header_style="bold yellow")
    cost.add_column("Metric", style="cyan")
    cost.add_column("Value", justify="right")
    cost.add_row("Total Prompt Tokens", f"{result.total_prompt_tokens:,}")
    cost.add_row("Total Completion Tokens", f"{result.total_completion_tokens:,}")
    cost.add_row("Total Cost", f"[green]${result.total_cost:.4f}[/]")
    console.print(cost)

    if report_path:
        console.print(f"\n[bold]📄 Strategy Report:[/] [link=file://{report_path.resolve()}]{report_path.resolve()}[/]")


def register_command(app: typer.Typer) -> None:
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
            settings.llm_provider,
            "--provider",
            help="LLM provider (e.g. anthropic, openai, google).",
        ),
        model: str | None = typer.Option(
            settings.llm_model,
            "--model",
            help="LLM model name to run.",
        ),
        run_dir: str | None = typer.Option(
            str(settings.run_dir),
            "--run-dir",
            help="Directory to store runs and SQLite ledger.",
        ),
        target_metric: str = typer.Option(
            "sharpe",
            "--target-metric",
            help="Target metric for optimization: sharpe, sortino, or information_ratio.",
        ),
        resume: str | None = typer.Option(
            None,
            "--resume",
            help="Run ID to resume optimization from.",
        ),
        holdout_peek_limit: int = typer.Option(
            20,
            "--holdout-peek-limit",
            help="Maximum holdout peeks before early termination.",
        ),
        early_stop_patience: int = typer.Option(
            settings.early_stop_patience,
            "--early-stop-patience",
            help="Number of consecutive rejections allowed before early stopping.",
        ),
        json_output: bool = typer.Option(
            False,
            "--json",
            help="Output raw JSON instead of the Rich summary dashboard.",
        ),
        quiet: bool = typer.Option(
            settings.quiet,
            "--quiet",
            "-q",
            help="Suppress non-critical warnings and reduce terminal noise.",
        ),
    ) -> None:
        """Start the autonomous LLM-driven strategy optimization loop."""
        configure_verbosity(quiet=quiet)
        if not quiet:
            console = Console()
            console.print(
                Panel(
                    "[bold yellow]WARNING:[/] The default yfinance data provider is susceptible to "
                    "survivorship bias (omitting delisted/bankrupt tickers from past universes).\n"
                    "Verify performance against survivorship-free point-in-time data feeds before deploying live.",
                    title="Quantitative Due Diligence Notice",
                    border_style="yellow",
                )
            )

        try:
            metric = TargetMetric(target_metric)
        except ValueError as err:
            error_msg = f"Error: Unknown target metric '{target_metric}'. Use: sharpe, sortino, information_ratio."
            typer.echo(error_msg)
            raise typer.Exit(code=1) from err

        from autobacktest.llm.litellm_provider import LiteLLMProvider
        from autobacktest.llm.mock_provider import MockProvider

        provider_impl: Any
        if provider == "mock":
            provider_impl = MockProvider()
        else:
            model_str = model or settings.llm_model
            if provider and provider != "litellm" and "/" not in model_str:
                model_str = f"{provider}/{model_str}"
            provider_impl = LiteLLMProvider(
                model=model_str,
                temperature=settings.llm_temperature,
                max_tokens=settings.llm_max_tokens,
            )

        program_path = Path(program)
        run_dir_path = Path(run_dir) if run_dir else settings.run_dir

        try:
            result: OrchestratorResult = run_optimization(
                program_path=program_path,
                strategy_name=strategy,
                iterations=iterations,
                provider=provider_impl,
                run_dir=run_dir_path,
                target_metric=metric,
                holdout_peek_limit=holdout_peek_limit,
                early_stop_patience=early_stop_patience,
                resume=resume,
                quiet=quiet,
            )
        except Exception as e:
            typer.echo(f"Error: {e}")
            raise typer.Exit(code=1) from e

        if json_output:
            typer.echo(result.final_report.to_json())
            return

        strategy_path = settings.strategies_dir / f"{strategy}.py"
        config_path = settings.configs_dir / f"{strategy}.yaml"
        strategy_code = strategy_path.read_text(encoding="utf-8") if strategy_path.exists() else ""
        config_yaml = config_path.read_text(encoding="utf-8") if config_path.exists() else ""
        program_text = program_path.read_text(encoding="utf-8") if program_path.exists() else ""

        output_dir = run_dir_path / result.run_id
        baseline_report = result.baseline_report

        baseline_returns = getattr(baseline_report, "holdout_net_returns", None) if baseline_report else None
        final_returns = result.final_report.holdout_net_returns
        if (
            final_returns is not None
            and not final_returns.empty
            and baseline_returns is not None
            and not baseline_returns.empty
        ):
            benchmark_returns = getattr(result.final_report, "benchmark_returns", None)
            benchmark_ticker = getattr(result.final_report, "benchmark_ticker", "SPY")
            plot_equity_curves(
                baseline_returns,
                final_returns,
                result.run_id,
                output_dir,
                benchmark_returns=benchmark_returns,
                benchmark_ticker=benchmark_ticker,
            )

        mc_sharpes = getattr(result.final_report, "mc_sharpes", None)
        if mc_sharpes is not None and mc_sharpes.size > 0:
            plot_mc_histogram(
                mc_sharpes,
                result.final_report.observed_sharpe,
                result.final_report.mc_sharpe_5th,
                result.final_report.mc_sharpe_50th,
                result.final_report.mc_sharpe_95th,
                result.run_id,
                output_dir,
            )

        if result.final_report.walk_forward_metrics:
            plot_walk_forward_bars(
                result.final_report.walk_forward_metrics,
                result.run_id,
                output_dir,
            )

        failure_summary = compile_failure_summary(output_dir)

        if baseline_report is not None:
            compile_strategy_report(
                baseline_report=baseline_report,
                final_report=result.final_report,
                run_id=result.run_id,
                output_dir=output_dir,
                program_text=program_text,
                config_yaml=config_yaml,
                failure_summary=failure_summary,
                strategy_code=strategy_code,
            )

        if result.early_stopped and not quiet:
            console = Console()
            console.print(
                Panel(
                    f"[bold red]⚠ Run stopped early at iteration "
                    f"{result.early_stop_iteration}/{iterations}[/]\n"
                    f"No candidate passed all gates for "
                    f"{early_stop_patience} consecutive iterations.",
                    border_style="red",
                )
            )

        report_path = output_dir / "strategy_report.md"
        config_path = settings.configs_dir / f"{strategy}.yaml"
        _render_rich_summary(
            result,
            iterations,
            report_path if report_path.exists() else None,
            config_path=config_path,
        )
