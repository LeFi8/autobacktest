"""CLI command 'spa' implementation."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from autobacktest.config import settings
from autobacktest.evaluator.spa import calculate_hansen_spa
from autobacktest.ledger.store import LedgerStore


def register_command(app: typer.Typer) -> None:
    @app.command("spa")
    def hansen_spa(
        run_dir: str = typer.Option(
            str(settings.run_dir),
            "--run-dir",
            help="Path to runs directory containing ledger.db.",
        ),
        run_id: str | None = typer.Option(
            None,
            "--run-id",
            help="Run ID to analyze. Defaults to the latest run.",
        ),
        paths: int = typer.Option(
            1000,
            "--paths",
            "-p",
            help="Number of bootstrap paths for SPA test.",
        ),
        block_size: int = typer.Option(
            21,
            "--block-size",
            "-b",
            help="Block size/expected block size for stationary bootstrap.",
        ),
        accepted_only: bool = typer.Option(
            False,
            "--accepted-only",
            help="Audit only accepted/confirmed strategies instead of all evaluated.",
        ),
        seed: int = typer.Option(
            42,
            "--seed",
            help="Random seed for bootstrap reproducibility.",
        ),
    ) -> None:
        """Audit optimization runs using Hansen's Superior Predictive Ability (SPA) test."""
        db_path = Path(run_dir) / "ledger.db"
        if not db_path.exists():
            typer.echo(f"Error: Ledger store database not found at '{db_path}'.")
            raise typer.Exit(code=1)

        store = LedgerStore(db_path)
        try:
            target_run_id = run_id or store.latest_run_id()
            if not target_run_id:
                typer.echo("Error: No runs found in the ledger database.")
                raise typer.Exit(code=1)

            benchmark_returns, alternative_returns = store.fetch_run_returns(target_run_id, accepted_only=accepted_only)

            if benchmark_returns is None:
                typer.echo(f"Error: Benchmark returns (iteration 0) not found for run '{target_run_id}'.")
                raise typer.Exit(code=1)

            if alternative_returns.empty:
                scope = "accepted " if accepted_only else ""
                typer.echo(f"Warning: No {scope}alternative candidate strategies found for run '{target_run_id}'.")
                raise typer.Exit(code=0)

            results = calculate_hansen_spa(
                benchmark_returns=benchmark_returns,
                alternative_returns=alternative_returns,
                n_paths=paths,
                block_size=block_size,
                seed=seed,
            )
        finally:
            store.close()

        p_consistent = results["p_consistent"]
        p_upper = results["p_upper"]
        p_lower = results["p_lower"]
        t_spa = results["t_spa"]

        console = Console()
        console.print("")
        header = Panel(
            Text(f"Hansen's Superior Predictive Ability (SPA) Audit — {target_run_id}", style="bold cyan"),
            border_style="cyan",
        )
        console.print(header)

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", justify="right")
        table.add_column("Statistical Interpretation")

        table.add_row("Observed Test Statistic T^SPA", f"{t_spa:.4f}", "Max normalized mean performance difference.")

        p_c_color = "[green]" if p_consistent <= 0.05 else "[yellow]"
        table.add_row(
            "Consistent P-value", f"{p_c_color}{p_consistent:.4f}[/]", "Standard SPA p-value (threshold-centered)."
        )
        table.add_row(
            "Upper P-value Bound (Conservative)",
            f"{p_upper:.4f}",
            "Assumes all alternatives have zero mean (conservative).",
        )
        table.add_row(
            "Lower P-value Bound (Liberal)",
            f"{p_lower:.4f}",
            "Assumes underperforming alternatives have mean <= 0 (liberal).",
        )

        console.print(table)
        console.print("")

        if p_consistent <= 0.05:
            verdict = (
                "[bold green]VERDICT: PASS[/]\n"
                "At least one optimized candidate strategy significantly outperforms the baseline "
                "at the 5% level, even after correcting for the data-snooping bias of trying "
                f"{alternative_returns.shape[1]} alternative configurations."
            )
        else:
            verdict = (
                "[bold yellow]VERDICT: FAIL[/]\n"
                "No candidate strategy significantly outperforms the baseline at the 5% level "
                "after correcting for multiple testing. The observed outperformance of the best "
                "candidate could easily be due to data-snooping/chance."
            )

        console.print(
            Panel(
                verdict,
                title="Statistical Conclusion",
                border_style="green" if p_consistent <= 0.05 else "yellow",
            )
        )
        console.print("")
