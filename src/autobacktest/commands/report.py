"""CLI command 'report' implementation."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import typer
from rich.console import Console
from rich.table import Table

from autobacktest.config import settings
from autobacktest.ledger.store import LedgerStore


def register_command(app: typer.Typer) -> None:
    @app.command()
    def report(
        run_dir: str = typer.Option(
            str(settings.run_dir),
            "--run-dir",
            help="Path to runs directory containing ledger.db.",
        ),
        run_id: str | None = typer.Option(
            None,
            "--run-id",
            help="Run id to report. Defaults to the latest run.",
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

        if compare_all and strategy is not None:
            typer.echo("Error: Cannot specify both --strategy and --compare-all.")
            raise typer.Exit(code=1)

        store = LedgerStore(db_path)
        try:
            selected_run_id = run_id or store.latest_run_id()
            if selected_run_id is None:
                typer.echo("No runs found in ledger.")
                return

            selected_run = store.get_run(selected_run_id)
            attempts_in_run = store.attempts_for_run(selected_run_id)
            if selected_run is None and not attempts_in_run:
                typer.echo(f"No run found with run id '{selected_run_id}'.")
                return

            if not compare_all and strategy is None:
                if selected_run is not None:
                    strategy = str(selected_run["strategy_name"])
                else:
                    strategy = str(attempts_in_run[0]["strategy_name"])

            if compare_all:
                rows = store.leaderboard(run_id=selected_run_id)
            else:
                rows = store.leaderboard(strategy_name=strategy, run_id=selected_run_id)

            if not rows:
                typer.echo(f"No accepted attempts found for run '{selected_run_id}'.")
                return

            console = Console()
            table = Table(title=f"AutoBacktest Optimization Leaderboard ({selected_run_id})")
            table.add_column("Strategy", style="cyan", justify="left")
            table.add_column("Run ID", style="magenta", justify="left")
            table.add_column("Iter", style="blue", justify="center")
            table.add_column("Target Metric", style="magenta", justify="center")
            table.add_column("Target Value", style="green", justify="right")
            table.add_column("Observed Sharpe", justify="right")
            table.add_column("Deflated Sharpe", style="yellow", justify="right")
            table.add_column("Max DD", style="red", justify="right")
            table.add_column("Turnover", style="cyan", justify="right")
            table.add_column("Date", style="dim", justify="center")

            for r in rows:
                sharpe_val = float(cast(float, r["observed_sharpe"]))
                sharpe_style = "green" if sharpe_val > 0 else "red"
                sharpe_str = f"[bold {sharpe_style}]{sharpe_val:.3f}[/]"

                deflated_val = float(cast(float, r["deflated_sharpe"]))
                deflated_str = f"{deflated_val:.3f}"

                max_dd_val = float(cast(float, r["in_sample_max_drawdown"]))
                max_dd_str = f"{max_dd_val * 100:.2f}%"

                turnover_val = float(cast(float, r["in_sample_turnover"]))
                turnover_str = f"{turnover_val:.2f}x"

                date_str = r["created_at"]
                if isinstance(date_str, str) and len(date_str) >= 10:
                    date_str = date_str[:10]

                target_metric_str = str(r.get("target_metric", "sharpe")).upper()
                target_val = float(cast(float, r.get("target_metric_value", 0.0)))
                target_str = f"{target_val:.3f}"

                table.add_row(
                    str(r["strategy_name"]),
                    str(r["run_id"]),
                    str(r["iteration"]),
                    target_metric_str,
                    target_str,
                    sharpe_str,
                    deflated_str,
                    max_dd_str,
                    turnover_str,
                    str(date_str),
                )

            console.print(table)

            total_prompt = sum(cast(int, r.get("prompt_tokens", 0)) for r in attempts_in_run)
            total_completion = sum(cast(int, r.get("completion_tokens", 0)) for r in attempts_in_run)
            total_tokens = sum(cast(int, r.get("total_tokens", 0)) for r in attempts_in_run)
            total_cost = sum(cast(float, r.get("cost", 0.0)) for r in attempts_in_run)
            console.print(
                f"\n[bold]Total Run Optimization Cost:[/] [green]${total_cost:.4f}[/] "
                f"(Prompt: {total_prompt:,} tokens, "
                f"Completion: {total_completion:,} tokens, "
                f"Total: {total_tokens:,} tokens)"
            )
        finally:
            store.close()
