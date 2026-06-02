import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.use("Agg")

from autobacktest.evaluator.report import EvaluationReport, WindowReport


def plot_equity_curves(
    baseline_returns: pd.Series,
    final_returns: pd.Series,
    run_id: str,
    output_dir: Path,
    benchmark_returns: pd.Series | None = None,
    benchmark_ticker: str = "SPY",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_cum = (1 + baseline_returns).cumprod()
    final_cum = (1 + final_returns).cumprod()

    has_benchmark = benchmark_returns is not None and not benchmark_returns.empty
    if has_benchmark:
        assert benchmark_returns is not None
        bench_cum = (1 + benchmark_returns).cumprod()
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [2, 1]},
            sharex=True,
        )
    else:
        fig, ax_top = plt.subplots(figsize=(12, 6))
        ax_bot = None

    ax_top.plot(
        baseline_cum.index,
        baseline_cum.values,
        label="Pre-Optimization",
        linewidth=1.5,
        linestyle="--",
        color="#1f77b4",
    )
    ax_top.plot(
        final_cum.index,
        final_cum.values,
        label="Optimized",
        linewidth=1.5,
    )
    if has_benchmark:
        ax_top.plot(
            bench_cum.index,
            bench_cum.values,
            label=benchmark_ticker,
            linewidth=1.5,
            linestyle="--",
            color="#ff7f0e",
        )
    ax_top.set_title(f"Holdout Period Equity Curves — {run_id}")
    ax_top.set_ylabel("Cumulative Return")
    ax_top.legend(loc="upper left")
    ax_top.grid(True, alpha=0.3)

    if has_benchmark and ax_bot is not None:
        aligned = pd.concat([baseline_cum, final_cum, bench_cum], axis=1, join="inner")
        baseline_active = aligned.iloc[:, 0] - aligned.iloc[:, 2]
        final_active = aligned.iloc[:, 1] - aligned.iloc[:, 2]
        ax_bot.plot(
            baseline_active.index,
            baseline_active.values,
            label="Pre-Optimization",
            linewidth=1.5,
            linestyle="--",
            color="#1f77b4",
        )
        ax_bot.plot(
            final_active.index,
            final_active.values,
            label="Optimized",
            linewidth=1.5,
        )
        ax_bot.axhline(y=0, color="gray", linestyle=":", linewidth=1)
        ax_bot.set_title("Active Return vs Benchmark")
        ax_bot.set_xlabel("Date")
        ax_bot.set_ylabel("Cumulative Active Return")
        ax_bot.legend(loc="upper left")
        ax_bot.grid(True, alpha=0.3)
    else:
        ax_top.set_xlabel("Date")

    fig.tight_layout()
    out_path = output_dir / "equity_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def compile_failure_summary(run_dir: Path) -> dict[str, Any]:
    events_path = run_dir / "events.jsonl"
    if not events_path.exists():
        return {}

    summary: dict[str, Any] = {
        "Validation": {},
        "Gate": {},
        "Diversity": {},
        "LLM Error": 0,
        "Eval Error": 0,
    }

    with events_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue

            if not isinstance(event, dict):
                continue

            val = event.get("validation")
            if isinstance(val, dict) and val.get("passed") is False:
                code = val.get("error_code", "unknown")
                bucket = summary["Validation"]
                bucket[code] = bucket.get(code, 0) + 1
                continue

            gate = event.get("gate")
            if isinstance(gate, dict) and gate.get("accepted") is False:
                fg = gate.get("failed_gate", "unknown")
                bucket = summary["Gate"]
                bucket[fg] = bucket.get(fg, 0) + 1
                continue

            div = event.get("diversity")
            if isinstance(div, dict) and div.get("passed") is False:
                tier = div.get("tier", "unknown")
                bucket = summary["Diversity"]
                bucket[tier] = bucket.get(tier, 0) + 1
                continue

            if event.get("llm_error") is not None:
                summary["LLM Error"] += 1
                continue

            eval_err = event.get("evaluation")
            if isinstance(eval_err, dict) and eval_err.get("error") is not None:
                summary["Eval Error"] += 1
                continue

    return summary


def compile_strategy_report(
    baseline_report: EvaluationReport,
    final_report: EvaluationReport,
    run_id: str,
    output_dir: Path,
    program_text: str,
    config_yaml: str,
    failure_summary: dict[str, Any],
    strategy_code: str,
) -> Path:
    lines: list[str] = []

    _h1(lines, "Strategy Optimization Report")
    _h2(lines, "Executive Summary")
    _kv(lines, "Strategy", final_report.strategy_name)
    _kv(lines, "Run ID", run_id)
    _kv(lines, "Baseline Sharpe", f"{baseline_report.observed_sharpe:.4f}")
    _kv(lines, "Final Sharpe", f"{final_report.observed_sharpe:.4f}")
    _kv(lines, "Holdout Sharpe", f"{final_report.holdout_metrics.sharpe_ratio:.4f}")
    _kv(lines, "Change", f"{final_report.observed_sharpe - baseline_report.observed_sharpe:+.4f}")

    _h2(lines, "Configuration")
    lines.append("```yaml")
    lines.append(config_yaml.rstrip())
    lines.append("```")
    lines.append("")

    _h2(lines, "Objective")
    lines.append("> " + program_text.strip().replace("\n", "\n> "))
    lines.append("")

    _h2(lines, "Robustness Diagnostics")
    _h3(lines, "In-Sample Metrics")
    _window_table(lines, final_report.in_sample_metrics, "In-Sample")
    _h3(lines, "Holdout Metrics")
    _window_table(lines, final_report.holdout_metrics, "Holdout")
    _h3(lines, "Monte Carlo Bootstrap (Sharpe)")
    _kv(lines, "5th Percentile", f"{final_report.mc_sharpe_5th:.4f}")
    _kv(lines, "50th Percentile", f"{final_report.mc_sharpe_50th:.4f}")
    _kv(lines, "95th Percentile", f"{final_report.mc_sharpe_95th:.4f}")
    _h3(lines, "Deflated Sharpe Ratio")
    _kv(lines, "DSR (In-Sample)", f"{final_report.deflated_sharpe:.4f}")
    _kv(lines, "DSR (Holdout)", f"{final_report.holdout_deflated_sharpe:.4f}")
    _kv(lines, "Effective Trials", str(final_report.effective_trials))
    _h3(lines, "Regime Stress Tests")
    _kv(lines, "Passed", str(final_report.regime_passed))
    if final_report.regime_drawdowns:
        lines.append("")
        lines.append("| Regime | Max Drawdown |")
        lines.append("|--------|-------------|")
        for regime, dd in final_report.regime_drawdowns.items():
            lines.append(f"| {regime} | {dd * 100:.2f}% |")

    _h2(lines, "Walk-Forward Windows")
    if final_report.walk_forward_metrics:
        lines.append("")
        lines.append("| Window | Sharpe | Max DD | Turnover |")
        lines.append("|--------|--------|--------|----------|")
        for i, w in enumerate(final_report.walk_forward_metrics, 1):
            lines.append(
                f"| WF-{i} ({w.start_date} - {w.end_date}) "
                f"| {w.sharpe_ratio:.3f} "
                f"| {w.max_drawdown * 100:.2f}% "
                f"| {w.turnover:.2f}x |"
            )

    _h2(lines, "Failure Summary")
    if failure_summary:
        lines.append("")
        _render_failure_summary_table(lines, failure_summary)
    else:
        lines.append("No failures recorded (events.jsonl not found or empty).")
        lines.append("")

    _h2(lines, "Equity Curves")
    lines.append("")
    png_path = output_dir / "equity_curves.png"
    if png_path.exists():
        lines.append("![equity_curves](equity_curves.png)")
    else:
        lines.append("*(Chart not generated)*")
    lines.append("")

    _h2(lines, "Final Source Code")
    lines.append("```python")
    lines.append(strategy_code.rstrip())
    lines.append("```")
    lines.append("")

    content = "\n".join(lines)
    out_path = output_dir / "strategy_report.md"
    out_path.write_text(content, encoding="utf-8")
    return out_path


def _h1(lines: list[str], text: str) -> None:
    lines.append("")
    lines.append(f"# {text}")
    lines.append("")


def _h2(lines: list[str], text: str) -> None:
    lines.append("")
    lines.append(f"## {text}")
    lines.append("")


def _h3(lines: list[str], text: str) -> None:
    lines.append(f"### {text}")
    lines.append("")


def _kv(lines: list[str], key: str, value: str) -> None:
    lines.append(f"- **{key}:** {value}")


def _window_table(lines: list[str], report: WindowReport, label: str) -> None:
    lines.append("")
    lines.append(f"| Metric | {label} |")
    lines.append("|--------|--------|")
    lines.append(f"| Sharpe Ratio | {report.sharpe_ratio:.4f} |")
    lines.append(f"| Sortino Ratio | {report.sortino_ratio:.4f} |")
    lines.append(f"| Annualized Return | {report.annualized_return * 100:.2f}% |")
    lines.append(f"| Annualized Volatility | {report.annualized_volatility * 100:.2f}% |")
    lines.append(f"| Max Drawdown | {report.max_drawdown * 100:.2f}% |")
    lines.append(f"| Turnover | {report.turnover:.2f}x |")


def _render_failure_summary_table(lines: list[str], failure_summary: dict[str, Any]) -> None:
    for bucket, value in failure_summary.items():
        if isinstance(value, dict):
            if not value:
                continue
            total = sum(value.values())
            lines.append(f"- **{bucket}** (total: {total})")
            for sub_code, count in sorted(value.items()):
                lines.append(f"  - `{sub_code}`: {count}")
        elif isinstance(value, int) and value > 0:
            lines.append(f"- **{bucket}**: {value}")
