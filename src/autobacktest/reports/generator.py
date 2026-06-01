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
) -> Path:
    baseline_cum = (1 + baseline_returns).cumprod()
    final_cum = (1 + final_returns).cumprod()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(baseline_cum.index, baseline_cum.values, label="Baseline", linewidth=1.5)
    ax.plot(final_cum.index, final_cum.values, label="Optimized", linewidth=1.5)
    ax.set_title(f"Equity Curves — {run_id}")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return")
    ax.legend()
    ax.grid(True, alpha=0.3)
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
