"""Report generation and visualization utilities for AutoBacktest.

Produces equity curve plots, Monte Carlo histograms, walk-forward bar charts,
failure summaries from the event log, and a self-contained institutional-grade
Markdown strategy report.
"""

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
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
    """Generate a Matplotlib comparison chart of cumulative returns and drawdowns.

    Creates a multi-panel figure with:
    - Top: cumulative equity curves for baseline, optimized, and benchmark
    - Middle: active return vs benchmark (when benchmark data is provided)
    - Bottom: drawdown profiles

    Args:
        baseline_returns: Daily returns series for the baseline strategy.
        final_returns: Daily returns series for the optimized strategy.
        run_id: Unique identifier for the run (used in chart title).
        output_dir: Directory to save the PNG file.
        benchmark_returns: Optional benchmark daily returns series.
        benchmark_ticker: Ticker label for the benchmark (default ``"SPY"``).

    Returns:
        Path to the saved ``equity_curves.png`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline_cum = (1 + baseline_returns).cumprod()
    final_cum = (1 + final_returns).cumprod()

    has_benchmark = benchmark_returns is not None and not benchmark_returns.empty
    if has_benchmark:
        assert benchmark_returns is not None  # narrow type inside guard
        bench_cum = (1 + benchmark_returns).cumprod()
        fig, (ax_top, ax_mid, ax_bot) = plt.subplots(
            3,
            1,
            figsize=(12, 10),
            gridspec_kw={"height_ratios": [2, 1, 1]},
            sharex=True,
        )
    else:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=(12, 8),
            gridspec_kw={"height_ratios": [2, 1]},
            sharex=True,
        )
        ax_mid = None

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
        color="#ff7f0e",
    )
    if has_benchmark:
        ax_top.plot(
            bench_cum.index,
            bench_cum.values,
            label=benchmark_ticker,
            linewidth=1.5,
            linestyle="--",
            color="#7f8c8d",
        )
    ax_top.set_title(f"Holdout Period Equity Curves — {run_id}")
    ax_top.set_ylabel("Cumulative Return")
    ax_top.legend(loc="upper left")
    ax_top.grid(True, alpha=0.3)

    if has_benchmark and ax_mid is not None:
        aligned = pd.concat([baseline_cum, final_cum, bench_cum], axis=1, join="inner")
        if aligned.empty:
            baseline_active = pd.Series(dtype=float)
            final_active = pd.Series(dtype=float)
        else:
            baseline_active = aligned.iloc[:, 0] - aligned.iloc[:, 2]
            final_active = aligned.iloc[:, 1] - aligned.iloc[:, 2]
        ax_mid.plot(
            baseline_active.index,
            baseline_active.values,
            label="Pre-Optimization",
            linewidth=1.5,
            linestyle="--",
            color="#1f77b4",
        )
        ax_mid.plot(
            final_active.index,
            final_active.values,
            label="Optimized",
            linewidth=1.5,
            color="#ff7f0e",
        )
        ax_mid.axhline(
            y=0,
            color="#7f8c8d",
            linestyle=":",
            linewidth=1.2,
            label=benchmark_ticker,
        )
        ax_mid.set_title("Active Return vs Benchmark")
        ax_mid.set_ylabel("Cumulative Active Return")
        ax_mid.legend(loc="upper left")
        ax_mid.grid(True, alpha=0.3)

    # Drawdown subplot (always last)
    def _drawdown(returns: pd.Series) -> pd.Series:
        cum = (1 + returns).cumprod()
        running_max = cum.cummax()
        return (cum - running_max) / running_max

    final_dd = _drawdown(final_returns)
    baseline_dd = _drawdown(baseline_returns)
    ax_bot.fill_between(final_dd.index, 0, final_dd.values * 100, label="Optimized", alpha=0.4, color="#ff7f0e")
    ax_bot.plot(
        baseline_dd.index,
        baseline_dd.values * 100,
        label="Pre-Optimization",
        linewidth=1.0,
        linestyle="--",
        color="#1f77b4",
    )
    if has_benchmark:
        bench_dd = _drawdown(benchmark_returns)
        ax_bot.plot(
            bench_dd.index,
            bench_dd.values * 100,
            label=benchmark_ticker,
            linewidth=1.0,
            linestyle="--",
            color="#7f8c8d",
        )
    ax_bot.set_title("Drawdown")
    ax_bot.set_xlabel("Date")
    ax_bot.set_ylabel("Drawdown %")
    ax_bot.legend(loc="lower left")
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = output_dir / "equity_curves.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_mc_histogram(
    mc_sharpes: np.ndarray | None,
    observed_sharpe: float,
    mc_sharpe_5th: float,
    mc_sharpe_50th: float,
    mc_sharpe_95th: float,
    run_id: str,
    output_dir: Path,
) -> Path | None:
    """Generate a Monte Carlo distribution histogram of bootstrapped Sharpe ratios.

    Overlays the observed Sharpe ratio and percentile thresholds (p5, p50, p95)
    from the bootstrap simulation.  Annotates the probability of negative Sharpe.

    Args:
        mc_sharpes: Array of bootstrapped Sharpe ratios.
        observed_sharpe: The strategy's observed Sharpe ratio.
        mc_sharpe_5th: 5th percentile of the bootstrap distribution.
        mc_sharpe_50th: 50th percentile of the bootstrap distribution.
        mc_sharpe_95th: 95th percentile of the bootstrap distribution.
        run_id: Unique identifier for the run (used in chart title).
        output_dir: Directory to save the PNG file.

    Returns:
        Path to the saved ``mc_histogram.png`` file, or ``None`` if no data.
    """
    if mc_sharpes is None or mc_sharpes.size == 0:
        return None
    fig, ax = plt.subplots(figsize=(10, 6))
    neg_prob = float((mc_sharpes < 0).mean()) * 100
    ax.hist(mc_sharpes, bins=50, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.axvline(mc_sharpe_5th, color="red", linestyle="--", linewidth=1.2, label=f"p5: {mc_sharpe_5th:.3f}")
    ax.axvline(mc_sharpe_50th, color="green", linestyle="--", linewidth=1.2, label=f"p50: {mc_sharpe_50th:.3f}")
    ax.axvline(mc_sharpe_95th, color="purple", linestyle="--", linewidth=1.2, label=f"p95: {mc_sharpe_95th:.3f}")
    ax.axvline(observed_sharpe, color="black", linestyle="-", linewidth=1.5, label=f"Observed: {observed_sharpe:.3f}")
    ax.set_title(f"Monte Carlo Distribution of Sharpe Ratios — {run_id}")
    ax.set_xlabel("Sharpe Ratio")
    ax.set_ylabel("Frequency")
    ax.legend(loc="upper right")
    ax.annotate(
        f"P(Sharpe < 0) = {neg_prob:.1f}%",
        xy=(0.02, 0.95),
        xycoords="axes fraction",
        fontsize=10,
        va="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.8},
    )
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path = output_dir / "mc_histogram.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def plot_walk_forward_bars(
    walk_forward_metrics: list[WindowReport],
    run_id: str,
    output_dir: Path,
) -> Path | None:
    """Generate a grouped bar chart of Sharpe ratio, max drawdown, and turnover per walk-forward window.

    Args:
        walk_forward_metrics: List of ``WindowReport`` objects from each walk-forward test window.
        run_id: Unique identifier for the run (used in chart title).
        output_dir: Directory to save the PNG file.

    Returns:
        Path to the saved ``walk_forward_bars.png`` file, or ``None`` if no data.
    """
    if not walk_forward_metrics:
        return None
    labels = [f"WF-{i + 1}" for i in range(len(walk_forward_metrics))]
    sharpes = [w.sharpe_ratio for w in walk_forward_metrics]
    max_dds = [w.max_drawdown * 10 for w in walk_forward_metrics]
    turnovers = [w.turnover for w in walk_forward_metrics]

    x = np.arange(len(labels))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width, sharpes, width, label="Sharpe", color="#1f77b4")
    ax.bar(x, max_dds, width, label="Max DD x 10", color="#ff7f0e")
    ax.bar(x + width, turnovers, width, label="Turnover", color="#2ca02c")
    ax.set_title(f"Walk-Forward Window Metrics — {run_id}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(y=0, color="gray", linestyle=":", linewidth=0.8)
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    out_path = output_dir / "walk_forward_bars.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def _tally_legacy_event(event: dict[str, object], summary: dict[str, Any]) -> None:
    """Accumulate failure counts from a legacy flat-key event schema.

    Handles the schema produced before the ``candidates`` array was introduced.
    Returns early after the first matching key to avoid double-counting.

    Args:
        event: Parsed event dict with optional top-level ``validation``, ``gate``,
            ``diversity``, ``llm_error``, and ``evaluation`` keys.
        summary: Running failure summary dict updated in-place.
    """
    val = event.get("validation")
    if isinstance(val, dict) and val.get("passed") is False:
        code = val.get("error_code", "unknown")
        summary["Validation"][code] = summary["Validation"].get(code, 0) + 1
        return
    gate = event.get("gate")
    if isinstance(gate, dict) and gate.get("accepted") is False:
        fg = gate.get("failed_gate", "unknown")
        summary["Gate"][fg] = summary["Gate"].get(fg, 0) + 1
        return
    div = event.get("diversity")
    if isinstance(div, dict) and div.get("passed") is False:
        tier = div.get("tier", "unknown")
        summary["Diversity"][tier] = summary["Diversity"].get(tier, 0) + 1
        return
    if event.get("llm_error") is not None:
        summary["LLM Error"] = summary["LLM Error"] + 1
        return
    eval_err = event.get("evaluation")
    if isinstance(eval_err, dict) and eval_err.get("error") is not None:
        summary["Eval Error"] = summary["Eval Error"] + 1


def _tally_candidate(c: dict[str, object], summary: dict[str, Any]) -> None:
    """Accumulate failure counts from a single candidate dict in the modern schema.

    Args:
        c: Candidate dict with keys such as ``passed``, ``stage``, ``detail``,
            ``failed_gate``, and ``llm_error``.
        summary: Running failure summary dict updated in-place.
    """
    if c.get("passed") is True:
        return
    if c.get("llm_error"):
        summary["LLM Error"] = summary["LLM Error"] + 1
        return
    stage = c.get("stage", "")
    if stage == "validation":
        code = c.get("detail", "unknown")
        summary["Validation"][code] = summary["Validation"].get(code, 0) + 1
    elif stage == "gate":
        fg = c.get("failed_gate") or "unknown"
        summary["Gate"][fg] = summary["Gate"].get(fg, 0) + 1
    elif stage in ("diversity_config", "diversity_returns"):
        tier = stage.removeprefix("diversity_")
        summary["Diversity"][tier] = summary["Diversity"].get(tier, 0) + 1
    elif stage == "eval_error":
        summary["Eval Error"] = summary["Eval Error"] + 1


def compile_failure_summary(run_dir: Path) -> dict[str, Any]:
    """Parse the events JSONL file and compile per-category failure statistics.

    Groups failures into buckets: Validation, Gate, Diversity, LLM Error,
    and Eval Error.  Handles both the modern ``candidates`` array schema
    and the legacy flat key schema for backward compatibility.

    Args:
        run_dir: Path to the run directory containing ``events.jsonl``.

    Returns:
        Dict mapping failure category to either a sub-dict of error codes
        with counts, or an integer counter.
    """
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

            candidates = event.get("candidates")
            if not isinstance(candidates, list):
                _tally_legacy_event(event, summary)
                continue

            for c in candidates:
                if isinstance(c, dict):
                    _tally_candidate(c, summary)

    return summary


def _add_robustness_section(lines: list[str], report: EvaluationReport) -> None:
    """Add robustness diagnostics section."""
    _h2(lines, "Robustness Diagnostics")
    # Note: IS/HO metrics now shown in unified table in Benchmark Comparison section
    _h3(lines, "Monte Carlo Bootstrap (Sharpe)")
    _kv(lines, "5th Percentile", f"{report.mc_sharpe_5th:.4f}")
    _kv(lines, "50th Percentile", f"{report.mc_sharpe_50th:.4f}")
    _kv(lines, "95th Percentile", f"{report.mc_sharpe_95th:.4f}")
    _h3(lines, "Deflated Sharpe Ratio")
    _kv(lines, "DSR (In-Sample)", f"{report.deflated_sharpe:.4f}")
    _kv(lines, "DSR (Holdout)", f"{report.holdout_deflated_sharpe:.4f}")
    _kv(lines, "Effective Trials", str(report.effective_trials))
    _h3(lines, "Regime Stress Tests")
    _kv(lines, "Passed", str(report.regime_passed))
    if report.regime_drawdowns:
        lines.append("")
        lines.append("| Regime | Max Drawdown |")
        lines.append("|--------|-------------|")
        for regime, dd in report.regime_drawdowns.items():
            lines.append(f"| {regime} | {dd * 100:.2f}% |")


def _add_walk_forward_section(lines: list[str], report: EvaluationReport) -> None:
    """Add walk-forward windows section."""
    _h2(lines, "Walk-Forward Windows")
    if report.walk_forward_metrics:
        lines.append("")
        lines.append("| Window | Sharpe | Max DD | Turnover |")
        lines.append("|--------|--------|--------|----------|")
        for i, w in enumerate(report.walk_forward_metrics, 1):
            lines.append(
                f"| WF-{i} ({w.start_date} - {w.end_date}) "
                f"| {w.sharpe_ratio:.3f} "
                f"| {w.max_drawdown * 100:.2f}% "
                f"| {w.turnover:.2f}x |"
            )


def _add_benchmark_section(lines: list[str], report: EvaluationReport) -> None:
    """Add benchmark comparison section."""
    _h2(lines, "Benchmark Comparison")
    bench_is = report.benchmark_in_sample_metrics
    bench_ho = report.benchmark_holdout_metrics

    # Unified comparison table
    _unified_comparison_table(
        lines,
        report.in_sample_metrics,
        bench_is,
        report.holdout_metrics,
        bench_ho,
        report.benchmark_ticker,
    )

    # Active return metrics if benchmark available
    if bench_is is not None and bench_ho is not None:
        ticker = report.benchmark_ticker
        strat_ho_ret = report.holdout_metrics.annualized_return
        bench_ho_ret = bench_ho.annualized_return
        excess = strat_ho_ret - bench_ho_ret
        lines.append("")
        _kv(lines, f"Active Return ({ticker})", f"{excess * 100:+.2f}%")
        excess_vol = (report.holdout_metrics.annualized_volatility - bench_ho.annualized_volatility) * 100
        _kv(lines, "Excess Volatility", f"{excess_vol:+.2f}%")
        ho_ir = report.holdout_metrics.information_ratio
        ho_ir_str = f"{ho_ir:+.4f}" if ho_ir is not None else "N/A"
        _kv(lines, "Information Ratio (HO)", ho_ir_str)
        if bench_is.information_ratio is None:
            lines.append("- *Benchmark IR shown as 0.0 — computed against itself, not meaningful.*")
    else:
        lines.append("")
        lines.append("*(Benchmark performance data not available — re-run evaluation with benchmark price data.)*")
        lines.append("")


def _add_failure_summary_section(lines: list[str], failure_summary: dict[str, Any]) -> None:
    """Add failure summary section."""
    _h2(lines, "Failure Summary")
    if failure_summary:
        lines.append("")
        _render_failure_summary_table(lines, failure_summary)
    else:
        lines.append("No failures recorded (events.jsonl not found or empty).")
        lines.append("")


def _embed_chart(lines: list[str], output_dir: Path, filename: str, title: str, fallback: str) -> None:
    """Embed a chart PNG if it exists, otherwise show fallback text."""
    _h2(lines, title)
    lines.append("")
    chart_path = output_dir / filename
    if chart_path.exists():
        lines.append(f"![{title}]({filename})")
    else:
        lines.append(f"*({fallback})*")
    lines.append("")


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
    """Generate a self-contained institutional-grade Markdown strategy report.

    Sections include: executive summary, YAML configuration, objective,
    robustness diagnostics (walk-forward, Monte Carlo, DSR), regime stress
    tests, benchmark comparison, failure summary, and finalized source code.
    Embeds references to PNG chart files generated by other functions.

    Args:
        baseline_report: Evaluation report for the baseline strategy.
        final_report: Evaluation report for the optimized strategy.
        run_id: Unique run identifier.
        output_dir: Directory to write ``strategy_report.md``.
        program_text: Raw program.md content.
        config_yaml: Final YAML configuration string.
        failure_summary: Dict from ``compile_failure_summary``.
        strategy_code: Final strategy source code.

    Returns:
        Path to the generated ``strategy_report.md`` file.
    """
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

    _add_robustness_section(lines, final_report)

    _add_walk_forward_section(lines, final_report)

    _add_benchmark_section(lines, final_report)

    _add_failure_summary_section(lines, failure_summary)

    _embed_chart(lines, output_dir, "equity_curves.png", "Equity Curves", "Chart not generated")
    _embed_chart(lines, output_dir, "mc_histogram.png", "Monte Carlo Distribution", "MC histogram not available")
    _embed_chart(
        lines, output_dir, "walk_forward_bars.png", "Walk-Forward Metrics", "Walk-forward bar chart not available"
    )

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
    """Append an H1 heading line to the report line buffer."""
    lines.append("")
    lines.append(f"# {text}")
    lines.append("")


def _h2(lines: list[str], text: str) -> None:
    """Append an H2 heading line to the report line buffer."""
    lines.append("")
    lines.append(f"## {text}")
    lines.append("")


def _h3(lines: list[str], text: str) -> None:
    """Append an H3 heading line to the report line buffer."""
    lines.append(f"### {text}")
    lines.append("")


def _kv(lines: list[str], key: str, value: str) -> None:
    """Append a bold key / value pair as a list item."""
    lines.append(f"- **{key}:** {value}")


def _window_table(lines: list[str], report: WindowReport, label: str) -> None:
    """Append a Markdown table with key metrics from a WindowReport."""
    lines.append("")
    lines.append(f"| Metric | {label} |")
    lines.append("|--------|--------|")
    lines.append(f"| Sharpe Ratio | {report.sharpe_ratio:.4f} |")
    lines.append(f"| Sortino Ratio | {report.sortino_ratio:.4f} |")
    lines.append(f"| Annualized Return | {report.annualized_return * 100:.2f}% |")
    lines.append(f"| Annualized Volatility | {report.annualized_volatility * 100:.2f}% |")
    lines.append(f"| Max Drawdown | {report.max_drawdown * 100:.2f}% |")
    lines.append(f"| Turnover | {report.turnover:.2f}x |")


def _unified_comparison_table(
    lines: list[str],
    strat_is: WindowReport,
    bench_is: WindowReport | None,
    strat_ho: WindowReport,
    bench_ho: WindowReport | None,
    bench_ticker: str = "SPY",
) -> None:
    """Append a unified comparison table with all metrics in columns."""
    lines.append("")
    lines.append("### Performance Comparison")
    lines.append("")

    if bench_is is not None and bench_ho is not None:
        lines.append(f"| Metric | Strategy (IS) | {bench_ticker} (IS) | Strategy (HO) | {bench_ticker} (HO) |")
        lines.append("|--------|--------------|----------|---------------|----------|")
        lines.append(
            f"| Sharpe Ratio | {strat_is.sharpe_ratio:.4f} | {bench_is.sharpe_ratio:.4f} "
            f"| {strat_ho.sharpe_ratio:.4f} | {bench_ho.sharpe_ratio:.4f} |"
        )
        lines.append(
            f"| Sortino Ratio | {strat_is.sortino_ratio:.4f} | {bench_is.sortino_ratio:.4f} "
            f"| {strat_ho.sortino_ratio:.4f} | {bench_ho.sortino_ratio:.4f} |"
        )
        lines.append(
            f"| Annualized Return | {strat_is.annualized_return * 100:.2f}% | {bench_is.annualized_return * 100:.2f}% "
            f"| {strat_ho.annualized_return * 100:.2f}% | {bench_ho.annualized_return * 100:.2f}% |"
        )
        lines.append(
            f"| Annualized Volatility | {strat_is.annualized_volatility * 100:.2f}% "
            f"| {bench_is.annualized_volatility * 100:.2f}% "
            f"| {strat_ho.annualized_volatility * 100:.2f}% "
            f"| {bench_ho.annualized_volatility * 100:.2f}% |"
        )
        lines.append(
            f"| Max Drawdown | {strat_is.max_drawdown * 100:.2f}% | {bench_is.max_drawdown * 100:.2f}% "
            f"| {strat_ho.max_drawdown * 100:.2f}% | {bench_ho.max_drawdown * 100:.2f}% |"
        )
        lines.append(
            f"| Turnover | {strat_is.turnover:.2f}x | {bench_is.turnover:.2f}x "
            f"| {strat_ho.turnover:.2f}x | {bench_ho.turnover:.2f}x |"
        )
    else:
        lines.append("| Metric | In-Sample | Holdout |")
        lines.append("|--------|-----------|---------|")
        lines.append(f"| Sharpe Ratio | {strat_is.sharpe_ratio:.4f} | {strat_ho.sharpe_ratio:.4f} |")
        lines.append(f"| Sortino Ratio | {strat_is.sortino_ratio:.4f} | {strat_ho.sortino_ratio:.4f} |")
        lines.append(
            f"| Annualized Return | {strat_is.annualized_return * 100:.2f}% | {strat_ho.annualized_return * 100:.2f}% |"
        )
        lines.append(
            f"| Annualized Volatility | {strat_is.annualized_volatility * 100:.2f}% "
            f"| {strat_ho.annualized_volatility * 100:.2f}% |"
        )
        lines.append(f"| Max Drawdown | {strat_is.max_drawdown * 100:.2f}% | {strat_ho.max_drawdown * 100:.2f}% |")
        lines.append(f"| Turnover | {strat_is.turnover:.2f}x | {strat_ho.turnover:.2f}x |")


def _render_failure_summary_table(lines: list[str], failure_summary: dict[str, Any]) -> None:
    """Append a formatted failure summary table to the report line buffer.

    Renders dict-valued buckets as indented sub-lists with totals,
    and integer-valued entries as simple list items.
    """
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
