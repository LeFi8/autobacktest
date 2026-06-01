"""System and user prompts construction for the LLM strategy optimizer."""

from typing import Any

from autobacktest.config import settings
from autobacktest.llm.base import AgentContext

# System prompt outlining constraints and role
sorted_imports = sorted(settings.parsed_safe_imports)
SYSTEM_PROMPT = f"""You are an expert quantitative strategist and Python developer.
Your goal is to optimize mathematical quantitative trading strategies to maximize
backtest performance while satisfying all provided guidelines and constraints.

You operate in a strict execution loop and MUST adhere to the following rules:
1. You ONLY modify the strategy code and config YAML.
2. You MUST return valid Python code that exports a single function matching:
   def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
3. The returned DataFrame MUST have the exact same shape, columns, and index as input.
4. You MUST NOT import any module outside whitelisted ALLOWED_IMPORTS:
   {sorted_imports}
5. You MUST keep the portfolio weights non-negative (weights >= 0.0)
   and summing to at most 1.0 (sum <= 1.0) for every rebalance day
6. The output strategy code and config YAML MUST be complete file contents, NOT diffs.
7. You MUST maintain a running markdown document of "lessons learned"
   in the lessons_text field. In every response, you will output an updated
   "lessons_text" field to record what worked, what failed, and general
   principles discovered.
   - Summarize findings of the previous iteration (e.g. if the previous
     edit failed AST checks, execution, or the gate, analyze why and record it).
   - Keep lessons structured, concise, and action-oriented.
    - If the current lessons exceed the 4096 token limit (~16k characters),
      you MUST prune, compress, and consolidate older or less useful lessons to fit.
8. Diversity Rule: Your proposed strategy config YAML will be compared
   against ALL past attempts with the same asset universe. If it has
    >95% similarity (same params, same asset sets, same structure), the
   iteration will be rejected WITHOUT backtesting and the iteration
   budget is consumed. To avoid this, you MUST explore structurally
   different approaches each time — change the asset universe, swap the
   momentum metric (e.g., EWMA crossover instead of 13612U), alter the
   canary logic, or modify the weighting scheme. Stale parameter tweaks
   (varying hysteresis by ±0.005) will be caught.
9. Strict JSON/Formatting Rule: Do not output any conversational text
   before or after the JSON payload. For reasoning/thinking models,
   the very first character immediately following the closing </think>
   tag must be the opening {{ of the JSON payload. No markdown
   wrapping (like ```json) is permitted.
10. Attempt History Rule: Before proposing a strategy, consult the
    ## Attempt History section. Do NOT re-propose configs in already-explored
    regions — cross-reference the history table to identify which metric
    directions or structural approaches remain unexplored and target those.
    Reason explicitly about gaps in the explored space.
"""


def build_messages(context: AgentContext) -> list[dict[str, str]]:
    """Build the system and user message payload for the LLM completion API.

    Args:
        context: Immutable context defining the current optimization state.

    Returns:
        List containing the system message and user message dicts.
    """
    system_message = {
        "role": "system",
        "content": SYSTEM_PROMPT,
    }

    # Format the latest evaluation — in-sample walk-forward aggregate only.
    # Holdout metrics are deliberately hidden from the LLM.
    if context.evaluation_report is not None:
        rep = context.evaluation_report
        try:
            m = rep.in_sample_metrics
            wf_span = f"{m.start_date} → {m.end_date}"
            wf_count = len(rep.walk_forward_metrics)
            folds_detail = ""
            if wf_count > 1:
                fold_sharpes = [f.sharpe_ratio for f in rep.walk_forward_metrics]
                min_s = min(fold_sharpes)
                max_s = max(fold_sharpes)
                folds_detail = f"\n  Per-fold Sharpe:   {min_s:.4f} - {max_s:.4f} (across {wf_count} windows)"
            eval_report_str = (
                f"In-Sample Walk-Forward Aggregate (selection basis):\n"
                f"  Window:            {wf_span}\n"
                f"  Sharpe:            {m.sharpe_ratio:.4f}\n"
                f"  Sortino:           {m.sortino_ratio:.4f}\n"
                f"  Information Ratio: {m.information_ratio:.4f}\n"
                f"  Max Drawdown:      {m.max_drawdown:.4f}\n"
                f"  Turnover:          {m.turnover:.4f}"
                f"{folds_detail}\n"
                f"  DSR (selection):   {rep.deflated_sharpe:.4f}\n"
                f"  Effective Trials:  {rep.effective_trials}\n"
                f"  Regime tests:      {'PASS' if rep.regime_passed else 'FAIL'}\n"
                f"\n"
                f"> **OOS holdout** is reserved as a **budgeted confirmation gate** — "
                f"it is never shown here and cannot be optimised against."
            )
        except Exception:
            eval_report_str = str(context.evaluation_report)
    else:
        eval_report_str = "First iteration (no prior evaluation report exists)."

    # Calculate character-based token proxy for lessons limit warning
    lessons_tokens = len(context.lessons_text) // 4
    warning_str = ""
    if lessons_tokens > 4096:
        warning_str = (
            f"\n> [!WARNING]\n"
            f"> The lessons document size (approx. {lessons_tokens} tokens) "
            f"exceeds the cap of 4096 tokens.\n"
            f"> You MUST compress, consolidate, or prune the lessons in lessons_text "
            f"to keep them under the cap.\n"
        )

    # Attempt history table section
    attempt_history_section = ""
    if context.attempt_history:
        committed_rows = [r for r in context.attempt_history if r.get("committed")]
        non_committed_rows = [r for r in context.attempt_history if not r.get("committed")]
        total_non_committed = len(non_committed_rows)
        omitted_count = max(0, total_non_committed - 25)
        displayed_non_committed = non_committed_rows[-25:] if omitted_count > 0 else non_committed_rows
        rows_to_render = committed_rows + displayed_non_committed

        def _fmt(val: Any, default: float = 0.0) -> str:
            try:
                return f"{float(val if val is not None else default):.4f}"
            except (TypeError, ValueError):
                return "-"

        def _outcome(r: dict[str, Any]) -> str:
            if r.get("committed"):
                return "✓ committed"
            if r.get("accepted"):
                return "✓ accepted"
            reason = r.get("rejection_reason") or ""
            return f"✗ {str(reason)[:30]}"

        def _fingerprint_repr(r: dict[str, Any]) -> str:
            fp = r.get("config_fingerprint", {})
            s = str(fp)
            return s[:60] + "..." if len(s) > 60 else s

        def _ho_flag(r: dict[str, Any]) -> str:
            if r.get("holdout_confirmed"):
                return "✓ HO"
            return ""

        header = "| iter | outcome | target | DSR | regime | reason | config | HO |"
        separator = "|------|---------|--------|-----|--------|--------|--------|-----|"
        table_lines = [header, separator]
        for r in rows_to_render:
            rejection_reason = r.get("rejection_reason") or None
            reason_col = "-" if (r.get("committed") or rejection_reason is None) else str(rejection_reason)[:30]
            row = (
                f"| {r.get('iteration', '')} "
                f"| {_outcome(r)} "
                f"| {_fmt(r.get('target_metric_value'))} "
                f"| {_fmt(r.get('deflated_sharpe'))} "
                f"| {'pass' if r.get('regime_passed') else ('FAIL' if 'regime_passed' in r else '-')} "
                f"| {reason_col} "
                f"| {_fingerprint_repr(r)} "
                f"| {_ho_flag(r)} |"
            )
            table_lines.append(row)

        table_str = "\n".join(table_lines)
        omit_note = ""
        if omitted_count > 0:
            total_non_committed_shown = len(displayed_non_committed)
            omit_note = (
                f"\n(showing {total_non_committed_shown + len(committed_rows)} of "
                f"{len(context.attempt_history)} total — oldest non-committed omitted)"
            )
        attempt_history_section = f"\n## Attempt History\n{table_str}{omit_note}\n"

    # Diversity warning section
    diversity_warning = ""
    if context.n_historical_configs > 0:
        diversity_warning = (
            f"\n## Diversity Warning\n"
            f"There are {context.n_historical_configs} attempted strategy variants "
            f"tracked (including rejected ones) for this asset universe. The config "
            f"similarity gate will reject proposals with >95% fingerprint overlap.\n"
        )

    # Build the "Previous Attempt Result" section if a failed attempt exists
    previous_attempt_section = ""
    if context.last_attempt is not None:
        attempt = context.last_attempt
        stage = attempt.get("stage", "unknown")
        lines: list[str] = ["## Previous Attempt Result", f"**Stage:** {stage}"]

        if stage == "validation":
            error_code = attempt.get("error_code", "")
            detail = attempt.get("detail", "")
            lines.append(f"**Error:** `{error_code}`")
            lines.append(f"**Detail:** {detail}")
            if error_code == "lookahead_detected":
                lines.append(
                    "**Explanation:** `lookahead_detected` means the strategy code reads "
                    "future price rows (e.g. using `.shift(-n)` with a negative shift, or "
                    "indexing beyond `t` at evaluation time). This is a hard disqualifier."
                )
            code = attempt.get("candidate_strategy_code", "")
            config = attempt.get("candidate_config_yaml", "")
            if code:
                lines.append(f"\n**Failed strategy code:**\n```python\n{code}\n```")
            if config:
                lines.append(f"\n**Failed config:**\n```yaml\n{config}\n```")

        elif stage == "diversity_config":
            detail = attempt.get("detail", "")
            lines.append(f"**Detail:** {detail}")
            config = attempt.get("candidate_config_yaml", "")
            if config:
                lines.append(f"\n**Rejected config:**\n```yaml\n{config}\n```")

        elif stage == "eval_error":
            detail = attempt.get("detail", "")
            lines.append(f"**Error:** {detail}")
            code = attempt.get("candidate_strategy_code", "")
            config = attempt.get("candidate_config_yaml", "")
            if code:
                lines.append(f"\n**Failed strategy code:**\n```python\n{code}\n```")
            if config:
                lines.append(f"\n**Failed config:**\n```yaml\n{config}\n```")

        elif stage == "diversity_returns":
            detail = attempt.get("detail", "")
            lines.append(f"**Detail:** {detail}")
            metrics = attempt.get("candidate_metrics", {})
            if metrics:
                lines.append("**Observed metrics:**")
                for k, v in metrics.items():
                    lines.append(f"  - {k}: {v}")
            config = attempt.get("candidate_config_yaml", "")
            if config:
                lines.append(f"\n**Rejected config:**\n```yaml\n{config}\n```")

        elif stage == "gate":
            rejection_reason = attempt.get("rejection_reason", "")
            failed_gate = attempt.get("failed_gate", "")
            lines.append(f"**Rejection reason:** {rejection_reason}")
            lines.append(f"**Failed gate:** `{failed_gate}`")
            metrics = attempt.get("candidate_metrics", {})
            if metrics:
                lines.append("**Candidate metrics at rejection:**")
                for k, v in metrics.items():
                    lines.append(f"  - {k}: {v}")
            code = attempt.get("candidate_strategy_code", "")
            config = attempt.get("candidate_config_yaml", "")
            if code:
                lines.append(f"\n**Failed strategy code:**\n```python\n{code}\n```")
            if config:
                lines.append(f"\n**Failed config:**\n```yaml\n{config}\n```")

        else:
            detail = attempt.get("detail", attempt.get("rejection_reason", ""))
            if detail:
                lines.append(f"**Detail:** {detail}")

        lines.append(
            "\n**Diagnose the failure above. Do NOT regenerate the same code or config. "
            "Your fix must address the specific error shown.**"
        )
        previous_attempt_section = "\n".join(lines) + "\n\n"

    # Build the "Performance Target" section (in-sample walk-forward basis)
    if context.evaluation_report is not None:
        rep = context.evaluation_report
        m = rep.in_sample_metrics
        performance_target_section = (
            f"## Performance Target\n"
            f"Incumbent in-sample walk-forward aggregate (you must beat these):\n"
            f"  - Sharpe: {m.sharpe_ratio:.4f}\n"
            f"  - Sortino: {m.sortino_ratio:.4f}\n"
            f"  - Information Ratio: {m.information_ratio:.4f}\n"
            f"  - Max Drawdown: {m.max_drawdown:.4f}\n"
            f"  - Turnover: {m.turnover:.4f}\n\n"
            f"Hard gate limits (select — all must pass):\n"
            f"  - In-sample max drawdown <= 0.20\n"
            f"  - In-sample turnover <= 2.0\n"
            f"  - All historical crisis regime stress tests must pass\n"
            f"  - Target metric must strictly exceed the incumbent in-sample value above.\n"
            f"  - Deflated Sharpe (DSR) non-degradation is **always enforced** on the "
            f"in-sample selection basis.\n\n"
            f"Strategies that pass the in-sample select gate face a hidden OOS holdout\n"
            f"confirmation gate before commit. That holdout is **not visible** here.\n\n"
        )
    else:
        performance_target_section = (
            "## Performance Target\n"
            "No incumbent evaluation yet (first iteration). "
            "Hard gate limits: in-sample drawdown <= 0.20, turnover <= 2.0, "
            "all regime stress tests must pass. "
            "DSR non-degradation is always enforced on the in-sample selection basis.\n"
            "Strategies that pass select are confirmed against a hidden OOS holdout.\n\n"
        )

    # Mode-aware instruction section
    if context.mode == "exploit":
        mode_section = (
            "## Mode\n"
            "**EXPLOIT** — Locally refine the incumbent strategy. The diversity gate is suspended this round.\n"
            "Make small, targeted parameter tweaks or minor signal adjustments to the best strategy found so far.\n"
            "Do NOT make large structural changes. Focus on squeezing out marginal improvements."
        )
    else:
        mode_section = (
            "## Mode\n"
            "**EXPLORE** — Search for structurally different strategies. The diversity gate is active.\n"
            "You MUST propose approaches that differ meaningfully from previous attempts (see Attempt History)."
        )

    user_content = f"""## Iteration
Current Loop Iteration: {context.iteration}

{mode_section}

## Objective
{context.program_text}

## Lessons
{context.lessons_text or "No lessons recorded yet."}
{warning_str}
## Current Strategy Code
```python
{context.strategy_code}
```

## Current Config
```yaml
{context.config_yaml}
```

## Latest Evaluation
{eval_report_str}{attempt_history_section}
{diversity_warning}
{previous_attempt_section}{performance_target_section}## Instructions
Improve the strategy per the objective. Optimize parameters, signal
logic, or asset weights.
Your response must be returned as a JSON object containing the keys:
- "strategy_code": Complete, updated Python source code for the strategy.
- "config_yaml": Complete, updated YAML parameters content.
- "reasoning": Concise explanation of the quantitative logic and changes.
- "lessons_text": Complete, updated lessons markdown text incorporating learnings
  from the previous and current iterations, pruning if needed.

Strict JSON/Formatting Constraint: The response must be a single,
valid JSON object. Do not wrap the JSON object in markdown code
block markers (such as ```json or ```). The very first character
immediately following the </think> tag must be the opening {{ of the
JSON payload.
"""

    user_message = {
        "role": "user",
        "content": user_content.strip(),
    }

    return [system_message, user_message]
