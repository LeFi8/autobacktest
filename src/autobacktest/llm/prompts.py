"""System and user prompts construction for the LLM strategy optimizer."""

import re
from typing import Any

import numpy as _np
import pandas as _pd

from autobacktest.config import settings
from autobacktest.llm.base import AgentContext
from autobacktest.strategy.config_schema import StrategyConfig as _StrategyConfig

_PANDAS_VERSION = _pd.__version__
_NUMPY_VERSION = _np.__version__
_ALLOWED_TOP_LEVEL_KEYS = sorted(_StrategyConfig.model_fields.keys())

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
   Critically, you MUST always include `from typing import Any` — strategies
   that type-annotate with `dict[str, Any]` without this import will crash at runtime.
5. You MUST keep the portfolio weights non-negative (weights >= 0.0)
   and summing to at most 1.0 (sum <= 1.0) for every rebalance day.
   As a mandatory final step before returning, always apply:
   `weights = weights.clip(lower=0.0)`
   `weights = weights.div(weights.sum(axis=1), axis=0).fillna(0.0)`
   This renormalization prevents float-accumulation errors from multiple sequential
   normalization/capping passes that cause weight sums to drift above 1.0.
6. The output strategy code and config YAML MUST be complete file contents, NOT diffs.
7. You MUST maintain a running markdown document of "lessons learned"
   in the lessons_text field. In every response, you will output an updated
   "lessons_text" field to record what worked, what failed, and general
   principles discovered.
   - Summarize findings of the previous iteration (e.g. if the previous
     edit failed AST checks, execution, or the gate, analyze why and record it).
    - Keep lessons structured, concise, and action-oriented.
     - Each lesson block MUST include a type tag of format: `- **Type:** <ENUM>` (ENUM:
       BUG, DIVERSITY, GATE_REJECTION, PERFORMANCE_INSIGHT, or STRUCTURAL).
    - If the current lessons exceed the 4096 token limit (~16k characters),
      you MUST prune, compress, and consolidate older or less useful lessons to fit.
8. Runtime Environment & Banned pandas APIs Rule:
   The strategy code runs against pandas=={_PANDAS_VERSION} and numpy=={_NUMPY_VERSION}.
   The following pandas APIs are REMOVED in this version and will crash at runtime. NEVER use them:
   - `.groupby(axis=...)`: The `axis` parameter is removed. Drop it entirely.
   - Frequency aliases are CONTEXT-SENSITIVE — using the wrong one crashes at runtime:
     * DatetimeIndex operations (resample, date_range, bdate_range, pd.Grouper, asfreq on a
       DatetimeIndex, date offsets): use the NEW aliases
       'ME' (not 'M'), 'BME' (not 'BM'), 'QE' (not 'Q'), 'YE' (not 'A'/'Y'),
       'h' (not 'H'), 'min' (not 'T'), 's' (not 'S').
     * Period operations (to_period, period_range, pd.Period, asfreq on a PeriodIndex): use the
       ORIGINAL codes 'M', 'Q', 'Y' — passing 'ME'/'QE'/'YE' to a Period raises
       ValueError("for Period, please use 'M' instead of 'ME'").
   - `.mean(level=)` / `.sum(level=)` / `.std(level=)`: Use `.groupby(level=).mean()` etc.
   - `.fillna(method='ffill')` / `.fillna(method='bfill')`: Use `.ffill()` / `.bfill()` directly.
   - `DataFrame.append(other)`: Use `pd.concat([df, other])`.
   - `Series.iteritems()`: Use `.items()`.
   Additionally:
   - NEVER use a pandas Series in a boolean `if` statement. Use `.any()`, `.all()`, `.empty`, or `.item()`.
   - When assigning a value to a DataFrame column, ensure the right-hand side is a single Series or
     scalar — NOT a multi-column DataFrame. Use `.squeeze()` or select a specific column first.
   - `pd.unique(x)` / `x.unique()` require a Series, Index, or ndarray — NEVER a Python list.
     Wrap with `pd.Series(x)`, or use `set(x)` / `np.unique(x)` for plain lists.
   - NEVER call `np.nanmean` / `np.nanstd` on arrays that may have all-NaN slices — the warning
     is emitted INSIDE nanmean before any post-hoc fix can suppress it. Instead, compute mean
     manually using pre-filled arrays so the warning never fires:
     1-D: `val = arr[~np.isnan(arr)].mean() if (~np.isnan(arr)).any() else 0.0`
     N-D (any axis, e.g. axis=2 on a 3-D stack):
       `valid = ~np.isnan(stacked); count = valid.sum(axis=2)`
       `avg = np.where(count > 0, np.where(valid, stacked, 0.0).sum(axis=2) / np.maximum(count, 1.0), 0.0)`
9. Config Schema Rule:
   The config YAML is validated by a Pydantic model with `extra="forbid"`. Only these top-level keys
   are permitted: {_ALLOWED_TOP_LEVEL_KEYS}
   ALL strategy-specific parameters (momentum windows, thresholds, top_n, etc.) MUST go under the
   `params:` key. Placing any custom parameter at the top level will cause a hard validation error.
   Every ticker symbol referenced in the strategy code MUST appear in the `universe` list in config.
10. No Full-Sample Statistics Rule (Lookahead Prevention):
    NEVER compute mean, std, min, max, rank, quantile, z-score, or any other normalization over
    an entire column or the full price history. This is lookahead bias — future rows affect past signals.
    ALL statistics MUST use only trailing windows up to time t:
    - Use rolling(window=N) or expanding() windows, not plain .mean() / .std() on the whole series.
    - If you normalize (z-score, rank), do it within each rolling window.
    - Appending future price rows MUST NOT change any signal for past dates. This is tested automatically.
    Common violations that fail the lookahead test:
    - `prices[ticker].rank(pct=True)` — ranks whole column, sees future
    - `(x - x.mean()) / x.std()` — normalizes over full history
    - `prices.pct_change().mean()` — mean over full history
    - `prices.groupby(prices.index.to_period("M")).tail(1).index` — rebalance dates derived
      from the last OBSERVED day of each month. When future data is appended to the same month,
      the "last day" changes and past signals flip. THIS ALWAYS FAILS THE SNIFF TEST.
      Safe pattern: use purely calendar-based month-ends that never depend on what data exists:
      `pd.date_range(start=prices.index.min(), end=prices.index.max(), freq='BME').intersection(prices.index)`
11. Mandatory Decomposition Rule:
    The complexity and line-count limits are enforced PER FUNCTION. Use this to your advantage:
    - `generate_signals` MUST be an orchestrator — it calls helper functions, does not inline logic.
    - Extract signal computation, normalization, weight calculation, and regime detection into
      separate named helper functions (e.g., `_compute_momentum`, `_apply_regime_filter`,
      `_normalize_weights`).
    - Each helper function must stay under {settings.max_cyclomatic_complexity} cyclomatic complexity
      and {settings.max_function_lines} lines. Deeply nested if/elif chains, multiple loops, and
      list comprehensions with multiple conditions all increase complexity — split them into helpers.
    - A `generate_signals` that is a flat sequence of complex logic will always fail the AST check.
12. Diversity Rule: Diversity is enforced on the strategy's **return profile** (behavioral),
   not on config syntax alone. After backtesting, if your strategy's returns are too highly
   correlated (>95%) with any prior attempt for the same universe, it will be rejected as
   behaviorally redundant. To avoid this, you MUST generate strategies that behave differently
   from prior attempts — change the momentum metric (e.g., EWMA crossover instead of 13612U),
   alter the signal logic, modify the weighting or regime scheme, or change the asset universe.
   Stale parameter tweaks that produce near-identical return streams will be caught.
13. Strict JSON/Formatting Rule: Do not output any conversational text
   before or after the JSON payload. For reasoning/thinking models,
   the very first character immediately following the closing </think>
   tag must be the opening {{ of the JSON payload. No markdown
   wrapping (like ```json) is permitted.
14. Attempt History Rule: Before proposing a strategy, consult the
    ## Attempt History section. Do NOT re-propose configs in already-explored
    regions — cross-reference the history table to identify which metric
    directions or structural approaches remain unexplored and target those.
    Reason explicitly about gaps in the explored space.
15. AST Complexity and Size Limits Rule:
    Your strategy file has strict structural limits enforced by AST checks:
    - Maximum McCabe cyclomatic complexity of any function is 20. Keep functions
      simple: avoid heavily nested conditions (if/elif/else, deeply nested loops,
      list comprehensions with multiple if clauses, or extensive boolean/and/or
      chains). You MUST refactor `generate_signals` into small helper functions — it must be an
      orchestrator that calls helpers, not a flat block of inline logic. See Rule 11 (Mandatory Decomposition).
    - Maximum line count of any single function is 100.
16. Whitelist & Forbidden Names Rule:
    - The `.format()` method and attributes like `__dict__`, `__class__`, etc.,
      are strictly forbidden by AST checks. To format strings, you MUST use
      f-strings or standard string concatenation.
    - Only imports from the allowed whitelist ({sorted_imports}) are permitted.
      Wildcard imports (`*`) are blocked. Avoid accessing forbidden variables or
      built-in functions such as `eval`, `exec`, `getattr`, `setattr`, or
      pandas/numpy filesystem read/write operations (e.g., `read_csv`, `to_csv`).
"""


def parse_lessons(lessons_text: str) -> list[dict[str, str]]:
    """Parse lessons.md markdown content into a list of parsed lesson dicts.
    Each dict has keys: 'title', 'type', 'body'.
    """
    if not lessons_text:
        return []

    # Split by h3 header: '### ' at the beginning of a line
    pattern = r"^###\s+(.+)$"
    parts = re.split(pattern, lessons_text, flags=re.MULTILINE)

    lessons = []
    for i in range(1, len(parts), 2):
        title = parts[i].strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""

        # Search for Type metadata, e.g., "- **Type:** BUG" or "- **Type:** DIVERSITY"
        type_match = re.search(r"-\s+\*\*Type:\*\*\s*(\w+)", body, re.IGNORECASE)
        lesson_type = type_match.group(1).upper() if type_match else "STRUCTURAL"

        lessons.append(
            {
                "title": title,
                "type": lesson_type,
                "body": body,
            }
        )
    return lessons


def filter_lessons(lessons_text: str, context_stage: str | None) -> str:
    """Filter and reconstruct lessons based on the active stage/context."""
    if not lessons_text:
        return "No lessons recorded yet."

    lessons = parse_lessons(lessons_text)
    if not lessons:
        return lessons_text.strip()

    # Determine the target lesson type based on context_stage
    target_type = None
    if context_stage in ("validation", "eval_error"):
        target_type = "BUG"
    elif context_stage in ("diversity_config", "diversity_returns"):
        target_type = "DIVERSITY"
    elif context_stage == "gate":
        target_type = "GATE_REJECTION"

    if not target_type:
        return lessons_text.strip()

    # Prioritize target_type first, followed by others (or target_type + general STRUCTURAL)
    filtered = []
    for lesson in lessons:
        if lesson["type"] == target_type or lesson["type"] == "STRUCTURAL":
            filtered.append(f"### {lesson['title']}\n{lesson['body']}")

    if not filtered:
        return lessons_text.strip()

    return "\n\n".join(filtered)


def _text_block(text: str, cache: bool = False) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def build_messages(
    context: AgentContext,
    cache_supported: bool = False,
) -> list[dict[str, Any]]:
    """Build the system and user message payload for the LLM completion API.

    Args:
        context: Immutable context defining the current optimization state.
        cache_supported: When True, emit Anthropic-style cache_control breakpoints
            on the stable prefix so the provider caches SYSTEM_PROMPT + program_text.

    Returns:
        List containing the system message and user message dicts.
    """
    # The stable prefix (SYSTEM_PROMPT + program_text) is byte-identical across all
    # iterations within a run. For Anthropic we mark the boundary with cache_control;
    # for OpenAI the identical system string triggers automatic server-side caching.
    system_content: str | list[dict[str, Any]]
    if cache_supported:
        system_content = [
            _text_block(SYSTEM_PROMPT),
            _text_block(f"## Objective\n{context.program_text}", cache=True),
        ]
    else:
        system_content = f"{SYSTEM_PROMPT}\n\n## Objective\n{context.program_text}"

    system_message: dict[str, Any] = {
        "role": "system",
        "content": system_content,
    }

    context_stage = context.last_attempt.get("stage") if context.last_attempt else None
    injected_lessons = filter_lessons(context.lessons_text, context_stage)

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
            f"tracked for this asset universe. Strategies whose return streams correlate "
            f">95% with any prior attempt will be rejected after backtesting. "
            f"Ensure your strategy behaves differently — not just has different parameters.\n"
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
                    "**Explanation:** `lookahead_detected` means past signals changed when future "
                    "price rows were appended. The most common causes are:\n"
                    "  1. Full-sample normalization: `(x - x.mean()) / x.std()` or `.rank(pct=True)` "
                    "over the whole column — future rows shift the mean/rank for ALL past dates.\n"
                    "  2. Negative shifts: `.shift(-n)` looks forward in time.\n"
                    "  3. Any statistic computed over the full history at signal generation time.\n"
                    "Fix: replace whole-column statistics with `rolling(N).mean()` / `rolling(N).std()` "
                    "so each signal only sees data up to time t. This is a hard disqualifier."
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
            f"  - In-sample max drawdown <= {context.dd_limit:.2f}\n"
            f"  - In-sample turnover <= {context.turnover_limit:.2f}\n"
            f"  - All historical crisis regime stress tests must pass\n"
            f"  - Target metric must strictly exceed the incumbent in-sample value above.\n"
            f"  - Annualized return must be at least {context.min_return_ratio * 100:.0f}% "
            f"of the incumbent's annualized return.\n"
            f"  - Deflated Sharpe (DSR) non-degradation is **always enforced** on the "
            f"in-sample selection basis.\n\n"
            f"> [!WARNING]\n"
            f"> While the hard drawdown limit is {context.dd_limit:.2f}, strategies pushing close to\n"
            f"> the boundary are risky — prefer candidates that operate well within constraints.\n\n"
            f"Strategies that pass the in-sample select gate face a hidden OOS holdout\n"
            f"confirmation gate before commit. That holdout is **not visible** here.\n\n"
        )
    else:
        performance_target_section = (
            "## Performance Target\n"
            "No incumbent evaluation yet (first iteration). "
            f"Hard gate limits: in-sample drawdown <= {context.dd_limit:.2f}, "
            f"turnover <= {context.turnover_limit:.2f}, "
            "all regime stress tests must pass. "
            f"Once a baseline exists, annualized return must be at least "
            f"{context.min_return_ratio * 100:.0f}% of the baseline.\n"
            "DSR non-degradation is always enforced on the in-sample selection basis.\n"
            f"> [!WARNING]\n"
            f"> While the hard drawdown limit is {context.dd_limit:.2f}, strategies pushing close to\n"
            f"> the boundary are risky — prefer candidates that operate well within constraints.\n\n"
            "Strategies that pass select are confirmed against a hidden OOS holdout.\n\n"
        )

    # Mode-aware instruction section
    if context.mode == "exploit":
        mode_section = (
            "## Mode\n"
            "**EXPLOIT** — Locally refine the incumbent strategy. The diversity gate is suspended this round.\n"
            "Make small, targeted parameter tweaks or minor signal adjustments to the best strategy found so far.\n"
            "Do NOT make large structural changes. Focus on squeezing out marginal improvements.\n"
            "IMPORTANT: You MUST change at least one parameter value — returning the incumbent config unchanged\n"
            "is rejected automatically. Every exploit candidate must differ from the current best strategy."
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

## Lessons
{injected_lessons}
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
