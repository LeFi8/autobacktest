"""System and user prompts construction for the LLM strategy optimizer."""

import difflib
import re
from typing import Any

import numpy as _np
import pandas as _pd

from autobacktest.config import settings
from autobacktest.llm.base import AgentContext
from autobacktest.strategy.config_schema import StrategyConfig as _StrategyConfig
from autobacktest.strategy.constants import FORBIDDEN_NAMES

_PANDAS_VERSION = _pd.__version__
_NUMPY_VERSION = _np.__version__
_ALLOWED_TOP_LEVEL_KEYS = sorted(_StrategyConfig.model_fields.keys())
_CONSTRAINTS_TEXT_INDENTED = "\n".join("    " + line for line in _StrategyConfig.constraints_text().splitlines())
_SORTED_FORBIDDEN_NAMES = ", ".join(sorted(FORBIDDEN_NAMES))

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
   The config schema constraints are:
{_CONSTRAINTS_TEXT_INDENTED}
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
    - Maximum line count of any single function is {settings.max_function_lines}. Aim to keep
      every function under 80 lines (hard limit {settings.max_function_lines}).
16. Whitelist & Forbidden Names Rule:
    - The `.format()` method and attributes like `__dict__`, `__class__`, etc.,
      are strictly forbidden by AST checks. To format strings, you MUST use
      f-strings or standard string concatenation.
    - Only imports from the allowed whitelist ({sorted_imports}) are permitted.
      Wildcard imports (`*`) are blocked.
    - The following names are strictly forbidden anywhere in the strategy code
      (variables, functions, attributes, imports, etc.) by AST checks:
      {_SORTED_FORBIDDEN_NAMES}
      Use of any of these names will trigger a hard validation failure.
"""


def parse_lessons(lessons_text: str) -> list[dict[str, str]]:
    """Parse lessons markdown content into a list of lesson dictionaries.

    Splits on ``### `` H1 headers and extracts the type tag from each lesson body.

    Args:
        lessons_text: Raw markdown text containing lessons with ``### `` headers.

    Returns:
        List of dicts, each with keys ``'title'``, ``'type'`` (BUG, DIVERSITY,
        GATE_REJECTION, PERFORMANCE_INSIGHT, or STRUCTURAL), and ``'body'``.
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
    """Filter and reconstruct lessons based on the active optimization stage.

    Prioritizes lessons whose type matches the current failure stage (e.g.,
    BUG lessons for validation errors, DIVERSITY lessons for diversity gate
    failures). STRUCTURAL lessons are always included as general guidance.

    Args:
        lessons_text: Raw lessons markdown text.
        context_stage: Current optimization stage (``'validation'``,
            ``'eval_error'``, ``'diversity_config'``, ``'diversity_returns'``,
            ``'gate'``), or ``None`` to return all lessons unfiltered.

    Returns:
        Filtered lessons as a markdown string. Returns the original text
        unfiltered when no stage-specific filtering applies.
    """
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
    """Create an Anthropic-style text content block with optional cache control.

    Args:
        text: The text content for this block.
        cache: When True, attaches ``cache_control: {"type": "ephemeral"}`` to
            mark this block as a cache breakpoint for Anthropic's prompt caching.

    Returns:
        Dict suitable for inclusion in a ``content`` list for Anthropic tool use.
    """
    block: dict[str, Any] = {"type": "text", "text": text}
    if cache:
        block["cache_control"] = {"type": "ephemeral"}
    return block


def _diff_code(old: str, new: str, context_lines: int = 3) -> str:
    """Produce a unified diff between two code strings.

    Returns empty string if either input is empty.

    Args:
        old: Baseline (incumbent) source code.
        new: Candidate (failed) source code to compare.
        context_lines: Number of context lines around each change (default 3).

    Returns:
        Unified diff string with `---`/`+++` markers, or empty string.
    """
    if not old or not new:
        return ""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines, fromfile="incumbent", tofile="candidate", n=context_lines)
    return "".join(diff)


def _diff_config(old_yaml: str, new_yaml: str) -> str:
    """Produce a unified diff between two config YAML strings.

    Returns empty string if either input is empty or if the configs are identical.

    Args:
        old_yaml: Baseline (incumbent) config YAML.
        new_yaml: Candidate (failed) config YAML to compare.

    Returns:
        Unified diff string with ``---``/``+++`` markers, or empty string.
    """
    if not old_yaml or not new_yaml:
        return ""
    return _diff_code(old_yaml, new_yaml)


def _format_previous_attempt_hint(error_code: str, detail: str) -> str:
    """Return a contextual remediation hint for a specific validation error.

    Maps each error code to targeted advice that helps the LLM understand
    what went wrong and how to fix it.

    Args:
        error_code: The ``ValidationError`` string code (e.g.,
            ``'lookahead_detected'``, ``'ast_blocked_import'``).
        detail: The original error detail message from the validation check.

    Returns:
        Markdown-formatted remediation hint, or empty string for unknown codes.
    """
    if error_code == "lookahead_detected":
        return (
            "\nLookahead bias was detected via a shifted-rerun test, "
            "meaning past signals changed when future price rows were appended. "
            "The 3 common causes are:\n"
            "  1. Calculating statistics on the full sample "
            "(like `.mean()`, `.std()`, `.min()`, `.max()`, or `.rank(pct=True)`) "
            "on the entire column instead of a trailing/expanding window.\n"
            "  2. Missing or negative shifts "
            "(e.g., using `.shift(-1)` or not shifting lookbacks/features by at least `.shift(1)`).\n"
            "  3. Using `center=True` in rolling calculations."
        )
    elif error_code in ("ast_line_limit_exceeded", "ast_cyclomatic_complexity_exceeded"):
        return (
            "\nTo resolve function length or cyclomatic complexity limit failures, "
            "you must extract logic into top-level helper functions, "
            "vectorize operations instead of branching "
            "(e.g. using loops or nested if-statements), and keep the core logic identical."
        )
    elif error_code in ("ast_blocked_import", "undefined_name", "config_schema_invalid"):
        alt = ""
        if error_code == "config_schema_invalid":
            alt = "Strategy-specific parameters must go under the 'params' dictionary at the root."
        elif error_code == "ast_blocked_import":
            alt = (
                "Use f-strings or concatenation instead of .format(), and import only "
                "from pandas, numpy, math, typing, scipy, dataclasses, collections, "
                "itertools, functools, decimal, statistics, numbers, json."
            )
        elif error_code == "undefined_name":
            alt = "Ensure all names are defined, imported (e.g., 'from typing import Any'), or passed in config/prices."
        return f'\nError detail: "{detail}"\nAllowed alternative: {alt}'
    elif error_code == "smoke_test_failed":
        return (
            "\nSmoke test execution failed. This usually means your code raised an exception at runtime "
            "(e.g., KeyError, IndexError, ValueError), or it returned invalid weight outputs "
            "(e.g. weights exceeding the sum <= 1.0 limit, incorrect shapes, or containing NaN values). "
            "Ensure all operations handle missing or NaN data gracefully, avoid dividing by zero, "
            "check shape consistency, and verify that portfolio weights sum to at most 1.0 at every step."
        )
    elif error_code == "import_failed":
        return (
            "\nImport failed. The Python interpreter failed to load your strategy file. "
            "This is typically caused by syntax errors, invalid syntax in type annotations, or "
            "code crashing at import time. Double-check your syntax and ensure the code compiles "
            "without any side effects during import."
        )
    elif error_code == "signature_mismatch":
        return (
            "\nSignature mismatch. Your strategy must define a function with the exact signature:\n"
            "  `def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:`\n"
            "Ensure the function is exported, name is spelled correctly, and parameters match."
        )
    return ""


def _format_evaluation_report(context: AgentContext) -> str:
    """Format the incumbent's in-sample evaluation report for the LLM prompt.

    Renders walk-forward aggregate metrics (Sharpe, Sortino, IR, drawdown,
    turnover), DSR, regime test results, and per-fold Sharpe range when
    multiple windows exist.

    Args:
        context: The current agent context containing the evaluation report.

    Returns:
        Formatted markdown string describing the evaluation results, or
        ``"First iteration (no prior evaluation report exists)."``
        when no report is available.
    """
    if context.evaluation_report is None:
        return "First iteration (no prior evaluation report exists)."

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
        return (
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
        return str(context.evaluation_report)


def _attempt_fmt(val: Any, default: float = 0.0) -> str:
    """Format a metric value to 4 decimal places for the attempt history table.

    Args:
        val: Numeric value to format. ``None`` or non-numeric values fall
            back to ``default``.
        default: Fallback value when ``val`` is ``None`` or not convertible.

    Returns:
        Formatted string like ``"1.2345"`` or ``"-"`` on conversion failure.
    """
    try:
        return f"{float(val if val is not None else default):.4f}"
    except (TypeError, ValueError):
        return "-"


def _attempt_outcome(r: dict[str, Any]) -> str:
    """Return a human-readable outcome label for an attempt history row.

    Args:
        r: Attempt record dict with keys like ``'committed'``, ``'accepted'``,
            and ``'rejection_reason'``.

    Returns:
        One of ``"committed"``, ``"accepted"``, or ``"rejected (<reason>)"``.
    """
    if r.get("committed"):
        return "✓ committed"
    if r.get("accepted"):
        return "✓ accepted"
    reason = r.get("rejection_reason") or ""
    return f"✗ {str(reason)[:30]}"


def _attempt_fingerprint_repr(r: dict[str, Any]) -> str:
    """Return a truncated string representation of a config fingerprint.

    Args:
        r: Attempt record dict containing ``'config_fingerprint'``.

    Returns:
        Truncated fingerprint string (max 60 chars) for display in the
        attempt history table.
    """
    fp = r.get("config_fingerprint", {})
    s = str(fp)
    return s[:60] + "..." if len(s) > 60 else s


def _attempt_ho_flag(r: dict[str, Any]) -> str:
    """Return holdout confirmation flag for an attempt history row.

    Args:
        r: Attempt record dict containing ``'holdout_confirmed'``.

    Returns:
        ``"HO"`` if holdout was confirmed, empty string otherwise.
    """
    if r.get("holdout_confirmed"):
        return "✓ HO"
    return ""


def _format_attempt_history(context: AgentContext) -> str:
    """Render the attempt history as a markdown table for the LLM prompt.

    Shows all committed attempts and up to 25 most recent non-committed
    attempts. Each row includes iteration number, outcome, target metric,
    DSR, regime status, rejection reason, config fingerprint, and
    holdout flag.

    Args:
        context: Agent context with ``attempt_history`` list.

    Returns:
        Markdown table string, or empty string if no history exists.
    """
    if not context.attempt_history:
        return ""

    committed_rows = [r for r in context.attempt_history if r.get("committed")]
    non_committed_rows = [r for r in context.attempt_history if not r.get("committed")]
    total_non_committed = len(non_committed_rows)
    omitted_count = max(0, total_non_committed - 25)
    displayed_non_committed = non_committed_rows[-25:] if omitted_count > 0 else non_committed_rows
    rows_to_render = committed_rows + displayed_non_committed

    header = "| iter | outcome | target | DSR | regime | reason | config | HO |"
    separator = "|------|---------|--------|-----|--------|--------|--------|-----|"
    table_lines = [header, separator]
    for r in rows_to_render:
        rejection_reason = r.get("rejection_reason") or None
        reason_col = "-" if (r.get("committed") or rejection_reason is None) else str(rejection_reason)[:30]
        row = (
            f"| {r.get('iteration', '')} "
            f"| {_attempt_outcome(r)} "
            f"| {_attempt_fmt(r.get('target_metric_value'))} "
            f"| {_attempt_fmt(r.get('deflated_sharpe'))} "
            f"| {'pass' if r.get('regime_passed') else ('FAIL' if 'regime_passed' in r else '-')} "
            f"| {reason_col} "
            f"| {_attempt_fingerprint_repr(r)} "
            f"| {_attempt_ho_flag(r)} |"
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
    return f"\n## Attempt History\n{table_str}{omit_note}\n"


def _format_previous_validation(
    attempt: dict[str, Any],
    lines: list[str],
    incumbent_code: str = "",
    incumbent_config: str = "",
) -> None:
    """Append validation failure details to the previous attempt section.

    Formats the error code, detail, and a diff of the failed code/config. For
    ``lookahead_detected`` errors, includes an extended explanation
    of common causes and fixes.

    Args:
        attempt: Attempt record dict with validation failure details.
        lines: Mutable list of markdown lines to append to.
        incumbent_code: The incumbent strategy code to diff against.
        incumbent_config: The incumbent config YAML to diff against.
    """
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
        diff = _diff_code(incumbent_code, code)
        if diff:
            lines.append(f"\n**Failed strategy code (diff vs incumbent):**\n```diff\n{diff}\n```")
        else:
            lines.append(f"\n**Failed strategy code:**\n```python\n{code}\n```")
    if config:
        config_diff = _diff_config(incumbent_config, config)
        if config_diff:
            lines.append(f"\n**Failed config (diff vs incumbent):**\n```diff\n{config_diff}\n```")
        else:
            lines.append(f"\n**Failed config:**\n```yaml\n{config}\n```")


def _format_previous_diversity_config(
    attempt: dict[str, Any],
    lines: list[str],
    incumbent_code: str = "",
    incumbent_config: str = "",
) -> None:
    """Append config diversity gate failure details to the previous attempt section.

    Args:
        attempt: Attempt record dict with config diversity rejection details.
        lines: Mutable list of markdown lines to append to.
        incumbent_code: The incumbent strategy code to diff against.
        incumbent_config: The incumbent config YAML to diff against.
    """
    detail = attempt.get("detail", "")
    lines.append(f"**Detail:** {detail}")
    code = attempt.get("candidate_strategy_code", "")
    if code:
        code_diff = _diff_code(incumbent_code, code)
        if code_diff:
            lines.append(f"\n**Rejected strategy code (diff vs incumbent):**\n```diff\n{code_diff}\n```")
        else:
            lines.append(f"\n**Rejected strategy code:**\n```python\n{code}\n```")
    config = attempt.get("candidate_config_yaml", "")
    if config:
        config_diff = _diff_config(incumbent_config, config)
        if config_diff:
            lines.append(f"\n**Rejected config (diff vs incumbent):**\n```diff\n{config_diff}\n```")
        else:
            lines.append(f"\n**Rejected config:**\n```yaml\n{config}\n```")


def _format_previous_eval_error(
    attempt: dict[str, Any],
    lines: list[str],
    incumbent_code: str = "",
    incumbent_config: str = "",
) -> None:
    """Append evaluation error details to the previous attempt section.

    Args:
        attempt: Attempt record dict with evaluation error details.
        lines: Mutable list of markdown lines to append to.
        incumbent_code: The incumbent strategy code to diff against.
        incumbent_config: The incumbent config YAML to diff against.
    """
    detail = attempt.get("detail", "")
    lines.append(f"**Error:** {detail}")
    code = attempt.get("candidate_strategy_code", "")
    config = attempt.get("candidate_config_yaml", "")
    if code:
        diff = _diff_code(incumbent_code, code)
        if diff:
            lines.append(f"\n**Failed strategy code (diff vs incumbent):**\n```diff\n{diff}\n```")
        else:
            lines.append(f"\n**Failed strategy code:**\n```python\n{code}\n```")
    if config:
        config_diff = _diff_config(incumbent_config, config)
        if config_diff:
            lines.append(f"\n**Failed config (diff vs incumbent):**\n```diff\n{config_diff}\n```")
        else:
            lines.append(f"\n**Failed config:**\n```yaml\n{config}\n```")


def _format_previous_diversity_returns(
    attempt: dict[str, Any],
    lines: list[str],
    incumbent_code: str = "",
    incumbent_config: str = "",
) -> None:
    """Append returns correlation diversity failure details to the previous attempt section.

    Includes the detail message, observed metrics, and the rejected config YAML.

    Args:
        attempt: Attempt record dict with returns diversity rejection details.
        lines: Mutable list of markdown lines to append to.
        incumbent_code: The incumbent strategy code to diff against.
        incumbent_config: The incumbent config YAML to diff against.
    """
    detail = attempt.get("detail", "")
    lines.append(f"**Detail:** {detail}")
    metrics = attempt.get("candidate_metrics", {})
    if metrics:
        lines.append("**Observed metrics:**")
        for k, v in metrics.items():
            lines.append(f"  - {k}: {v}")
    code = attempt.get("candidate_strategy_code", "")
    if code:
        code_diff = _diff_code(incumbent_code, code)
        if code_diff:
            lines.append(f"\n**Rejected strategy code (diff vs incumbent):**\n```diff\n{code_diff}\n```")
        else:
            lines.append(f"\n**Rejected strategy code:**\n```python\n{code}\n```")
    config = attempt.get("candidate_config_yaml", "")
    if config:
        config_diff = _diff_config(incumbent_config, config)
        if config_diff:
            lines.append(f"\n**Rejected config (diff vs incumbent):**\n```diff\n{config_diff}\n```")
        else:
            lines.append(f"\n**Rejected config:**\n```yaml\n{config}\n```")


def _format_previous_gate(
    attempt: dict[str, Any],
    lines: list[str],
    incumbent_code: str = "",
    incumbent_config: str = "",
) -> None:
    """Append gate rejection details to the previous attempt section.

    Includes rejection reason, failed gate name, candidate metrics,
    and the diff of the failed code/config.

    Args:
        attempt: Attempt record dict with gate rejection details.
        lines: Mutable list of markdown lines to append to.
        incumbent_code: The incumbent strategy code to diff against.
        incumbent_config: The incumbent config YAML to diff against.
    """
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
        diff = _diff_code(incumbent_code, code)
        if diff:
            lines.append(f"\n**Failed strategy code (diff vs incumbent):**\n```diff\n{diff}\n```")
        else:
            lines.append(f"\n**Failed strategy code:**\n```python\n{code}\n```")
    if config:
        config_diff = _diff_config(incumbent_config, config)
        if config_diff:
            lines.append(f"\n**Failed config (diff vs incumbent):**\n```diff\n{config_diff}\n```")
        else:
            lines.append(f"\n**Failed config:**\n```yaml\n{config}\n```")


def _format_previous_attempt(context: AgentContext) -> str:
    """Build the previous attempt failure section for the LLM prompt.

    Dispatches to the appropriate sub-formatter based on the failure stage
    (validation, diversity_config, eval_error, diversity_returns, gate).
    Appends a diagnostic instruction telling the LLM to fix the specific
    error rather than regenerating the same code.

    Args:
        context: Agent context with ``last_attempt`` dict.

    Returns:
        Formatted markdown string for the previous attempt section,
        or empty string if no previous attempt exists.
    """
    if context.last_attempt is None:
        return ""

    attempt = context.last_attempt
    stage = attempt.get("stage", "unknown")
    lines: list[str] = ["## Previous Attempt Result", f"**Stage:** {stage}"]

    if stage == "validation":
        _format_previous_validation(
            attempt, lines, incumbent_code=context.strategy_code, incumbent_config=context.config_yaml
        )
    elif stage == "diversity_config":
        _format_previous_diversity_config(
            attempt, lines, incumbent_code=context.strategy_code, incumbent_config=context.config_yaml
        )
    elif stage == "eval_error":
        _format_previous_eval_error(
            attempt, lines, incumbent_code=context.strategy_code, incumbent_config=context.config_yaml
        )
    elif stage == "diversity_returns":
        _format_previous_diversity_returns(
            attempt, lines, incumbent_code=context.strategy_code, incumbent_config=context.config_yaml
        )
    elif stage == "gate":
        _format_previous_gate(
            attempt, lines, incumbent_code=context.strategy_code, incumbent_config=context.config_yaml
        )
    else:
        detail = attempt.get("detail", attempt.get("rejection_reason", ""))
        if detail:
            lines.append(f"**Detail:** {detail}")

    lines.append(
        "\n**Diagnose the failure above. Do NOT regenerate the same code or config. "
        "Your fix must address the specific error shown.**"
    )
    return "\n".join(lines) + "\n\n"


def _format_performance_target(context: AgentContext) -> str:
    """Render the performance target section showing incumbent metrics and gate limits.

    When an evaluation report exists, shows the incumbent's in-sample
    walk-forward metrics (Sharpe, Sortino, IR, drawdown, turnover) and
    the hard gate limits the candidate must pass. Always includes a
    warning about the hidden OOS holdout confirmation gate.

    Args:
        context: Agent context with optional ``evaluation_report``.

    Returns:
        Formatted markdown string with performance targets and gate limits.
    """
    if context.evaluation_report is not None:
        rep = context.evaluation_report
        m = rep.in_sample_metrics
        return (
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
        return (
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


def _format_repair_request(context: AgentContext) -> str:
    """Render the repair request section when the LLM's previous edit failed preflight.

    Shows the failed code, failed config, error code, and error detail,
    along with a contextual remediation hint. Instructs the LLM to make
    only the minimal fix required.

    Args:
        context: Agent context with ``repair_request`` dict.

    Returns:
        Formatted markdown string for the repair section, or empty string
        when no repair request is active.
    """
    if not context.repair_request:
        return ""

    req = context.repair_request
    failed_code = req.get("failed_code", "")
    failed_config = req.get("failed_config_yaml", "")
    err_code = req.get("error_code", "")
    err_detail = req.get("error_detail", "")
    hint = _format_previous_attempt_hint(err_code, err_detail)
    diff = _diff_code(context.strategy_code, failed_code) if failed_code else ""
    code_block = f"```diff\n{diff}\n```" if diff else f"```python\n{failed_code}\n```"
    return (
        f"## Repair Request\n"
        f"Your previous proposal failed preflight validation with the following error:\n"
        f"**Error Code:** {err_code}\n"
        f"**Error Detail:** {err_detail}\n\n"
        f"**Failed Code (diff vs incumbent):**\n{code_block}\n\n"
        f"**Failed Config:**\n```yaml\n{failed_config}\n```\n\n"
        f"**Instruction:** Fix ONLY this validation error. Make the minimal change. "
        f"Do not redesign the strategy.{hint}\n\n"
        f"---\n\n"
    )


def _format_last_iteration_failures(context: AgentContext) -> str:
    """Render a summary of all candidate failures from the previous iteration.

    Lists each failed candidate with its stage, error code, detail, and
    parameters, helping the LLM avoid repeating the same mistakes across
    parallel candidates.

    Args:
        context: Agent context with ``last_iteration_failures`` list.

    Returns:
        Formatted markdown string with numbered failure entries,
        or empty string if no failures occurred.
    """
    if not context.last_iteration_failures:
        return ""

    lines = [
        "## Previous Iteration — All Candidates",
        "The following failures were observed across the parallel candidates generated in the last iteration:",
    ]
    for idx, fail in enumerate(context.last_iteration_failures):
        stage = fail.get("stage", "unknown")
        err = fail.get("error_code")
        err_str = f" ({err})" if err else ""
        detail = fail.get("detail") or ""
        params = fail.get("params") or {}
        params_str = f" | Parameters: {params}" if params else ""
        lines.append(f"{idx + 1}. **Stage:** {stage}{err_str} | **Detail:** {detail}{params_str}")
    return "\n".join(lines) + "\n\n"


def build_messages(
    context: AgentContext,
    cache_supported: bool = False,
) -> list[dict[str, Any]]:
    """Build the system and user message payload for the LLM completion API.

    Args:
        context: Immutable context defining the current optimization state.
        cache_supported: When True, emit Anthropic-style cache_control breakpoints
            on the stable prefix so the provider caches SYSTEM_PROMPT + program_text
            and the user-message stable body (iteration context, strategy code, config,
            evaluation, instructions).

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
    eval_report_str = _format_evaluation_report(context)

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

    attempt_history_section = _format_attempt_history(context)

    diversity_warning = ""
    if context.n_historical_configs > 0:
        diversity_warning = (
            f"\n## Diversity Warning\n"
            f"There are {context.n_historical_configs} attempted strategy variants "
            f"tracked for this asset universe. Strategies whose return streams correlate "
            f">95% with any prior attempt will be rejected after backtesting. "
            f"Ensure your strategy behaves differently — not just has different parameters.\n"
        )

    previous_attempt_section = _format_previous_attempt(context)
    performance_target_section = _format_performance_target(context)

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

    repair_request_section = _format_repair_request(context)

    directive_section = ""
    if context.directive:
        directive_section = f"\n\n## Candidate Directive\nThis candidate MUST: {context.directive}\n"

    explored_config_section = ""
    if context.explored_config_summary:
        explored_config_section = (
            f"\n## Explored Config Space\n"
            f"{context.explored_config_summary}\n\n"
            f"> [!WARNING]\n"
            f"> Any config with similarity > {settings.diversity_config_threshold} to ANY of these "
            f"is auto-rejected. Choose parameter values outside these ranges. "
            f"Specifically, pick numeric values ≥ ~25% away from every tried value, "
            f"or change the parameter set structurally.\n"
        )

    last_iteration_failures_section = _format_last_iteration_failures(context)

    stable_body = (
        f"## Iteration\n"
        f"Current Loop Iteration: {context.iteration}\n\n"
        f"{mode_section}"
        f"\n\n## Lessons\n"
        f"{injected_lessons}"
        f"{warning_str}"
        f"\n## Current Strategy Code\n"
        f"```python\n{context.strategy_code}\n```\n\n"
        f"## Current Config\n"
        f"```yaml\n{context.config_yaml}\n```\n\n"
        f"## Latest Evaluation\n"
        f"{eval_report_str}{attempt_history_section}{explored_config_section}"
        f"{diversity_warning}"
        f"{last_iteration_failures_section}{previous_attempt_section}"
        f"{performance_target_section}"
        f"## Instructions\n"
        f"Improve the strategy per the objective. Optimize parameters, signal\n"
        f"logic, or asset weights.\n"
        f"Your response must be returned as a JSON object containing the keys:\n"
        f'- "strategy_code": Complete, updated Python source code for the strategy.\n'
        f'- "config_yaml": Complete, updated YAML parameters content.\n'
        f'- "reasoning": Concise explanation of the quantitative logic and changes.\n'
        f'- "lessons_text": Optional concise markdown lesson update, or null.\n'
        f"  Use null when there is no new reusable lesson. Do not repeat the existing lessons document.\n\n"
        f"Strict JSON/Formatting Constraint: The response must be a single,\n"
        f"valid JSON object. Do not wrap the JSON object in markdown code\n"
        f"block markers (such as ```json or ```). The very first character\n"
        f"immediately following the </think> tag must be the opening {{ of the\n"
        f"JSON payload.\n"
    )

    dynamic_tail = f"{repair_request_section}{directive_section}"

    if cache_supported:
        user_content: list[dict[str, Any]] = [_text_block(stable_body, cache=True)]
        if dynamic_tail:
            user_content.append(_text_block(dynamic_tail))
    else:
        stripped = f"{stable_body}{dynamic_tail}".strip()
        user_content = stripped  # type: ignore[assignment]

    user_message: dict[str, Any] = {
        "role": "user",
        "content": user_content,
    }

    return [system_message, user_message]
