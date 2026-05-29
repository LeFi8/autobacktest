"""System and user prompts construction for the LLM strategy optimizer."""

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
   >90% similarity (same params, same asset sets, same structure), the
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

    # Format the latest evaluation report
    if context.evaluation_report is not None:
        try:
            eval_report_str = context.evaluation_report.to_json(indent=2)
        except Exception:
            # Fallback if evaluation_report cannot be serialized
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

    # Diversity warning section
    diversity_warning = ""
    if context.n_historical_configs > 0:
        diversity_warning = (
            f"\n## Diversity Warning\n"
            f"There are {context.n_historical_configs} historical strategy variants "
            f"tracked for this asset universe. The config similarity gate will reject "
            f"proposals with >90% fingerprint overlap.\n"
        )

    user_content = f"""## Iteration
Current Loop Iteration: {context.iteration}

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
{eval_report_str}
{diversity_warning}
## Instructions
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
