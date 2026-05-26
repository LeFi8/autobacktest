"""System and user prompts construction for the LLM strategy optimizer."""

from autobacktest.llm.base import AgentContext
from autobacktest.strategy.validator import ALLOWED_IMPORTS

# System prompt outlining constraints and role
sorted_imports = sorted(ALLOWED_IMPORTS)
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
   and summing to at most 1.0 (sum <= 1.0) for every rebalance day.
6. The output strategy code and config YAML MUST be complete file contents, NOT diffs.
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

    user_content = f"""## Iteration
Current Loop Iteration: {context.iteration}

## Objective
{context.program_text}

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

## Instructions
Improve the strategy per the objective. Optimize parameters, signal
logic, or asset weights.
Your response must be returned as a JSON object containing the keys:
- "strategy_code": Complete, updated Python source code for the strategy.
- "config_yaml": Complete, updated YAML parameters content.
- "reasoning": Concise explanation of the quantitative logic and changes.
"""

    user_message = {
        "role": "user",
        "content": user_content.strip(),
    }

    return [system_message, user_message]
