"""Strategy signature and output contract verification."""

import inspect
from types import ModuleType

import pandas as pd

REQUIRED_FUNCTION = "generate_signals"
LEVERAGE_TOLERANCE = 1e-7


def validate_signature(module: ModuleType) -> tuple[bool, str | None]:
    """Verify that a loaded module implements the correct generate_signals signature.

    Args:
        module: The dynamically imported module.

    Returns:
        tuple[bool, str | None]: (True, None) if contract satisfied,
                                 (False, error_msg) otherwise.
    """
    if not hasattr(module, REQUIRED_FUNCTION):
        return False, f"Module must export a '{REQUIRED_FUNCTION}' function."

    func = getattr(module, REQUIRED_FUNCTION)
    if not callable(func):
        return (
            False,
            f"'{REQUIRED_FUNCTION}' must be a callable function, not {type(func)}.",
        )

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    if len(params) < 2:
        return False, (
            f"'{REQUIRED_FUNCTION}' must accept at least 2 parameters: "
            f"(prices: pd.DataFrame, config: dict/StrategyConfig). Got {len(params)}."
        )

    # First two parameters must not be positional-only keyword mismatch or
    # strictly keyword-only
    p1 = params[0]
    p2 = params[1]

    if p1.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
        return False, "First parameter must be positional (prices)."
    if p2.kind in (inspect.Parameter.KEYWORD_ONLY, inspect.Parameter.VAR_KEYWORD):
        return False, "Second parameter must be positional (config)."

    return True, None


def validate_output(
    weights: pd.DataFrame, tickers: list[str]
) -> tuple[bool, str | None]:
    """Verify the validity of a strategy's generated weights DataFrame.

    Checks:
    1. Returns a pandas DataFrame.
    2. No NaN values are present.
    3. Long-only constraint: no negative weights.
    4. Leverage constraint: row sums must be <= 1.0 (with absolute tolerance).
    5. Columns are a subset of the permitted universe.

    Args:
        weights: Strategy signal weights DataFrame.
        tickers: Permitted asset tickers in the config universe.

    Returns:
        tuple[bool, str | None]: (True, None) if valid, (False, error_msg) otherwise.
    """
    if not isinstance(weights, pd.DataFrame):
        return (
            False,
            f"Expected pandas DataFrame from signals generator, got {type(weights)}.",
        )

    if weights.empty:
        return False, "Weights DataFrame is empty."

    # 1. No NaNs allowed
    if weights.isna().any().any():
        return False, "Strategy weights must not contain NaN values."

    # 2. Check permitted columns (columns must be a subset of tickers)
    invalid_cols = [col for col in weights.columns if col not in tickers]
    if invalid_cols:
        return (
            False,
            f"Strategy weights contain tickers outside config universe: {invalid_cols}",
        )

    # 3. Long-only constraint: all weights >= 0
    # Allow a tiny negative tolerance for float precision issues (e.g. -1e-7)
    if (weights < -1e-7).any().any():
        return False, "Strategy weights must be non-negative (long-only)."

    # 4. Leverage constraint: row sums <= 1.0 (plus tiny float tolerance)
    row_sums = weights.sum(axis=1)
    if (row_sums > 1.0 + LEVERAGE_TOLERANCE).any():
        offending_rows = row_sums[row_sums > 1.0 + LEVERAGE_TOLERANCE]
        return False, (
            f"Strategy weights row sums exceed 1.0. "
            f"Max sum found: {offending_rows.max()} on {offending_rows.idxmax()}"
        )

    return True, None
