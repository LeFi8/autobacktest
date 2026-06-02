"""Strategy signature and output contract verification."""

import inspect
from types import ModuleType

import pandas as pd

REQUIRED_FUNCTION = "generate_signals"
LEVERAGE_TOLERANCE = 1e-5


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
    # strictly keyword-only / vararg / kwarg (Finding 12)
    p1 = params[0]
    p2 = params[1]

    allowed_kinds = (
        inspect.Parameter.POSITIONAL_ONLY,
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
    )
    if p1.kind not in allowed_kinds:
        return False, "First parameter must be positional (prices)."
    if p2.kind not in allowed_kinds:
        return False, "Second parameter must be positional (config)."

    for param in params[2:]:
        if param.default is inspect.Parameter.empty and param.kind not in (
            inspect.Parameter.VAR_POSITIONAL,
            inspect.Parameter.VAR_KEYWORD,
        ):
            return False, (
                f"'{REQUIRED_FUNCTION}' has an invalid signature. "
                f"Parameter '{param.name}' is required but has no default value. "
                f"Only the first two parameters (prices, config) can be required."
            )

    return True, None


def validate_output(
    weights: pd.DataFrame, tickers: list[str], expected_index: pd.Index | None = None
) -> tuple[bool, str | None]:
    """Verify the validity of a strategy's generated weights DataFrame.

    Checks:
    1. Returns a pandas DataFrame.
    2. No NaN values are present.
    3. Long-only constraint: no negative weights.
    4. Leverage constraint: row sums must be <= 1.0 (with absolute tolerance).
    5. Columns are a subset of the permitted universe.
    6. Index dates are a subset of expected price history dates.

    Args:
        weights: Strategy signal weights DataFrame.
        tickers: Permitted asset tickers in the config universe.
        expected_index: Optional pandas Index representing daily trading dates.

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

    # Check for duplicate columns
    if weights.columns.duplicated().any():
        duplicated_cols = list(set(weights.columns[weights.columns.duplicated()]))
        return False, f"Strategy weights contain duplicate columns: {duplicated_cols}"

    # Check that all columns are numeric
    for col in weights.columns:
        if not pd.api.types.is_numeric_dtype(weights[col]):
            return (
                False,
                f"Strategy weights column '{col}' must be numeric, got {weights[col].dtype}.",
            )

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
            f"Strategy weights row sums exceed 1.0. Max sum found: {offending_rows.max()} on {offending_rows.idxmax()}"
        )

    # 5. Must have at least one non-zero weight (not all zeros)
    if weights.abs().max().max() < 1e-7:
        return False, ("Strategy weights must not be all zeros (must have at least one non-zero weight).")

    # 6. Index validation (Finding 14)
    if not isinstance(weights.index, pd.DatetimeIndex):
        return (
            False,
            f"Expected DatetimeIndex for weights DataFrame, got {type(weights.index)}.",
        )

    if expected_index is not None:
        invalid_dates = weights.index.difference(expected_index)
        if not invalid_dates.empty:
            return (
                False,
                (f"Strategy weights index contains dates not in the price history: {list(invalid_dates[:5])}"),
            )

    return True, None
