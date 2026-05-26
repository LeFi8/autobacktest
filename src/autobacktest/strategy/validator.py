import ast
import importlib.util
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.contract import validate_output, validate_signature

# Whitelisted libraries for AI strategy imports
ALLOWED_IMPORTS = {
    "pandas",
    "numpy",
    "math",
    "typing",
    "collections",
    "functools",
    "itertools",
    "dataclasses",
    "decimal",
    "statistics",
    "numbers",
    "json",
}

# Forbidden variables, functions, and names that compromise sandboxing
FORBIDDEN_NAMES = {
    "exec",
    "eval",
    "compile",
    "open",
    "__import__",
    "globals",
    "locals",
    "getattr",
    "setattr",
    "delattr",
    "__builtins__",
}


class ValidationError(StrEnum):
    """Pre-flight validation error types."""

    AST_BLOCKED_IMPORT = "ast_blocked_import"
    CONFIG_SCHEMA_INVALID = "config_schema_invalid"
    IMPORT_FAILED = "import_failed"
    SIGNATURE_MISMATCH = "signature_mismatch"
    SMOKE_TEST_FAILED = "smoke_test_failed"
    LOOKAHEAD_DETECTED = "lookahead_detected"


@dataclass
class ValidationResult:
    """Pre-flight validation check result details."""

    passed: bool
    error_code: ValidationError | None = None
    detail: Any = None


def preflight(
    strategy_name: str,
    strategies_dir: Path,
    configs_dir: Path,
) -> ValidationResult:
    """Run all six pre-flight validations on a target strategy and config.

    Args:
        strategy_name: Name of the strategy to validate.
        strategies_dir: Absolute path to directory containing strategy .py files.
        configs_dir: Absolute path to directory containing config .yaml files.

    Returns:
        ValidationResult: The status of the validation suite.
    """
    strategy_path = strategies_dir / f"{strategy_name}.py"
    config_path = configs_dir / f"{strategy_name}.yaml"

    if not strategy_path.exists():
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=f"Strategy file not found at: {strategy_path}",
        )

    # 1. AST Static Whitelist check
    ast_res = _check_ast(strategy_path)
    if not ast_res.passed:
        return ast_res

    # 2. Pydantic Config Validation
    cfg_res = _check_config(config_path)
    if not cfg_res.passed or cfg_res.detail is None:
        return cfg_res

    # Parse config dictionary from Pydantic StrategyConfig
    config_model = cfg_res.detail  # Holds StrategyConfig instance

    # 3. Dynamic Import
    import_res = _check_import(strategy_name, strategy_path)
    if not import_res.passed or import_res.detail is None:
        return import_res

    module = import_res.detail

    # 4. Signature check
    sig_ok, sig_err = validate_signature(module)
    if not sig_ok:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SIGNATURE_MISMATCH,
            detail=sig_err,
        )

    # 5. Smoke test with 60d synthetic prices
    smoke_res = _check_smoke(module, config_model)
    if not smoke_res.passed:
        return smoke_res

    # 6. Lookahead sniff test
    lookahead_res = _check_lookahead(module, config_model)
    if not lookahead_res.passed:
        return lookahead_res

    return ValidationResult(passed=True)


def _check_ast(path: Path) -> ValidationResult:
    """Parse strategy code via AST and block non-whitelisted imports or unsafe calls."""
    try:
        content = path.read_text(encoding="utf-8")
        tree = ast.parse(content)
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=f"AST parsing syntax error: {e}",
        )

    for node in ast.walk(tree):
        # Inspect imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_module = alias.name.split(".")[0]
                if root_module not in ALLOWED_IMPORTS:
                    msg = (
                        f"Import of non-whitelisted module '{alias.name}' "
                        f"is strictly blocked."
                    )
                    return ValidationResult(
                        passed=False,
                        error_code=ValidationError.AST_BLOCKED_IMPORT,
                        detail=msg,
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                root_module = node.module.split(".")[0]
                if root_module not in ALLOWED_IMPORTS:
                    msg = (
                        f"Import from non-whitelisted module '{node.module}' "
                        f"is strictly blocked."
                    )
                    return ValidationResult(
                        passed=False,
                        error_code=ValidationError.AST_BLOCKED_IMPORT,
                        detail=msg,
                    )

        # Inspect forbidden variables, functions, and builtin names
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            msg = f"Use of forbidden name or builtin '{node.id}' is strictly blocked."
            return ValidationResult(
                passed=False,
                error_code=ValidationError.AST_BLOCKED_IMPORT,
                detail=msg,
            )

        # Inspect forbidden attributes (prevents dunder sandbox escapes)
        elif isinstance(node, ast.Attribute) and (
            node.attr in FORBIDDEN_NAMES or node.attr.startswith("__")
        ):
            msg = (
                f"Use of forbidden attribute or dunder property '{node.attr}' "
                f"is strictly blocked."
            )
            return ValidationResult(
                passed=False,
                error_code=ValidationError.AST_BLOCKED_IMPORT,
                detail=msg,
            )

    return ValidationResult(passed=True)


def _check_config(path: Path) -> ValidationResult:
    """Validate YAML configuration schema against the unified Pydantic model."""
    try:
        cfg = StrategyConfig.from_yaml(path)
        return ValidationResult(passed=True, detail=cfg)
    except PydanticValidationError as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.CONFIG_SCHEMA_INVALID,
            detail=f"Config validation error: {e}",
        )
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.CONFIG_SCHEMA_INVALID,
            detail=f"Config load error: {e}",
        )


def _check_import(strategy_name: str, path: Path) -> ValidationResult:
    """Dynamically construct module loader and exec the strategy module."""
    try:
        spec = importlib.util.spec_from_file_location(strategy_name, path)
        if spec is None or spec.loader is None:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.IMPORT_FAILED,
                detail=f"Failed to resolve loader spec for strategy: {strategy_name}",
            )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return ValidationResult(passed=True, detail=module)
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=f"Dynamic import execution failed: {e}",
        )


def _generate_synthetic_prices(
    tickers: list[str], n_days: int, seed: int = 42
) -> pd.DataFrame:
    """Helper to generate geometric random walk price DataFrame."""
    np.random.seed(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(index=dates)
    for ticker in tickers:
        steps = np.random.normal(0.0002, 0.01, n_days)
        prices[ticker] = 100.0 * np.exp(np.cumsum(steps))
    return prices


def _check_smoke(module: Any, config: StrategyConfig) -> ValidationResult:
    """Run signals on 756 days of synthetic prices and assert output constraints."""
    try:
        tickers = config.universe
        prices = _generate_synthetic_prices(tickers, n_days=756)

        # Merge core params and custom params to match CLI's dictionary passing
        config_dict = config.model_dump()
        params = config_dict.pop("params", {})
        config_dict.update(params)

        weights = module.generate_signals(prices, config_dict)
        ok, err = validate_output(weights, tickers)
        if not ok:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.SMOKE_TEST_FAILED,
                detail=f"Smoke test output constraints failed: {err}",
            )
        return ValidationResult(passed=True)
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SMOKE_TEST_FAILED,
            detail=f"Smoke test execution exception: {e}",
        )


def _check_lookahead(module: Any, config: StrategyConfig) -> ValidationResult:
    """Sniff out lookahead bias by validating sub-window signals stability."""
    try:
        tickers = config.universe
        config_dict = config.model_dump()
        params = config_dict.pop("params", {})
        config_dict.update(params)

        # 1. Base run on 756 days
        prices_base = _generate_synthetic_prices(tickers, n_days=756, seed=123)
        weights_base = module.generate_signals(prices_base, config_dict)

        # 2. Append future pricing data (additional 10 days of different random noise)
        prices_future = _generate_synthetic_prices(tickers, n_days=766, seed=456)
        # Re-align original 756 days to match base prices exactly to prevent drift
        prices_future.iloc[:756] = prices_base.iloc[:756]

        weights_future = module.generate_signals(prices_future, config_dict)

        # Compare weights returned for the original 756-day sub-window
        common_idx = weights_base.index.intersection(weights_future.index)
        if common_idx.empty:
            return ValidationResult(
                passed=True
            )  # sparse rebalance dates might not overlap

        w_base = weights_base.loc[common_idx]
        w_fut = weights_future.loc[common_idx]

        # Use float tolerance for numerical precision differences (e.g. 1e-7)
        if not np.allclose(w_base.values, w_fut.values, atol=1e-7):
            # Locate first discrepancy date
            diff = np.abs(w_base - w_fut)
            bad_row = diff.max(axis=1) > 1e-7
            first_bad_date = common_idx[bad_row][0].strftime("%Y-%m-%d")
            msg = (
                f"Lookahead bias sniff test failed. Rebalance signals at "
                f"'{first_bad_date}' changed when future data was "
                f"appended to the price history."
            )
            return ValidationResult(
                passed=False,
                error_code=ValidationError.LOOKAHEAD_DETECTED,
                detail=msg,
            )

        return ValidationResult(passed=True)
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.LOOKAHEAD_DETECTED,
            detail=f"Lookahead bias sniff test threw exception: {e}",
        )
