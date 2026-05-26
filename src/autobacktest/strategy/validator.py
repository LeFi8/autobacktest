import ast
import importlib.util
import signal
import sys
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from types import ModuleType
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
    "vars",
    "breakpoint",
    # numpy / pandas sandboxing escapes
    "load",
    "save",
    "savez",
    "savez_compressed",
    "memmap",
    "fromfile",
    "tofile",
    "loadtxt",
    "genfromtxt",
    "fromregex",
    "DataSource",
    "read_csv",
    "read_table",
    "read_fwf",
    "to_csv",
    "read_json",
    "to_json",
    "read_excel",
    "to_excel",
    "read_pickle",
    "to_pickle",
    "read_parquet",
    "to_parquet",
    "read_hdf",
    "to_hdf",
    "read_feather",
    "to_feather",
    "read_xml",
    "to_xml",
    "read_html",
    "to_html",
    "read_sql",
    "read_sql_table",
    "read_sql_query",
    "to_sql",
    "read_clipboard",
    "to_clipboard",
    "io",
    "get_handle",
    "lib",
    "npyio",
    "HDFStore",
    "ExcelWriter",
    "ExcelFile",
    "read_sas",
    "read_spss",
    "read_gbq",
    "read_stata",
    "read_orc",
    "to_stata",
    "to_orc",
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


class SandboxTimeoutError(Exception):
    """Exception raised when strategy execution times out."""

    pass


@contextmanager
def timeout_sandbox(
    seconds: int = 15,
    memory_limit_bytes: int = 1 * 1024 * 1024 * 1024,
):
    """Lightweight execution timeout and memory sandbox context manager."""
    try:
        import resource
    except ImportError:
        resource = None

    def signal_handler(_signum, _frame):
        raise SandboxTimeoutError("Strategy execution timed out (exceeded limit).")

    # Register the signal handler
    original_handler = signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)

    # Set virtual memory limit (soft limit) to protect against memory OOM attacks
    old_limits = None
    if resource is not None:
        with suppress(Exception):
            soft, hard = resource.getrlimit(resource.RLIMIT_AS)
            resource.setrlimit(resource.RLIMIT_AS, (memory_limit_bytes, hard))
            old_limits = (soft, hard)

    try:
        yield
    finally:
        # Cancel the alarm and restore the original handler
        signal.alarm(0)
        signal.signal(signal.SIGALRM, original_handler)
        if resource is not None and old_limits is not None:
            with suppress(Exception):
                resource.setrlimit(resource.RLIMIT_AS, old_limits)


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
    # 1. Path traversal check (Finding 2)
    try:
        strategy_path = strategies_dir / f"{strategy_name}.py"
        config_path = configs_dir / f"{strategy_name}.yaml"

        # Resolve paths safely to check traversal first
        resolved_strategies_dir = strategies_dir.resolve()
        resolved_strategy_path = strategy_path.resolve()
        if resolved_strategies_dir not in resolved_strategy_path.parents:
            raise ValueError("path traversal detected outside strategies directory.")

        resolved_configs_dir = configs_dir.resolve()
        resolved_config_path = config_path.resolve()
        if resolved_configs_dir not in resolved_config_path.parents:
            raise ValueError("path traversal detected outside configs directory.")

        if not strategy_path.exists():
            return ValidationResult(
                passed=False,
                error_code=ValidationError.IMPORT_FAILED,
                detail=f"Strategy file not found at: {strategy_path}",
            )

        if not config_path.exists():
            return ValidationResult(
                passed=False,
                error_code=ValidationError.IMPORT_FAILED,
                detail=f"Strategy config file not found at: {config_path}",
            )

    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=(
                f"Path traversal or validation error for "
                f"strategy '{strategy_name}': {e}"
            ),
        )

    # AST TOCTOU Protection: Read once (Finding 8)
    try:
        content = strategy_path.read_text(encoding="utf-8")
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=f"Failed to read strategy file: {e}",
        )

    # 1. AST Static Whitelist check
    ast_res = _check_ast(content)
    if not ast_res.passed:
        return ast_res

    # 2. Pydantic Config Validation
    cfg_res = _check_config(config_path)
    if not cfg_res.passed:
        return cfg_res
    assert cfg_res.detail is not None

    # Parse config dictionary from Pydantic StrategyConfig
    config_model = cfg_res.detail  # Holds StrategyConfig instance

    # 3. Dynamic Import using pre-read content
    import_res = _check_import(strategy_name, strategy_path, content)
    if not import_res.passed:
        return import_res
    assert import_res.detail is not None

    module = import_res.detail

    # 4. Signature check
    sig_ok, sig_err = validate_signature(module)
    if not sig_ok:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SIGNATURE_MISMATCH,
            detail=sig_err,
        )

    # 5. Smoke test with 756d synthetic prices
    smoke_res = _check_smoke(module, config_model)
    if not smoke_res.passed:
        return smoke_res

    # 6. Lookahead sniff test
    lookahead_res = _check_lookahead(module, config_model)
    if not lookahead_res.passed:
        return lookahead_res

    return ValidationResult(passed=True)


def _get_attribute_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        val = _get_attribute_chain(node.value)
        if val is not None:
            return f"{val}.{node.attr}"
    return None


def _check_ast(content: str) -> ValidationResult:
    """Parse strategy code via AST and block non-whitelisted imports or unsafe calls."""
    try:
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
                # Inspect imported alias names (Finding 4)
                if alias.asname and alias.asname in FORBIDDEN_NAMES:
                    return ValidationResult(
                        passed=False,
                        error_code=ValidationError.AST_BLOCKED_IMPORT,
                        detail=f"Import alias '{alias.asname}' is strictly blocked.",
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
                # Inspect imported names and aliases (Finding 4)
                for alias in node.names:
                    if alias.name == "*":
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail="Wildcard imports (*) are strictly blocked.",
                        )
                    imported_name = alias.name.split(".")[0]
                    if imported_name in FORBIDDEN_NAMES:
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail=f"Forbidden import '{alias.name}' is blocked.",
                        )
                    if alias.asname and alias.asname in FORBIDDEN_NAMES:
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail=f"Import alias '{alias.asname}' is blocked.",
                        )

        # Inspect forbidden variables, functions, and builtin names
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_NAMES:
            msg = f"Use of forbidden name or builtin '{node.id}' is strictly blocked."
            return ValidationResult(
                passed=False,
                error_code=ValidationError.AST_BLOCKED_IMPORT,
                detail=msg,
            )

        # Inspect forbidden attributes (prevents dunder escapes & chained - Finding 5)
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_NAMES or node.attr.startswith("__"):
                msg = (
                    f"Use of forbidden attribute or dunder property '{node.attr}' "
                    f"is strictly blocked."
                )
                return ValidationResult(
                    passed=False,
                    error_code=ValidationError.AST_BLOCKED_IMPORT,
                    detail=msg,
                )
            chain = _get_attribute_chain(node)
            if chain:
                parts = chain.split(".")
                for part in parts:
                    if part in FORBIDDEN_NAMES or part.startswith("__"):
                        msg = (
                            f"Use of forbidden attribute or dunder property '{part}' "
                            f"in '{chain}' is strictly blocked."
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
    except FileNotFoundError as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=f"Config load error: {e}",
        )
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.CONFIG_SCHEMA_INVALID,
            detail=f"Config load error: {e}",
        )


def _check_import(strategy_name: str, path: Path, content: str) -> ValidationResult:
    """Dynamically construct module and compile/exec checked content (Finding 8)."""
    try:
        module = ModuleType(strategy_name)
        module.__file__ = str(path)

        spec = importlib.util.spec_from_file_location(strategy_name, path)
        if spec is not None:
            module.__spec__ = spec
            module.__loader__ = spec.loader

        sys.modules[strategy_name] = module

        code_obj = compile(content, str(path), "exec")
        exec(code_obj, module.__dict__)
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
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(index=dates)
    for ticker in tickers:
        steps = rng.normal(0.0002, 0.01, n_days)
        prices[ticker] = 100.0 * np.exp(np.cumsum(steps))
    return prices


def _check_smoke(module: Any, config: StrategyConfig) -> ValidationResult:
    """Run signals on 756 days of synthetic prices and assert output constraints."""
    try:
        tickers = config.universe
        prices = _generate_synthetic_prices(tickers, n_days=756)

        config_dict = config.to_flat_dict()

        # Runtime Sandbox Execution (Finding 9)
        with timeout_sandbox(seconds=15):
            weights = module.generate_signals(prices, config_dict)

        ok, err = validate_output(weights, tickers, expected_index=prices.index)
        if not ok:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.SMOKE_TEST_FAILED,
                detail=f"Smoke test output constraints failed: {err}",
            )
        return ValidationResult(passed=True)
    except SandboxTimeoutError as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SMOKE_TEST_FAILED,
            detail=str(e),
        )
    except MemoryError as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SMOKE_TEST_FAILED,
            detail=f"Strategy execution exceeded memory limit: {e}",
        )
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
        config_dict = config.to_flat_dict()

        # 1. Base run on 756 days
        prices_base = _generate_synthetic_prices(tickers, n_days=756, seed=123)
        with timeout_sandbox(seconds=15):
            weights_base = module.generate_signals(prices_base, config_dict)

        # 2. Append continuous future pricing data
        # (additional 10 days of different random noise)
        rng_future = np.random.default_rng(456)
        future_dates = pd.date_range(
            prices_base.index[-1] + pd.offsets.BDay(), periods=10, freq="B"
        )
        prices_future_ext = pd.DataFrame(index=future_dates, columns=tickers)
        for ticker in tickers:
            steps = rng_future.normal(0.0002, 0.01, 10)
            base_last = prices_base[ticker].iloc[-1]
            prices_future_ext[ticker] = base_last * np.exp(np.cumsum(steps))

        prices_future = pd.concat([prices_base, prices_future_ext])
        with timeout_sandbox(seconds=15):
            weights_future = module.generate_signals(prices_future, config_dict)

        # Compare weights returned for the original 756-day sub-window
        common_idx = weights_base.index.intersection(weights_future.index)
        if common_idx.empty:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.LOOKAHEAD_DETECTED,
                detail=(
                    "Lookahead bias detected: no overlapping rebalance dates "
                    "between the base run and the future-extended run."
                ),
            )

        w_base = weights_base.loc[common_idx]
        w_fut = weights_future.loc[common_idx]

        if w_base.isna().any().any() or w_fut.isna().any().any():
            return ValidationResult(
                passed=False,
                error_code=ValidationError.SMOKE_TEST_FAILED,
                detail=(
                    "Lookahead bias sniff test failed: strategy weights contain NaNs."
                ),
            )

        if w_base.shape != w_fut.shape:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.LOOKAHEAD_DETECTED,
                detail=(
                    f"Lookahead bias sniff test failed: strategy weights shape "
                    f"changed from {w_base.shape} to {w_fut.shape} when future "
                    f"data was appended."
                ),
            )

        if not w_base.columns.equals(w_fut.columns):
            return ValidationResult(
                passed=False,
                error_code=ValidationError.LOOKAHEAD_DETECTED,
                detail=(
                    "Lookahead bias sniff test failed: strategy weights columns "
                    "diverged when future data was appended."
                ),
            )

        # Use float tolerance (e.g. 1e-7 - Finding 17)
        if not np.allclose(w_base.values, w_fut.values, rtol=0.0, atol=1e-7):
            # Locate first discrepancy date (Finding 6)
            diff = np.abs(w_base - w_fut)
            bad_row = diff.max(axis=1) > 1e-7
            if bad_row.any():
                first_bad_date = common_idx[bad_row][0].strftime("%Y-%m-%d")
                msg = (
                    f"Lookahead bias sniff test failed. Rebalance signals at "
                    f"'{first_bad_date}' changed when future data was "
                    f"appended to the price history."
                )
            else:
                msg = (
                    "Lookahead bias sniff test failed: strategy weights shape, "
                    "columns, or values diverged when future data was "
                    "appended to the price history."
                )
            return ValidationResult(
                passed=False,
                error_code=ValidationError.LOOKAHEAD_DETECTED,
                detail=msg,
            )

        return ValidationResult(passed=True)
    except SandboxTimeoutError as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SMOKE_TEST_FAILED,
            detail=str(e),
        )
    except MemoryError as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SMOKE_TEST_FAILED,
            detail=f"Strategy execution exceeded memory limit: {e}",
        )
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SMOKE_TEST_FAILED,
            detail=f"Lookahead bias sniff test execution failed: {e}",
        )
