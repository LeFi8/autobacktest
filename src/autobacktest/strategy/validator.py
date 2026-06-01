import ast
import json
import signal
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from autobacktest.config import settings
from autobacktest.strategy.config_schema import StrategyConfig

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
    AST_LINE_LIMIT_EXCEEDED = "ast_line_limit_exceeded"
    AST_CYCLOMATIC_COMPLEXITY_EXCEEDED = "ast_cyclomatic_complexity_exceeded"
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
) -> Generator[None, None, None]:
    """Lightweight execution timeout and memory sandbox context manager."""
    import threading

    resource: Any = None
    try:
        import resource as _resource

        resource = _resource
    except ImportError:
        pass

    def signal_handler(_signum: int, _frame: Any) -> None:
        raise SandboxTimeoutError("Strategy execution timed out (exceeded limit).")

    # Register the signal handler only if on the main thread
    use_signals = threading.current_thread() is threading.main_thread()
    original_handler = None
    if use_signals:
        try:
            original_handler = signal.signal(signal.SIGALRM, signal_handler)
            signal.alarm(seconds)
        except ValueError:
            use_signals = False

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
        if use_signals:
            with suppress(Exception):
                signal.alarm(0)
                if original_handler is not None:
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
            detail=(f"Path traversal or validation error for strategy '{strategy_name}': {e}"),
        )

    # AST TOCTOU Protection: Read once (Finding 8)
    try:
        # Verify file size limit
        file_size_kb = strategy_path.stat().st_size / 1024.0
        if file_size_kb > settings.max_file_size_kb:
            return ValidationResult(
                passed=False,
                error_code=ValidationError.IMPORT_FAILED,
                detail=(
                    f"Strategy file exceeds size limit of {settings.max_file_size_kb}KB (actual: {file_size_kb:.1f}KB)"
                ),
            )
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

    # 3-6. Subprocess Sandboxed Dynamic Validation
    return _run_validation_in_subprocess(strategy_name, strategy_path, config_model)


def _run_validation_in_subprocess(
    strategy_name: str,
    strategy_path: Path,
    config: StrategyConfig,
) -> ValidationResult:
    """Run dynamic import, signature, smoke, and lookahead tests in a sandboxed subprocess."""
    payload = {
        "strategy_name": strategy_name,
        "strategy_path": str(strategy_path),
        "config_dict": config.to_flat_dict(),
        "universe": config.universe,
        "sandbox_timeout": settings.sandbox_timeout,
    }

    # Define the runner code block as a multi-line string
    runner_code = """
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
from types import ModuleType
import importlib.util

# Read input payload
payload = json.loads(sys.stdin.read())
strategy_name = payload["strategy_name"]
strategy_path = Path(payload["strategy_path"])
config_dict = payload["config_dict"]
universe = payload["universe"]
sandbox_timeout = payload.get("sandbox_timeout", 15)

from autobacktest.strategy.contract import validate_signature, validate_output
from autobacktest.strategy.validator import timeout_sandbox, SandboxTimeoutError, _generate_synthetic_prices

def run_checks():
    try:
        # 1. Dynamic Import
        content = strategy_path.read_text(encoding="utf-8")
        module = ModuleType(strategy_name)
        module.__file__ = str(strategy_path)
        spec = importlib.util.spec_from_file_location(strategy_name, strategy_path)
        if spec is not None:
            module.__spec__ = spec
            module.__loader__ = spec.loader
        sys.modules[strategy_name] = module
        code_obj = compile(content, str(strategy_path), "exec")
        exec(code_obj, module.__dict__)
    except Exception as e:
        return {
            "passed": False,
            "error_code": "import_failed",
            "detail": f"Dynamic import execution failed: {e}",
        }

    try:
        # 2. Signature Check
        sig_ok, sig_err = validate_signature(module)
        if not sig_ok:
            return {
                "passed": False,
                "error_code": "signature_mismatch",
                "detail": sig_err,
            }

        # 3. Smoke Test (756 days)
        try:
            prices = _generate_synthetic_prices(universe, n_days=756)
            with timeout_sandbox(seconds=sandbox_timeout):
                weights = module.generate_signals(prices, config_dict)
            ok, err = validate_output(weights, universe, expected_index=prices.index)
            if not ok:
                return {
                    "passed": False,
                    "error_code": "smoke_test_failed",
                    "detail": f"Smoke test output constraints failed: {err}",
                }
        except SandboxTimeoutError as e:
            return {
                "passed": False,
                "error_code": "smoke_test_failed",
                "detail": str(e),
            }
        except MemoryError as e:
            return {
                "passed": False,
                "error_code": "smoke_test_failed",
                "detail": f"Strategy execution exceeded memory limit: {e}",
            }
        except Exception as e:
            return {
                "passed": False,
                "error_code": "smoke_test_failed",
                "detail": f"Smoke test execution exception: {e}",
            }

        # 4. Lookahead Sniff Test
        try:
            prices_base = _generate_synthetic_prices(universe, n_days=756, seed=123)
            with timeout_sandbox(seconds=sandbox_timeout):
                weights_base = module.generate_signals(prices_base, config_dict)

            rng_future = np.random.default_rng(456)
            future_dates = pd.date_range(prices_base.index[-1] + pd.offsets.BDay(), periods=10, freq="B")
            prices_future_ext = pd.DataFrame(index=future_dates, columns=universe)
            for ticker in universe:
                steps = rng_future.normal(0.0002, 0.01, 10)
                base_last = prices_base[ticker].iloc[-1]
                prices_future_ext[ticker] = base_last * np.exp(np.cumsum(steps))

            prices_future = pd.concat([prices_base, prices_future_ext])
            with timeout_sandbox(seconds=sandbox_timeout):
                weights_future = module.generate_signals(prices_future, config_dict)

            common_idx = weights_base.index.intersection(weights_future.index)
            if common_idx.empty:
                return {
                    "passed": False,
                    "error_code": "lookahead_detected",
                    "detail": "Lookahead bias detected: no overlapping rebalance dates.",
                }

            w_base = weights_base.loc[common_idx]
            w_fut = weights_future.loc[common_idx]

            if w_base.isna().any().any() or w_fut.isna().any().any():
                return {
                    "passed": False,
                    "error_code": "smoke_test_failed",
                    "detail": "Lookahead bias sniff test failed: strategy weights contain NaNs.",
                }

            if w_base.shape != w_fut.shape:
                return {
                    "passed": False,
                    "error_code": "lookahead_detected",
                    "detail": (
                        f"Lookahead bias sniff test failed: strategy weights shape "
                        f"changed from {w_base.shape} to {w_fut.shape}."
                    ),
                }

            if not w_base.columns.equals(w_fut.columns):
                return {
                    "passed": False,
                    "error_code": "lookahead_detected",
                    "detail": "Lookahead bias sniff test failed: strategy columns diverged.",
                }

            if not np.allclose(w_base.values, w_fut.values, rtol=0.0, atol=1e-7):
                diff = np.abs(w_base - w_fut)
                bad_row = diff.max(axis=1) > 1e-7
                if bad_row.any():
                    first_bad_date = common_idx[bad_row.values][0].strftime("%Y-%m-%d")
                    msg = (
                        f"Lookahead bias sniff test failed. Rebalance signals at "
                        f"'{first_bad_date}' changed when future data was appended to the price history."
                    )
                else:
                    msg = "Lookahead bias sniff test failed."
                return {"passed": False, "error_code": "lookahead_detected", "detail": msg}

        except SandboxTimeoutError as e:
            return {
                "passed": False,
                "error_code": "smoke_test_failed",
                "detail": str(e),
            }
        except MemoryError as e:
            return {
                "passed": False,
                "error_code": "smoke_test_failed",
                "detail": f"Strategy execution exceeded memory limit: {e}",
            }
        except Exception as e:
            return {
                "passed": False,
                "error_code": "smoke_test_failed",
                "detail": f"Lookahead bias sniff test execution failed: {e}",
            }

        return {"passed": True, "error_code": None, "detail": None}
    finally:
        sys.modules.pop(strategy_name, None)

result = run_checks()
# Use a sentinel prefix so any print() calls in the strategy don't corrupt parsing.
print("__RESULT__" + json.dumps(result))
"""

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            input=json.dumps(payload, default=str),
            capture_output=True,
            text=True,
            timeout=25,  # Bounded wait with safety buffer
        )
        if proc.returncode != 0:
            err_msg = proc.stderr.strip() or f"Subprocess exited with non-zero code {proc.returncode}"
            return ValidationResult(
                passed=False,
                error_code=ValidationError.SMOKE_TEST_FAILED,
                detail=f"Subprocess execution crashed: {err_msg}",
            )

        # Find the sentinel result line; ignore any print() output from the strategy.
        result_line = None
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                result_line = line[len("__RESULT__") :]
                break
        if result_line is None:
            raise ValueError(f"Subprocess produced no result line. stderr: {proc.stderr.strip()!r}")
        res_data = json.loads(result_line)
        err_code = None
        if res_data["error_code"]:
            err_code = ValidationError(res_data["error_code"])

        return ValidationResult(
            passed=res_data["passed"],
            error_code=err_code,
            detail=res_data["detail"],
        )

    except subprocess.TimeoutExpired:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.SMOKE_TEST_FAILED,
            detail="Strategy execution timed out in sandboxed subprocess (exceeded limit).",
        )
    except Exception as e:
        return ValidationResult(
            passed=False,
            error_code=ValidationError.IMPORT_FAILED,
            detail=f"Subprocess sandboxing orchestration failed: {e}",
        )


def _get_attribute_chain(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    elif isinstance(node, ast.Attribute):
        val = _get_attribute_chain(node.value)
        if val is not None:
            return f"{val}.{node.attr}"
    return None


def _count_node_lines(node: ast.AST) -> int:
    """Measure the line count of a parsed AST node."""
    end: int | None = getattr(node, "end_lineno", None)
    start: int | None = getattr(node, "lineno", None)
    if end is not None and start is not None:
        return end - start + 1
    return 0


def _calculate_complexity(node: ast.AST) -> int:
    """Calculate McCabe-style cyclomatic complexity of a function AST.

    Counts decision points (if/for/while/and/or/ternary/except-handler/comprehensions)
    and adds 1 for the base path.
    """
    complexity = 1
    for child in ast.walk(node):
        if isinstance(
            child,
            (ast.If, ast.For, ast.While, ast.ExceptHandler, ast.IfExp),
        ):
            complexity += 1
        elif isinstance(child, ast.BoolOp):
            if isinstance(child.op, (ast.And, ast.Or)):
                complexity += len(child.values) - 1
        elif isinstance(
            child,
            (ast.ListComp, ast.DictComp, ast.SetComp, ast.GeneratorExp),
        ):
            complexity += 1

    return complexity


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
                if root_module not in settings.parsed_safe_imports:
                    msg = f"Import of non-whitelisted module '{alias.name}' is strictly blocked."
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
                if root_module not in settings.parsed_safe_imports:
                    msg = f"Import from non-whitelisted module '{node.module}' is strictly blocked."
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
                msg = f"Use of forbidden attribute or dunder property '{node.attr}' is strictly blocked."
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
                            f"Use of forbidden attribute or dunder property '{part}' in '{chain}' is strictly blocked."
                        )
                        return ValidationResult(
                            passed=False,
                            error_code=ValidationError.AST_BLOCKED_IMPORT,
                            detail=msg,
                        )

    # Complexity and line-count check (second pass over function defs)
    for func_node in ast.walk(tree):
        if isinstance(func_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            line_count = _count_node_lines(func_node)
            if line_count > settings.max_function_lines:
                return ValidationResult(
                    passed=False,
                    error_code=ValidationError.AST_LINE_LIMIT_EXCEEDED,
                    detail=(
                        f"Function '{func_node.name}' has {line_count} lines, "
                        f"exceeding the limit of {settings.max_function_lines}."
                    ),
                )
            complexity = _calculate_complexity(func_node)
            if complexity > settings.max_cyclomatic_complexity:
                return ValidationResult(
                    passed=False,
                    error_code=ValidationError.AST_CYCLOMATIC_COMPLEXITY_EXCEEDED,
                    detail=(
                        f"Function '{func_node.name}' has cyclomatic complexity "
                        f"{complexity}, exceeding the limit of "
                        f"{settings.max_cyclomatic_complexity}."
                    ),
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


def _generate_synthetic_prices(tickers: list[str], n_days: int, seed: int = 42) -> pd.DataFrame:
    """Helper to generate geometric random walk price DataFrame."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(index=dates)
    for ticker in tickers:
        steps = rng.normal(0.0002, 0.01, n_days)
        prices[ticker] = 100.0 * np.exp(np.cumsum(steps))
    return prices
