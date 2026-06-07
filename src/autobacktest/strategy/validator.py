"""Strategy pre-flight validation and sandboxed execution safety checks.

Provides the ``preflight()`` function which runs eight validation layers:
path traversal security, AST whitelist scan, Pydantic config validation,
dynamic import, signature verification, smoke testing with synthetic prices,
lookahead bias sniffing, and undefined-name AST scanning.  Validation runs
inside a sandboxed subprocess with memory limits and timeout protection.
"""

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import ValidationError as PydanticValidationError

from autobacktest.config import settings

# Import sandbox runner and AST linter helpers with explicit as re-exports
from autobacktest.strategy.ast_linter import (
    _build_function_scope as _build_function_scope,
)
from autobacktest.strategy.ast_linter import (
    _calculate_complexity as _calculate_complexity,
)
from autobacktest.strategy.ast_linter import (
    _check_ast as _check_ast,
)
from autobacktest.strategy.ast_linter import (
    _count_node_lines as _count_node_lines,
)
from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.sandbox_runner import (
    SandboxTimeoutError as SandboxTimeoutError,
)
from autobacktest.strategy.sandbox_runner import (
    _generate_synthetic_prices as _generate_synthetic_prices,
)
from autobacktest.strategy.sandbox_runner import (
    _run_validation_in_subprocess as _run_validation_in_subprocess,
)
from autobacktest.strategy.sandbox_runner import (
    timeout_sandbox as timeout_sandbox,
)

__all__ = [
    "SandboxTimeoutError",
    "ValidationError",
    "ValidationResult",
    "_build_function_scope",
    "_calculate_complexity",
    "_check_ast",
    "_count_node_lines",
    "_generate_synthetic_prices",
    "compare_signals_to_incumbent",
    "preflight",
    "timeout_sandbox",
]


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
    UNDEFINED_NAME = "undefined_name"


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
    """Run all eight pre-flight validations on a target strategy and config.

    Validates path traversal safety, AST import whitelist, Pydantic config
    schema, dynamic import, function signature, synthetic-price smoke test,
    lookahead bias sniff test, and lookahead shift test.  The runtime checks
    (import, smoke, lookahead) execute inside a sandboxed subprocess with
    memory limits and timeout protection.

    Args:
        strategy_name: The strategy name (stem, without ``.py`` or ``.yaml``).
        strategies_dir: Directory containing strategy ``.py`` files.
        configs_dir: Directory containing strategy ``.yaml`` config files.

    Returns:
        ValidationResult: ``passed=True`` when all checks succeed; contains
        the specific error code and detail when a check fails.
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
    if cfg_res.detail is None:
        return cfg_res

    # Parse config dictionary from Pydantic StrategyConfig
    config_model = cfg_res.detail  # Holds StrategyConfig instance

    # 3-6. Subprocess Sandboxed Dynamic Validation
    return _run_validation_in_subprocess(strategy_name, strategy_path, config_model)


def _check_config(path: Path) -> ValidationResult:
    """Validate YAML configuration schema against the unified Pydantic model.

    Parses the YAML file and validates it against ``StrategyConfig``,
    which enforces parameter boundaries, types, and ``extra="forbid"``.

    Args:
        path: Path to the YAML configuration file.

    Returns:
        ValidationResult: ``passed=True`` with the ``StrategyConfig``
        instance stored in ``detail`` when valid; error detail on failure.
    """
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


def compare_signals_to_incumbent(
    _strategy_name: str,
    candidate_code: str,
    candidate_config_yaml: str,
    incumbent_code: str,
    strategies_dir: Path,
    configs_dir: Path,
    epsilon: float = 1e-6,
) -> tuple[bool, float]:
    """Compare candidate signals against incumbent on synthetic prices.

    Returns:
        (is_identical, max_abs_diff) where is_identical is True if max_abs_diff < epsilon.
    """
    import tempfile

    strategies_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    fd_py, temp_py_path = tempfile.mkstemp(suffix=".py", prefix="compare_", dir=strategies_dir)
    fd_yaml, temp_yaml_path = tempfile.mkstemp(suffix=".yaml", prefix="compare_", dir=configs_dir)

    try:
        os.close(fd_py)
        os.close(fd_yaml)

        temp_py = Path(temp_py_path)
        temp_yaml = Path(temp_yaml_path)

        temp_py.write_text(candidate_code, encoding="utf-8")
        temp_yaml.write_text(candidate_config_yaml, encoding="utf-8")

        temp_name = temp_py.stem
        config_model = StrategyConfig.from_yaml(temp_yaml)

        res = _run_validation_in_subprocess(
            temp_name,
            temp_py,
            config_model,
            incumbent_code=incumbent_code,
        )

        if res.passed and res.detail and str(res.detail).startswith("max_abs_weight_diff:"):
            diff_str = str(res.detail).split(":")[1]
            diff = float(diff_str)
            return diff < epsilon, diff
        else:
            return False, 1.0

    finally:
        if Path(temp_py_path).exists():
            Path(temp_py_path).unlink()
        if Path(temp_yaml_path).exists():
            Path(temp_yaml_path).unlink()
