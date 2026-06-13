"""Sandboxed strategy subprocess execution and validation runner."""

import json
import os
import re
import signal
import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager, suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from autobacktest.config import settings
from autobacktest.strategy.config_schema import StrategyConfig

if TYPE_CHECKING:
    from autobacktest.strategy.validator import ValidationResult

# API key patterns to redact from error output
_SANITIZE_PATTERNS = [
    (re.compile(r"(sk-[a-zA-Z0-9]{20,})"), r"sk-***REDACTED***"),
    (re.compile(r"(sk-[a-zA-Z0-9]{32,})"), r"sk-***REDACTED***"),
]


def _sanitize_detail(text: str) -> str:
    """Redact potential API keys and credentials from error messages.

    Applies regex patterns to mask OpenAI-style ``sk-...`` keys.

    Args:
        text: Raw error message text.

    Returns:
        str: Sanitised text with keys replaced by ``sk-***REDACTED***``.
    """
    if not text:
        return text
    for pattern, replacement in _SANITIZE_PATTERNS:
        text = pattern.sub(replacement, text)
    return text


class SandboxTimeoutError(Exception):
    """Exception raised when strategy execution times out."""

    pass


@contextmanager
def timeout_sandbox(
    seconds: int = 15,
    memory_limit_bytes: int = 1 * 1024 * 1024 * 1024,
) -> Generator[None, None, None]:
    """Lightweight execution timeout and memory sandbox context manager.

    Uses ``SIGALRM`` (main thread only) for timeout enforcement and
    ``resource.RLIMIT_AS`` for virtual memory hard limits.  Falls back
    gracefully when ``resource`` is unavailable (non-POSIX platforms).

    Args:
        seconds: Execution timeout in seconds (default: 15).
        memory_limit_bytes: Virtual memory soft limit (default: 1 GB).

    Yields:
        None: Context in which the wrapped code executes under sandbox.
    """
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


def _generate_synthetic_prices(tickers: list[str], n_days: int, seed: int = 42) -> pd.DataFrame:
    """Helper to generate geometric random walk price DataFrame.

    Creates business-day-indexed prices starting from 2023-01-01 with
    daily drift 0.02% and volatility 1%.

    Args:
        tickers: List of ticker symbols.
        n_days: Number of trading days to generate.
        seed: Random seed for reproducibility (default: 42).

    Returns:
        pd.DataFrame: Synthetic close prices with DatetimeIndex.
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    prices = pd.DataFrame(index=dates)
    for ticker in tickers:
        steps = rng.normal(0.0002, 0.01, n_days)
        prices[ticker] = 100.0 * np.exp(np.cumsum(steps))
    return prices


def _run_validation_in_subprocess(
    strategy_name: str,
    strategy_path: Path,
    config: StrategyConfig,
    incumbent_code: str | None = None,
) -> "ValidationResult":
    """Run dynamic import, signature, smoke, and lookahead tests in a sandboxed subprocess.

    Spawns a separate Python process with a restricted ``PATH``/``PYTHONPATH``
    to sandbox the strategy code.  The subprocess runner (defined inline as
    ``runner_code``) performs:
    1. Dynamic import from the strategy file.
    2. Signature verification (``generate_signals(prices, config)``).
    3. Smoke test (756 days synthetic prices).
    4. Optional incumbent comparison for identical-behavior guard.
    5. Lookahead sniff test (compare signals with vs. without future data).
    6. Lookahead shift test (shift prices by 1 day, verify signal shift).

    Args:
        strategy_name: The strategy name.
        strategy_path: Path to the strategy ``.py`` file.
        config: Pydantic-validated ``StrategyConfig``.
        incumbent_code: Incumbent strategy source for identical-behavior comparison.

    Returns:
        ValidationResult: Pass/fail with error code and detail.
    """
    from autobacktest.strategy.validator import ValidationError, ValidationResult

    payload = {
        "strategy_name": strategy_name,
        "strategy_path": str(strategy_path),
        "config_dict": config.to_flat_dict(),
        "universe": config.universe,
        "sandbox_timeout": settings.sandbox_timeout,
    }
    if incumbent_code is not None:
        payload["incumbent_code"] = incumbent_code

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

        # Incumbent signals comparison (identical-behavior guard)
        if "incumbent_code" in payload:
            try:
                inc_code = payload["incumbent_code"]
                inc_module = ModuleType(strategy_name + "_incumbent")
                inc_spec = importlib.util.spec_from_file_location(strategy_name + "_incumbent", strategy_path)
                if inc_spec is not None:
                    inc_module.__spec__ = inc_spec
                    inc_module.__loader__ = inc_spec.loader
                sys.modules[strategy_name + "_incumbent"] = inc_module
                inc_code_obj = compile(inc_code, "<incumbent>", "exec")
                exec(inc_code_obj, inc_module.__dict__)
                
                with timeout_sandbox(seconds=sandbox_timeout):
                    inc_weights = inc_module.generate_signals(prices, config_dict)
                
                max_abs_diff = float(np.abs(weights.values - inc_weights.values).max())
                return {"passed": True, "error_code": None, "detail": f"max_abs_weight_diff:{max_abs_diff}"}
            except Exception as e:
                return {
                    "passed": False,
                    "error_code": "smoke_test_failed",
                    "detail": f"Incumbent signal comparison failed: {e}",
                }
            finally:
                sys.modules.pop(strategy_name + "_incumbent", None)

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
                    max_diff = float(diff.values.max())
                    worst_col = str(diff.max(axis=0).idxmax())
                    msg = (
                        f"Lookahead bias sniff test failed. Signal for '{worst_col}' at "
                        f"'{first_bad_date}' changed when future data was appended to the price history "
                        f"(max weight delta {max_diff:.4f}). Ensure no feature depends on "
                        "future rows (e.g. avoid .shift(-n), forward-fills, or rolling "
                        "windows without .shift(1))."
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

        # 5. Lookahead Shift Test
        try:
            changes = weights_base.diff().abs().sum(axis=1) > 0
            rebalance_ratio = float(changes.sum() / len(weights_base))
            if rebalance_ratio >= 0.15:
                prices_shifted = prices_base.shift(1).dropna()
                with timeout_sandbox(seconds=sandbox_timeout):
                    weights_shifted = module.generate_signals(prices_shifted, config_dict)

                w_base_shifted = weights_base.shift(1).dropna()
                w_shifted_clean = weights_shifted.dropna()

                common_shift_idx = w_base_shifted.index.intersection(w_shifted_clean.index)
                if not common_shift_idx.empty and len(common_shift_idx) > 260:
                    warmup_idx = common_shift_idx.tolist()[260:]
                    w_b_s = w_base_shifted.loc[warmup_idx]
                    w_s_c = w_shifted_clean.loc[warmup_idx]

                    if not np.allclose(w_b_s.values, w_s_c.values, rtol=0.0, atol=1e-5):
                        shift_diff = np.abs(w_b_s.values - w_s_c.values)
                        max_shift_diff = float(shift_diff.max())
                        n_bad_cells = int((shift_diff > 1e-5).sum())
                        return {
                            "passed": False,
                            "error_code": "lookahead_detected",
                            "detail": (
                                f"Lookahead Shift Test failed: signals did not shift consistently"
                                f" with price history shift — max post-warmup discrepancy {max_shift_diff:.4f}"
                                f" across {n_bad_cells} weight cell(s) (tolerance 1e-5)."
                                f" Ensure all features use .shift(1) and avoid .shift(-n)."
                            ),
                        }
        except Exception as e:
            return {
                "passed": False,
                "error_code": "smoke_test_failed",
                "detail": f"Lookahead Shift Test execution failed: {e}",
            }

        return {"passed": True, "error_code": None, "detail": None}
    finally:
        sys.modules.pop(strategy_name, None)

result = run_checks()
print("__RESULT__" + json.dumps(result))
"""

    safe_env = {
        k: v
        for k, v in os.environ.items()
        if k in {"PATH", "PYTHONPATH", "HOME", "USER"} or k.startswith("AUTOBACKTEST_")
    }
    safe_env["PYTHONPATH"] = os.pathsep.join(sys.path)

    try:
        proc = subprocess.run(
            [sys.executable, "-c", runner_code],
            input=json.dumps(payload, default=str),
            capture_output=True,
            text=True,
            timeout=25,
            env=safe_env,
            check=False,
        )
        if proc.returncode != 0:
            err_msg = _sanitize_detail(proc.stderr.strip() or f"Subprocess exited with non-zero code {proc.returncode}")
            return ValidationResult(
                passed=False,
                error_code=ValidationError.SMOKE_TEST_FAILED,
                detail=f"Subprocess execution crashed: {err_msg}",
            )

        result_line = None
        for line in proc.stdout.splitlines():
            if line.startswith("__RESULT__"):
                result_line = line[len("__RESULT__") :]
                break
        if result_line is None:
            raise ValueError(f"Subprocess produced no result line. stderr: {_sanitize_detail(proc.stderr.strip())!r}")
        res_data = json.loads(result_line)
        err_code = None
        if res_data["error_code"]:
            err_code = ValidationError(res_data["error_code"])

        return ValidationResult(
            passed=res_data["passed"],
            error_code=err_code,
            detail=_sanitize_detail(res_data["detail"]),
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
            detail=_sanitize_detail(f"Subprocess sandboxing orchestration failed: {e}"),
        )
