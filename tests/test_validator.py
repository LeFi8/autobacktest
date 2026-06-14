"""Unit tests for the pre-flight strategy and config validator."""

import ast
from pathlib import Path

import pytest

from autobacktest.config import settings
from autobacktest.strategy.validator import (
    ValidationError,
    ValidationResult,
    _build_function_scope,
    _calculate_complexity,
    _check_ast,
    _count_node_lines,
    preflight,
)


@pytest.fixture
def mock_dirs(tmp_path: Path) -> tuple[Path, Path]:
    """Helper fixture creating temp directories for strategy and config files."""
    strat_dir = tmp_path / "strategies"
    conf_dir = tmp_path / "configs"
    strat_dir.mkdir()
    conf_dir.mkdir()
    return strat_dir, conf_dir


def test_validator_valid_strategy(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that a valid strategy passes all pre-flight checks."""
    strat_dir, conf_dir = mock_dirs

    # Write a simple passing strategy
    strat_file = strat_dir / "simple.py"
    strat_file.write_text(
        """
import pandas as pd
import numpy as np
import json

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Always invest equally in SPY and BIL
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in weights.columns:
        weights["SPY"] = 0.5
    if "BIL" in weights.columns:
        weights["BIL"] = 0.5
    return weights
""",
        encoding="utf-8",
    )

    # Write its config
    conf_file = conf_dir / "simple.yaml"
    conf_file.write_text(
        """
universe:
  - SPY
  - BIL
benchmark: SPY
momentum_lookback: 12
params:
  offensive_universe:
    - SPY
""",
        encoding="utf-8",
    )

    res = preflight("simple", strat_dir, conf_dir)
    assert res.passed
    assert res.error_code is None


def test_validator_ast_blocks_forbidden_import(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies AST static analysis blocks forbidden module imports."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "bad_import.py"
    strat_file.write_text(
        """
import os
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return pd.DataFrame()
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "bad_import.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("bad_import", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT
    assert "Import of non-whitelisted module" in res.detail


def test_validator_ast_blocks_forbidden_call(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies AST blocks dynamic invocation calls like exec/eval."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "bad_call.py"
    strat_file.write_text(
        """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    eval("print('hack')")
    return pd.DataFrame()
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "bad_call.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("bad_call", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT
    assert "forbidden name or builtin" in res.detail


def test_validator_invalid_config_schema(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies validator correctly detects invalid config YAML files."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "bad_config.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return pd.DataFrame()
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "bad_config.yaml"
    # Bad config because universe is empty list (violates min_length=1)
    conf_file.write_text(
        """
universe: []
benchmark: SPY
""",
        encoding="utf-8",
    )

    res = preflight("bad_config", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.CONFIG_SCHEMA_INVALID
    assert "Config validation error" in res.detail


def test_validator_import_failure_syntax_error(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies syntax errors fail at the AST parser level."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "bad_syntax.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    this is invalid python syntax
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "bad_syntax.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("bad_syntax", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.IMPORT_FAILED
    assert "AST parsing syntax error" in res.detail


def test_validator_signature_mismatch(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies signature checker catches incorrect function signature contracts."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "bad_sig.py"
    strat_file.write_text(
        """
import pandas as pd
# Mismatch: missing config argument
def generate_signals(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame()
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "bad_sig.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("bad_sig", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.SIGNATURE_MISMATCH


def test_validator_smoke_test_nan_rejection(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that smoke test catches invalid NaNs in returned weights."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "nan_weights.py"
    strat_file.write_text(
        """
import pandas as pd
import numpy as np

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
    return weights
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "nan_weights.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("nan_weights", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.SMOKE_TEST_FAILED
    assert "must not contain NaN values" in res.detail


def test_validator_lookahead_sniff_detection(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies lookahead bias sniffer catches future leakage."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "lookahead.py"
    strat_file.write_text(
        """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    # LEAKAGE: weights on day t read prices from the VERY LAST day
    # of the entire DataFrame (future data!)
    last_val = prices.iloc[-1]["SPY"]
    if "SPY" in weights.columns:
        weights["SPY"] = last_val / (last_val + 1.0)
    return weights
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "lookahead.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("lookahead", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.LOOKAHEAD_DETECTED
    assert "changed when future data was appended" in res.detail


def test_validator_ast_blocks_security_bypasses(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies AST blocks various sandbox escape and bypass techniques."""
    strat_dir, conf_dir = mock_dirs

    conf_file = conf_dir / "sec_test.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    # 1. Test __builtins__.exec
    strat_file = strat_dir / "sec_test.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(_prices: pd.DataFrame, _config: dict) -> pd.DataFrame:
    __builtins__.exec("import os")
    return pd.DataFrame()
""",
        encoding="utf-8",
    )
    res = preflight("sec_test", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT
    assert "forbidden attribute or dunder property" in res.detail

    # 2. Test open() builtin
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(_prices: pd.DataFrame, _config: dict) -> pd.DataFrame:
    f = open("exploit.txt", "w")
    return pd.DataFrame()
""",
        encoding="utf-8",
    )
    res = preflight("sec_test", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT
    assert "forbidden name or builtin" in res.detail

    # 3. Test dunder escape (.__class__)
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(_prices: pd.DataFrame, _config: dict) -> pd.DataFrame:
    x = ().__class__.__base__
    return pd.DataFrame()
""",
        encoding="utf-8",
    )
    res = preflight("sec_test", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT
    assert "forbidden attribute or dunder property" in res.detail


def test_validator_file_size_limit(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that strategy files exceeding size limits are rejected."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "large_file.py"
    # Write a file exceeding the configured limit dynamically
    padding_size = (settings.max_file_size_kb + 1) * 1024
    padding = "#" * padding_size
    strat_file.write_text(
        f"""
import pandas as pd
{padding}
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return pd.DataFrame()
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "large_file.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("large_file", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.IMPORT_FAILED
    assert "exceeds size limit" in res.detail


# ---------------------------------------------------------------------------
# AST Complexity guard unit tests (Tasks 1.3, 1.4, 1.5)
# ---------------------------------------------------------------------------


def _parse_func(code: str) -> ast.FunctionDef:
    """Parse a single function definition and return its AST node."""
    tree = ast.parse(code)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            return node
    raise ValueError("No function definition found in code")


def test_count_node_lines_one_line() -> None:
    """A one-line function has line count = 1."""
    func = _parse_func("def f(): pass\n")
    assert _count_node_lines(func) == 1


def test_count_node_lines_multi_line() -> None:
    """A multi-line function returns the correct physical line count."""
    code = """
def f():
    x = 1
    y = 2
    z = x + y
    return z
"""
    func = _parse_func(code)
    assert _count_node_lines(func) == 5


def test_count_node_lines_large() -> None:
    """A function exceeding the default limit is detected by the helper."""
    lines = ["def f():"]
    for i in range(150):
        lines.append(f"    x{i} = {i}")
    code = "\n".join(lines)
    func = _parse_func(code)
    assert _count_node_lines(func) > settings.max_function_lines


def test_calculate_complexity_flat() -> None:
    """A function with no decision points has complexity 1."""
    func = _parse_func("def f(): return 42\n")
    assert _calculate_complexity(func) == 1


def test_calculate_complexity_one_if() -> None:
    """A single if-statement adds 1 to complexity."""
    func = _parse_func("def f(x):\n    if x > 0:\n        return x\n    return 0\n")
    assert _calculate_complexity(func) == 2


def test_calculate_complexity_nested_control_flow() -> None:
    """Nested if/for/while branches all contribute to complexity."""
    func = _parse_func(
        "def f(x, n):\n"
        "    if x > 0:\n"
        "        for i in range(n):\n"
        "            while i < 10:\n"
        "                i += 1\n"
        "    else:\n"
        "        return 0\n"
        "    return x\n"
    )
    # 1 (base) + 1 (if) + 1 (for) + 1 (while) = 4
    assert _calculate_complexity(func) == 4


def test_calculate_complexity_boolean_operators() -> None:
    """ast.And and ast.Or each count as decision points."""
    func = _parse_func("def f(a, b, c):\n    if a and b or c:\n        return 1\n    return 0\n")
    # 1 (base) + 1 (if) + 1 (and) + 1 (or) = 4
    assert _calculate_complexity(func) == 4


def test_calculate_complexity_comprehension() -> None:
    """List comprehensions count as decision points.

    Note: the 'if' clause inside a comprehension is an expression
    (Compare node), not an ast.If, so it does NOT add an extra branch.
    """
    func = _parse_func("def f(items):\n    return [x for x in items if x > 0]\n")
    # 1 (base) + 1 (list comp) = 2
    assert _calculate_complexity(func) == 2


def test_preflight_rejects_overly_long_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """preflight rejects a function exceeding max_function_lines."""
    lines = ["import pandas as pd\n", "\n", "def generate_signals(prices, config):\n"]
    for i in range(120):
        lines.append(f"    x{i} = {i}\n")
    lines.append("    return pd.DataFrame()\n")
    code = "".join(lines)
    # Override to a low limit for testing
    monkeypatch.setattr(settings, "max_function_lines", 50)
    mock_dirs_result = _run_with_code(code)
    assert mock_dirs_result is not None
    res = mock_dirs_result
    assert not res.passed
    assert res.error_code == ValidationError.AST_LINE_LIMIT_EXCEEDED
    assert "exceeding the limit" in res.detail


def test_preflight_rejects_overly_complex_function(monkeypatch: pytest.MonkeyPatch) -> None:
    """preflight rejects a function exceeding max_cyclomatic_complexity."""
    code = (
        "import pandas as pd\n"
        "\n"
        "def generate_signals(prices, config):\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    if True:\n"
        "        pass\n"
        "    return pd.DataFrame()\n"
    )
    # 1 (base) + 18 (ifs) = 19 > default 15
    monkeypatch.setattr(settings, "max_cyclomatic_complexity", 5)
    mock_dirs_result = _run_with_code(code)
    assert mock_dirs_result is not None
    res = mock_dirs_result
    assert not res.passed
    assert res.error_code == ValidationError.AST_CYCLOMATIC_COMPLEXITY_EXCEEDED
    assert "cyclomatic complexity" in res.detail


# ---------------------------------------------------------------------------
# Undefined-name pre-check tests
# ---------------------------------------------------------------------------


def test_check_undefined_name_catches_undefined():
    """A function referencing an undefined name is rejected."""
    code = """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return _canary_sign
"""
    res = _check_ast(code)
    assert not res.passed
    assert res.error_code == ValidationError.UNDEFINED_NAME
    assert "_canary_sign" in res.detail


def test_check_undefined_name_builtins_pass():
    """Python builtins must NOT be flagged."""
    code = """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    n = len(prices)
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_name_imports_pass():
    """Imported names must NOT be flagged."""
    code = """
import pandas as pd
import numpy as np

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    a = np.array([1, 2, 3])
    return pd.DataFrame(index=prices.index)
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_name_params_pass():
    """Function parameters must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_name_local_vars_pass():
    """Locally assigned variables must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    w = prices * 0.5
    return w
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_name_ret_missing():
    """Undefined name 'ret' in return is caught."""
    code = """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return ret
"""
    res = _check_ast(code)
    assert not res.passed
    assert res.error_code == ValidationError.UNDEFINED_NAME
    assert "ret" in res.detail


def test_check_undefined_nested_function_def_name_pass():
    """Inner function def'd inside outer function must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    def helper():
        return prices
    return helper()
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_nested_function_local_not_leaking():
    """Local var inside inner function must NOT leak into outer scope."""
    code = """
def generate_signals(prices, config):
    def inner():
        x = 1
    return x
"""
    res = _check_ast(code)
    assert not res.passed
    assert res.error_code == ValidationError.UNDEFINED_NAME
    assert "x" in res.detail


def test_check_undefined_nested_function_local_var_not_flagged():
    """Local var inside nested helper must NOT cause false-positive in outer function."""
    code = """
def generate_signals(prices, config):
    def helper():
        local_val = prices * 0.5
        return local_val
    return helper()
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_aug_assign():
    """Augmented assignment (x += 1) defines x."""
    code = """
def generate_signals(prices, config):
    x = 0
    x += 1
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_with_as():
    """with ... as x: defines x (tested via direct _build_function_scope call)."""
    tree = ast.parse("""
def generate_signals(prices, config):
    from contextlib import nullcontext
    with nullcontext() as x:
        pass
    return x
""")
    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_signals":
            func = node
            break
    assert func is not None
    scope = _build_function_scope(func)
    assert "x" in scope


def test_check_undefined_except_as():
    """except ... as e: defines e (tested via direct _build_function_scope call)."""
    tree = ast.parse("""
def generate_signals(prices, config):
    try:
        x = 1
    except Exception as e:
        pass
    return prices
""")
    func = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "generate_signals":
            func = node
            break
    assert func is not None
    scope = _build_function_scope(func)
    assert "e" in scope
    assert "x" in scope


def test_check_undefined_tuple_unpacking_pass():
    """Tuple unpacking (a, b = prices.shape) must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    rows, cols = prices.shape
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_for_enumerate_unpacking_pass():
    """For-loop tuple unpacking (for i, col in enumerate(...)) must NOT be flagged."""
    code = """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    for i, col in enumerate(prices.columns):
        pass
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_for_list_unpacking_pass():
    """List unpacking in for (for a, b in [(1,2)]) must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    pairs = [(1, 2), (3, 4)]
    for x, y in pairs:
        pass
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_top_level_constants_pass():
    """Top-level constants referenced inside functions must NOT be flagged."""
    code = """
DEFAULT_LAG = 21
MAX_WEIGHT = 0.5

def generate_signals(prices, config):
    lag = DEFAULT_LAG
    cap = MAX_WEIGHT
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_list_comprehension_pass():
    """List comprehension variables must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    y = [1, 2, 3]
    return [x for x in y]
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_dict_comprehension_pass():
    """Dict comprehension variables must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    y = {'a': 1, 'b': 2}
    return {k: v for k, v in y.items()}
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_set_comprehension_pass():
    """Set comprehension variables must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    y = [1, 2, 3, 3]
    return {x for x in y}
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_generator_expression_pass():
    """Generator expression variables must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    y = [1, 2, 3]
    return list(x for x in y)
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_nested_comprehension_pass():
    """Nested comprehension and multi-generator comprehension variables must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    matrix = [[1, 2], [3, 4]]
    flat = [y for x in matrix for y in x]
    return flat
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_lambda_arguments_pass():
    """Lambda argument references inside the lambda body must NOT be flagged."""
    code = """
def generate_signals(prices, config):
    f = lambda col: col.mean()
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_check_undefined_comprehension_fail():
    """Referencing an undefined variable inside a comprehension must fail."""
    code = """
def generate_signals(prices, config):
    return [x for x in z]
"""
    res = _check_ast(code)
    assert not res.passed
    assert res.error_code == ValidationError.UNDEFINED_NAME
    assert "z" in res.detail


def test_check_undefined_lambda_fail():
    """Referencing an undefined variable inside a lambda must fail."""
    code = """
def generate_signals(prices, config):
    f = lambda col: col.mean() + z
    return prices
"""
    res = _check_ast(code)
    assert not res.passed
    assert res.error_code == ValidationError.UNDEFINED_NAME
    assert "z" in res.detail


# ---------------------------------------------------------------------------
# End of undefined-name tests
# ---------------------------------------------------------------------------


def _run_with_code(code: str) -> ValidationResult | None:
    """Helper: run preflight on a temporary strategy file."""
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as d:
        strat_dir = Path(d) / "strategies"
        conf_dir = Path(d) / "configs"
        strat_dir.mkdir()
        conf_dir.mkdir()
        (strat_dir / "x.py").write_text(code, encoding="utf-8")
        (conf_dir / "x.yaml").write_text("universe: [SPY]\n", encoding="utf-8")
        return preflight("x", strat_dir, conf_dir)
    return None


def test_value_error_whitelisted_pass():
    """ValueError and other exceptions are whitelisted and should pass undefined name check."""
    code = """
def generate_signals(prices, config):
    if len(prices) < 10:
        raise ValueError("Prices too short")
    return prices
"""
    res = _check_ast(code)
    assert res.passed


def test_validator_calendar_bypass_shift_test(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies calendar-rebalanced strategies bypass Shift Test and pass preflight."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "calendar_strat.py"
    # Write a strategy that rebalances monthly (BME)
    strat_file.write_text(
        """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # rebalance monthly
    start = prices.index.min()
    end = prices.index.max()
    tz = prices.index.tz
    rebalance_dates = pd.date_range(start=start, end=end, freq="BME", tz=tz).intersection(prices.index)
    
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)
    if "SPY" in weights.columns:
        weights["SPY"] = 1.0
        
    weights = weights.reindex(prices.index).ffill().fillna(0.0)
    return weights
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "calendar_strat.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("calendar_strat", strat_dir, conf_dir)
    assert res.passed
    assert res.error_code is None


def test_compare_signals_to_incumbent_identical_and_distinct(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that compare_signals_to_incumbent correctly identifies identical and distinct signal generators."""
    strat_dir, conf_dir = mock_dirs

    incumbent_code = """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in weights.columns:
        weights["SPY"] = 1.0
    return weights
"""

    candidate_identical = """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Slightly different structure but mathematically identical
    w = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    w.loc[:, "SPY"] = 1.0
    return w
"""

    candidate_distinct = """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in weights.columns:
        weights["SPY"] = 0.5
    return weights
"""

    config_yaml = """
universe:
  - SPY
"""

    from autobacktest.strategy.validator import compare_signals_to_incumbent

    # Identical should return True for identical, and diff < 1e-6
    is_identical, diff = compare_signals_to_incumbent(
        "test_strat",
        candidate_identical,
        config_yaml,
        incumbent_code,
        strat_dir,
        conf_dir,
    )
    assert is_identical
    assert diff < 1e-6

    # Distinct should return False for identical, and diff close to 0.5
    is_identical, diff = compare_signals_to_incumbent(
        "test_strat",
        candidate_distinct,
        config_yaml,
        incumbent_code,
        strat_dir,
        conf_dir,
    )
    assert not is_identical
    assert abs(diff - 0.5) < 1e-6


def test_validator_lookahead_sniff_message_contains_ticker_and_magnitude(mock_dirs: tuple[Path, Path]) -> None:
    """Lookahead sniff failure message should include ticker name and weight delta."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "lookahead_detail.py"
    strat_file.write_text(
        """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    # LEAKAGE: weights on day t read prices from the VERY LAST day
    # of the entire DataFrame (future data!)
    last_val = prices.iloc[-1]["SPY"]
    if "SPY" in weights.columns:
        weights["SPY"] = last_val / (last_val + 1.0)
    return weights
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "lookahead_detail.yaml"
    conf_file.write_text("universe: [SPY]\n", encoding="utf-8")

    res = preflight("lookahead_detail", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.LOOKAHEAD_DETECTED
    # Message should name the ticker that leaked
    assert "SPY" in res.detail
    # Message should include a numeric weight delta
    assert "weight delta" in res.detail
