"""Unit tests for the pre-flight strategy and config validator."""

from pathlib import Path

import pytest

from autobacktest.strategy.validator import ValidationError, preflight


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
    # Write a file exceeding 100KB (e.g. 101KB of comment padding)
    padding = "#" * (101 * 1024)
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
