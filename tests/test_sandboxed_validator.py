"""Unit tests verifying sandboxed subprocess isolation and safety in pre-flight validation."""

import os
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


def test_sandboxed_valid_strategy_passes(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that a valid strategy passes all subprocess sandboxed validations."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "simple.py"
    strat_file.write_text(
        """
import pandas as pd
import numpy as np

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Always invest 100% in SPY
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in weights.columns:
        weights["SPY"] = 1.0
    return weights
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "simple.yaml"
    conf_file.write_text(
        """
universe:
  - SPY
benchmark: SPY
""",
        encoding="utf-8",
    )

    res = preflight("simple", strat_dir, conf_dir)
    assert res.passed
    assert res.error_code is None


def test_sandboxed_timeout_abort(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that an infinite loop inside strategy execution is aborted via subprocess timeout."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "slow.py"
    # Execute a pure Python infinite loop (no imports required)
    strat_file.write_text(
        """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Infinite loop
    x = 0
    while True:
        x += 1
    return pd.DataFrame()
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "slow.yaml"
    conf_file.write_text(
        """
universe:
  - SPY
benchmark: SPY
""",
        encoding="utf-8",
    )

    # The validator subprocess has a timeout that aborts it, or the inner sandbox raises timeout
    res = preflight("slow", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.SMOKE_TEST_FAILED
    assert "exceeded limit" in res.detail or "timed out" in res.detail


def test_sandboxed_crash_isolation(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that a division-by-zero or exception inside strategy does not crash the parent process."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "crash.py"
    strat_file.write_text(
        """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Deliberate division by zero
    x = 1 / 0
    return pd.DataFrame()
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "crash.yaml"
    conf_file.write_text(
        """
universe:
  - SPY
benchmark: SPY
""",
        encoding="utf-8",
    )

    res = preflight("crash", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.SMOKE_TEST_FAILED
    assert "division by zero" in res.detail


def test_sandboxed_system_pollution_isolation(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that environment mutations inside the strategy are isolated to the subprocess."""
    strat_dir, conf_dir = mock_dirs

    strat_file = strat_dir / "pollute.py"
    strat_file.write_text(
        """
import pandas as pd
import os

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Try to pollute environment variables
    os.environ["POLLUTED_BY_SANDBOX"] = "YES"
    return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "pollute.yaml"
    conf_file.write_text(
        """
universe:
  - SPY
benchmark: SPY
""",
        encoding="utf-8",
    )

    # Make sure we set whitelist environment to allow os import for testing
    old_whitelist = os.environ.get("AUTOBACKTEST_SAFE_IMPORTS_WHITELIST", "")
    os.environ["AUTOBACKTEST_SAFE_IMPORTS_WHITELIST"] = "pandas,numpy,os"

    try:
        preflight("pollute", strat_dir, conf_dir)
        # Pollution should have occurred in the subprocess, but not in the parent process
        assert "POLLUTED_BY_SANDBOX" not in os.environ
    finally:
        os.environ["AUTOBACKTEST_SAFE_IMPORTS_WHITELIST"] = old_whitelist


def test_ast_blocks_format_string_dunder_exploit(mock_dirs: tuple[Path, Path]) -> None:
    """Verifies that dunder access via .format() is blocked by AST checker.

    Attack vector: "{0.__class__}".format(obj) or format_map / vformat.
    """
    strat_dir, conf_dir = mock_dirs

    # Vector 1: str.format() with dunder in format string
    strat_file = strat_dir / "exploit1.py"
    strat_file.write_text(
        """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    template = "{0.__class__}"
    return prices
""",
        encoding="utf-8",
    )
    conf_file = conf_dir / "exploit1.yaml"
    conf_file.write_text("universe:\n  - SPY\nbenchmark: SPY\n", encoding="utf-8")

    res = preflight("exploit1", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT
    assert "dunder" in res.detail or "format" in res.detail

    # Vector 2: str.format_map() directly called
    strat_file = strat_dir / "exploit2.py"
    strat_file.write_text(
        """
import pandas as pd

def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return prices.format_map({"x": 1})
""",
        encoding="utf-8",
    )
    conf_file = conf_dir / "exploit2.yaml"
    conf_file.write_text("universe:\n  - SPY\nbenchmark: SPY\n", encoding="utf-8")

    res = preflight("exploit2", strat_dir, conf_dir)
    assert not res.passed
    assert res.error_code == ValidationError.AST_BLOCKED_IMPORT
    assert "format" in res.detail
