"""Unit tests for strategy contract signature and output checkers."""

from types import ModuleType
from typing import Any

import numpy as np
import pandas as pd

from autobacktest.strategy.contract import validate_output, validate_signature


class DummyModule(ModuleType):
    """Mock dynamic module."""


def test_validate_signature_missing_function() -> None:
    """Verifies module fails when generate_signals is missing."""
    mod = DummyModule("test_mod")
    ok, err = validate_signature(mod)
    assert not ok
    assert err is not None
    assert "Module must export a 'generate_signals' function" in err


def test_validate_signature_non_callable() -> None:
    """Verifies module fails when generate_signals is not callable."""
    mod = DummyModule("test_mod")
    mod.generate_signals = "not a function"  # type: ignore
    ok, err = validate_signature(mod)
    assert not ok
    assert err is not None
    assert "must be a callable function" in err


def test_validate_signature_correct() -> None:
    """Verifies that correct function signature passes."""
    mod = DummyModule("test_mod")

    def generate_signals(
        _prices: pd.DataFrame, _config: dict[str, Any]
    ) -> pd.DataFrame:
        return pd.DataFrame()

    mod.generate_signals = generate_signals  # type: ignore
    ok, err = validate_signature(mod)
    assert ok
    assert err is None


def test_validate_signature_insufficient_args() -> None:
    """Verifies signature fails with too few arguments."""
    mod = DummyModule("test_mod")

    def generate_signals(_prices: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    mod.generate_signals = generate_signals  # type: ignore
    ok, err = validate_signature(mod)
    assert not ok
    assert err is not None
    assert "must accept at least 2 parameters" in err


def test_validate_signature_keyword_only() -> None:
    """Verifies signature fails if parameters are strictly keyword-only."""
    mod = DummyModule("test_mod")

    def generate_signals(
        *, _prices: pd.DataFrame, _config: dict[str, Any]
    ) -> pd.DataFrame:
        return pd.DataFrame()

    mod.generate_signals = generate_signals  # type: ignore
    ok, err = validate_signature(mod)
    assert not ok
    assert err is not None
    assert "must be positional" in err


def test_validate_output_valid() -> None:
    """Verifies valid signal weights DataFrame."""
    dates = pd.date_range("2023-01-01", periods=3)
    df = pd.DataFrame(
        {"SPY": [0.4, 1.0, 0.0], "BIL": [0.6, 0.0, 0.5]},
        index=dates,
    )
    ok, err = validate_output(df, ["SPY", "BIL", "TIP"])
    assert ok
    assert err is None


def test_validate_output_not_a_dataframe() -> None:
    """Verifies type check fails on non-DataFrame returns."""
    ok, err = validate_output("not a dataframe", ["SPY"])
    assert not ok
    assert err is not None
    assert "Expected pandas DataFrame" in err


def test_validate_output_empty() -> None:
    """Verifies empty weights are rejected."""
    ok, err = validate_output(pd.DataFrame(), ["SPY"])
    assert not ok
    assert err is not None
    assert "DataFrame is empty" in err


def test_validate_output_has_nans() -> None:
    """Verifies NaNs in weights are strictly rejected."""
    dates = pd.date_range("2023-01-01", periods=2)
    df = pd.DataFrame({"SPY": [0.5, np.nan]}, index=dates)
    ok, err = validate_output(df, ["SPY"])
    assert not ok
    assert err is not None
    assert "must not contain NaN values" in err


def test_validate_output_invalid_ticker() -> None:
    """Verifies assets outside the universe are rejected."""
    dates = pd.date_range("2023-01-01", periods=2)
    df = pd.DataFrame({"AAPL": [1.0, 1.0]}, index=dates)
    ok, err = validate_output(df, ["SPY", "BIL"])
    assert not ok
    assert err is not None
    assert "outside config universe" in err


def test_validate_output_negative_weights() -> None:
    """Verifies that short weights are rejected."""
    dates = pd.date_range("2023-01-01", periods=2)
    df = pd.DataFrame({"SPY": [0.5, -0.01]}, index=dates)
    ok, err = validate_output(df, ["SPY"])
    assert not ok
    assert err is not None
    assert "must be non-negative (long-only)" in err


def test_validate_output_excessive_leverage() -> None:
    """Verifies that total row weights summing to > 1.0 are rejected."""
    dates = pd.date_range("2023-01-01", periods=2)
    df = pd.DataFrame(
        {"SPY": [0.6, 0.8], "BIL": [0.5, 0.3]},
        index=dates,
    )
    ok, err = validate_output(df, ["SPY", "BIL"])
    assert not ok
    assert err is not None
    assert "row sums exceed 1.0" in err
