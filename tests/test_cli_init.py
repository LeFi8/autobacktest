"""Tests for the CLI init-strategy subcommand."""

from __future__ import annotations

from collections.abc import Generator
from pathlib import Path
from unittest.mock import patch

import yaml
from pytest import fixture
from typer.testing import CliRunner

from autobacktest.cli import app

runner = CliRunner()


@fixture(autouse=True)
def _patch_settings(tmp_path: Path) -> Generator[None, None, None]:
    """Redirect strategy/config dirs to temp paths for every test."""
    with (
        patch("autobacktest.cli.settings.strategies_dir", tmp_path / "strategies"),
        patch("autobacktest.cli.settings.configs_dir", tmp_path / "configs"),
    ):
        yield


def test_init_strategy_basic_flow(tmp_path: Path) -> None:
    """Full interactive flow produces valid config and strategy files."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "configs").mkdir()

    inputs = (
        "\n".join(
            [
                "SPY, QQQ, BIL",  # universe
                "SPY",  # benchmark
                "0.15",  # max drawdown
                "1.5",  # turnover
                "12",  # momentum lookback
                "n",  # advanced params? no
                "n",  # custom params? no
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        ["init-strategy", "--name", "test_momentum"],
        input=inputs,
    )

    assert result.exit_code == 0, f"Exit: {result.exit_code}, out: {result.output}"
    assert "Success" in result.output

    config_path = tmp_path / "configs" / "test_momentum.yaml"
    assert config_path.exists()
    with config_path.open() as f:
        cfg = yaml.safe_load(f)
    assert cfg["universe"] == ["SPY", "QQQ", "BIL"]
    assert cfg["benchmark"] == "SPY"
    assert cfg["max_drawdown_limit"] == 0.15
    assert cfg["turnover_limit"] == 1.5
    assert cfg["momentum_lookback"] == 12
    assert cfg["params"] == {}

    strategy_path = tmp_path / "strategies" / "test_momentum.py"
    assert strategy_path.exists()
    content = strategy_path.read_text()
    assert "def generate_signals" in content
    assert "prices: pd.DataFrame" in content

    program_path = tmp_path / "program-test_momentum.md"
    assert program_path.exists()
    program_content = program_path.read_text()
    assert "# Objective" in program_content
    assert "# Constraints" in program_content
    assert "SPY, QQQ, BIL" in program_content
    assert "15%" in program_content  # 0.15 max drawdown
    assert "1.5" in program_content  # turnover limit


def test_init_strategy_invalid_name() -> None:
    """Invalid snake_case name is rejected."""
    result = runner.invoke(
        app,
        ["init-strategy", "--name", "Bad Name!!"],
    )
    assert result.exit_code == 1
    assert "snake_case" in result.output.lower()


def test_init_strategy_overwrite_prompt_cancelled(tmp_path: Path) -> None:
    """When files exist and user declines overwrite, operation cancels."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "strategies" / "existing_strat.py").touch()
    (tmp_path / "configs" / "existing_strat.yaml").touch()
    (tmp_path / "program-existing_strat.md").touch()

    result = runner.invoke(
        app,
        ["init-strategy", "--name", "existing_strat"],
        input="n\n",
    )
    assert result.exit_code == 0
    assert "Operation cancelled" in result.output


def test_init_strategy_overwrite_flag(tmp_path: Path) -> None:
    """--overwrite flag skips the confirmation prompt."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "strategies" / "overwrite_me.py").write_text("old code", encoding="utf-8")
    (tmp_path / "configs" / "overwrite_me.yaml").write_text("old: config", encoding="utf-8")
    (tmp_path / "program-overwrite_me.md").write_text("old: program", encoding="utf-8")

    inputs = (
        "\n".join(
            [
                "SPY, QQQ",  # universe
                "SPY",  # benchmark
                "0.20",  # drawdown
                "2.0",  # turnover
                "12",  # lookback
                "n",  # advanced params? no
                "n",  # custom params? no
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        ["init-strategy", "--name", "overwrite_me", "--overwrite"],
        input=inputs,
    )
    assert result.exit_code == 0
    assert "Success" in result.output

    cfg = (tmp_path / "configs" / "overwrite_me.yaml").read_text(encoding="utf-8")
    assert "old: config" not in cfg

    prog = (tmp_path / "program-overwrite_me.md").read_text(encoding="utf-8")
    assert "old: program" not in prog


def test_init_strategy_invalid_drawdown(tmp_path: Path) -> None:
    """Drawdown outside 0.0-1.0 range is rejected; user can retry."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "configs").mkdir()

    inputs = (
        "\n".join(
            [
                "SPY",  # universe
                "SPY",  # benchmark
                "1.5",  # invalid drawdown → retry
                "0.20",  # valid drawdown
                "2.0",  # turnover
                "12",  # lookback
                "n",  # advanced params? no
                "n",  # custom params? no
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        ["init-strategy", "--name", "dd_test"],
        input=inputs,
    )
    assert result.exit_code == 0
    assert "Success" in result.output
    assert "0.0 and 1.0" in result.output


def test_init_strategy_custom_params(tmp_path: Path) -> None:
    """Custom parameters flow into the params dict with type inference."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "configs").mkdir()

    inputs = (
        "\n".join(
            [
                "SPY, QQQ",  # universe
                "SPY",  # benchmark
                "0.20",  # drawdown
                "2.0",  # turnover
                "12",  # lookback
                "n",  # advanced params? no
                "y",  # custom params? yes
                "vol_span",  # param name
                "126",  # param value (int)
                "target_vol",  # param name
                "0.12",  # param value (float)
                "vol_targeting",  # param name
                "true",  # param value (bool)
                "",  # finish
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        ["init-strategy", "--name", "custom_params_test"],
        input=inputs,
    )
    assert result.exit_code == 0
    assert "Success" in result.output

    config_path = tmp_path / "configs" / "custom_params_test.yaml"
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    assert cfg["params"]["vol_span"] == 126
    assert cfg["params"]["target_vol"] == 0.12
    assert cfg["params"]["vol_targeting"] is True


def test_init_strategy_reserved_key_rejected(tmp_path: Path) -> None:
    """Custom param with a reserved schema field name is rejected."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "configs").mkdir()

    inputs = (
        "\n".join(
            [
                "SPY",  # universe
                "SPY",  # benchmark
                "0.20",  # drawdown
                "2.0",  # turnover
                "12",  # lookback
                "n",  # advanced params? no
                "y",  # custom params? yes
                "universe",  # reserved key — should be rejected, then reprompt
                "vol_span",  # valid custom param
                "21",  # value
                "",  # finish
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        ["init-strategy", "--name", "reserved_key_test"],
        input=inputs,
    )
    assert result.exit_code == 0
    assert "reserved schema field" in result.output
    assert "Success" in result.output

    config_path = tmp_path / "configs" / "reserved_key_test.yaml"
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    assert "universe" not in cfg["params"]
    assert cfg["params"]["vol_span"] == 21


def test_init_strategy_tz_aware_prices(tmp_path: Path) -> None:
    """Generated boilerplate handles tz-aware price indices."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "configs").mkdir()

    inputs = (
        "\n".join(
            [
                "SPY, QQQ, BIL",
                "SPY",
                "0.20",
                "2.0",
                "12",
                "n",  # advanced params? no
                "n",  # custom params? no
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        ["init-strategy", "--name", "tz_aware_test"],
        input=inputs,
    )
    assert result.exit_code == 0
    assert "Success" in result.output

    strategy_path = tmp_path / "strategies" / "tz_aware_test.py"
    assert strategy_path.exists()

    import importlib.util

    import numpy as np
    import pandas as pd

    spec = importlib.util.spec_from_file_location("tz_aware_test", strategy_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    dates = pd.date_range("2023-01-01", periods=500, freq="B", tz="US/Eastern")
    rng = np.random.default_rng(42)
    prices = pd.DataFrame(
        {t: 100.0 * np.exp(np.cumsum(rng.normal(0.0002, 0.01, 500))) for t in ["SPY", "QQQ", "BIL"]},
        index=dates,
    )

    config = {"universe": ["SPY", "QQQ", "BIL"], "params": {"cash_asset": "BIL"}}
    weights = module.generate_signals(prices, config)
    assert not weights.empty
    assert weights.shape[1] == len(["SPY", "QQQ", "BIL"])


def test_init_strategy_advanced_params(tmp_path: Path) -> None:
    """Advanced config parameters are prompted and written correctly."""
    (tmp_path / "strategies").mkdir()
    (tmp_path / "configs").mkdir()

    inputs = (
        "\n".join(
            [
                "SPY, QQQ",  # universe
                "SPY",  # benchmark
                "0.20",  # drawdown
                "2.0",  # turnover
                "12",  # lookback
                "y",  # advanced params? yes
                "50.0",  # borrow_cost_bps
                "8",  # cscv_blocks
                "",  # pbo_limit (no limit)
                "n",  # adaptive_slippage? no
                "0.01",  # min_improvement
                "0.6",  # select_min_return_ratio
                "y",  # require_dsr_non_degradation? yes
                "circular",  # mc_bootstrap_method
                "n",  # custom params? no
            ]
        )
        + "\n"
    )

    result = runner.invoke(
        app,
        ["init-strategy", "--name", "advanced_params_test"],
        input=inputs,
    )
    assert result.exit_code == 0, f"Exit: {result.exit_code}, out: {result.output}"
    assert "Success" in result.output

    config_path = tmp_path / "configs" / "advanced_params_test.yaml"
    assert config_path.exists()
    with config_path.open() as f:
        cfg = yaml.safe_load(f)

    assert cfg["borrow_cost_bps"] == 50.0
    assert cfg["cscv_blocks"] == 8
    assert "pbo_limit" not in cfg or cfg["pbo_limit"] is None
    assert cfg["adaptive_slippage"] is False
    assert cfg["min_improvement"] == 0.01
    assert cfg["select_min_return_ratio"] == 0.6
    assert cfg["require_dsr_non_degradation"] is True
    assert cfg["mc_bootstrap_method"] == "circular"

    program_path = tmp_path / "program-advanced_params_test.md"
    assert program_path.exists()
