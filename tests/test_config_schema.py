"""Unit tests for strategy config schema parsing and validation."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from autobacktest.strategy.config_schema import StrategyConfig


def test_valid_minimal_config() -> None:
    """Verifies that minimal fields with defaults parse correctly."""
    data = {
        "universe": ["SPY", "BIL"],
    }
    cfg = StrategyConfig.model_validate(data)
    assert cfg.universe == ["SPY", "BIL"]
    assert cfg.benchmark == "SPY"
    assert cfg.momentum_lookback == 12
    assert cfg.max_drawdown_limit == 0.20
    assert cfg.turnover_limit == 2.0
    assert cfg.params == {}


def test_valid_haa_config() -> None:
    """Verifies parsing a fully configured schema model."""
    data = {
        "universe": ["SPY", "IWM", "TIP"],
        "benchmark": "QQQ",
        "momentum_lookback": 6,
        "max_drawdown_limit": 0.20,
        "turnover_limit": 2.5,
        "params": {
            "offensive_universe": ["SPY", "IWM"],
            "defensive_universe": ["BIL"],
        },
    }
    cfg = StrategyConfig.model_validate(data)
    assert cfg.benchmark == "QQQ"
    assert cfg.momentum_lookback == 6
    assert cfg.max_drawdown_limit == 0.20
    assert cfg.turnover_limit == 2.5
    assert cfg.params["offensive_universe"] == ["SPY", "IWM"]


def test_empty_universe_raises_validation_error() -> None:
    """Verifies that an empty universe is strictly rejected."""
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate({"universe": []})


def test_invalid_benchmark_raises_error() -> None:
    """Verifies that benchmark must be a non-empty string."""
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate({"universe": ["SPY"], "benchmark": ""})


def test_negative_values_raises_error() -> None:
    """Verifies that limit values must be positive and drawdowns capped at 1.0."""
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate({"universe": ["SPY"], "max_drawdown_limit": -0.01})
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate({"universe": ["SPY"], "max_drawdown_limit": 1.05})
    with pytest.raises(ValidationError):
        StrategyConfig.model_validate({"universe": ["SPY"], "turnover_limit": -0.1})


def test_extra_fields_allowed() -> None:
    """Verifies that extra fields at the root are allowed by ConfigDict."""
    cfg = StrategyConfig.model_validate(
        {
            "universe": ["SPY"],
            "some_extra_field": "allowed",
        }
    )
    assert cfg.some_extra_field == "allowed"


def test_params_collision_raises_error() -> None:
    """Verifies that key collisions between params and top-level fields are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        StrategyConfig.model_validate(
            {
                "universe": ["SPY"],
                "params": {
                    "universe": ["QQQ"],
                },
            }
        )
    exc_str = str(exc_info.value)
    assert "Keys in 'params' collide with top-level schema fields" in exc_str


def test_from_yaml_loader(tmp_path: Path) -> None:
    """Verifies loading valid strategy config from YAML files."""
    file_path = tmp_path / "test_config.yaml"
    file_path.write_text(
        """
universe:
  - SPY
  - BIL
benchmark: SPY
momentum_lookback: 10
params:
  filter_ticker: TIP
""",
        encoding="utf-8",
    )
    cfg = StrategyConfig.from_yaml(file_path)
    assert cfg.momentum_lookback == 10
    assert cfg.params["filter_ticker"] == "TIP"


def test_from_yaml_file_not_found() -> None:
    """Verifies that loading a non-existent file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        StrategyConfig.from_yaml(Path("non_existent_file.yaml"))
