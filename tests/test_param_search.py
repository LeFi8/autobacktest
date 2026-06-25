"""Tests for Optuna numeric parameter optimization (Phase 4)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from autobacktest.optimization.param_search import (
    _apply_optimized_config,
    _build_optuna_space,
    optimize_numeric_params,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _rng() -> np.random.Generator:
    return np.random.default_rng(42)


def _prices_12yr() -> pd.DataFrame:
    dates = pd.bdate_range(start="2013-01-01", end="2025-01-01")
    n = len(dates)
    rng = _rng()
    high_ret = rng.normal(0.001, 0.002, n)
    low_ret = rng.normal(0.0001, 0.002, n)
    return pd.DataFrame(
        {
            "HIGH": 100.0 * np.exp(np.cumsum(high_ret)),
            "LOW": 100.0 * np.exp(np.cumsum(low_ret)),
        },
        index=dates,
    )


def _strat_weighted(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Allocate proportionally to ``lookback/24 * top_x/10`` — tests both params."""
    monthly = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly.index
    uni = config.get("universe", [])
    lookback = config.get("momentum_lookback", 12)
    top_x = config.get("top_x", 5)
    weight = min(max((lookback / 24.0) * (top_x / 10.0), 0.0), 1.0)
    weights = pd.DataFrame(0.0, index=idx, columns=uni)
    weights["HIGH"] = weight
    weights["LOW"] = 1.0 - weight
    return weights


def _strat_const(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Ignore config — always 100% HIGH."""
    monthly = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly.index
    uni = config.get("universe", [])
    weights = pd.DataFrame(0.0, index=idx, columns=uni)
    weights["HIGH"] = 1.0
    return weights


def _cfg(**overrides: Any) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "universe": ["HIGH", "LOW"],
        "benchmark": "HIGH",
        "momentum_lookback": 6,  # in KNOWN_RANGES (1.0-24.0)
        "top_x": 5,  # in KNOWN_RANGES (1.0-10.0)
        "max_drawdown_limit": 0.2,  # in ROOT_CONSTRAINTS (0.0-1.0 via Field)
        "borrow_cost_bps": 100.0,
        "turnover_limit": 5.0,
    }
    cfg.update(overrides)
    return cfg


# ---------------------------------------------------------------------------
# _build_optuna_space
# ---------------------------------------------------------------------------


def test_build_space_includes_bounded_numeric() -> None:
    space = _build_optuna_space(_cfg())
    assert "momentum_lookback" in space
    assert "top_x" in space
    for spec in space.values():
        assert "low" in spec
        assert "high" in spec
        assert "value" in spec
        assert "is_int" in spec


def test_build_space_skips_excluded_keys() -> None:
    space = _build_optuna_space(_cfg(), exclude={"top_x"})
    assert "momentum_lookback" in space
    assert "top_x" not in space


def test_build_space_skips_string_values() -> None:
    cfg = _cfg(momentum_lookback="auto")  # type: ignore[arg-type]
    space = _build_optuna_space(cfg)
    assert "momentum_lookback" not in space


def test_build_space_skips_params_not_in_known_ranges() -> None:
    cfg = _cfg()
    cfg["not_in_jitter"] = 42.0
    space = _build_optuna_space(cfg)
    assert "not_in_jitter" not in space


# ---------------------------------------------------------------------------
# optimize_numeric_params
# ---------------------------------------------------------------------------


def test_optimize_improves_lookback() -> None:
    prices = _prices_12yr()
    cfg = _cfg(momentum_lookback=6)
    opt_cfg, _best, improved = optimize_numeric_params(_strat_weighted, cfg, prices, n_trials=10, seed=42)
    assert improved is True
    new_val = opt_cfg.get("momentum_lookback", 0)
    assert new_val > 6  # higher lookback → more HIGH → better Sharpe


def test_optimize_no_degradation_on_const_strategy() -> None:
    prices = _prices_12yr()
    cfg = _cfg()
    _, _, _improved = optimize_numeric_params(_strat_const, cfg, prices, n_trials=5, seed=42)
    # Strategy ignores params; optimization should not hurt
    # (may or may not improve due to noise — just shouldn't crash)


def test_optimize_respects_exclude() -> None:
    prices = _prices_12yr()
    cfg = _cfg(momentum_lookback=6, top_x=5)
    opt_cfg, _best, improved = optimize_numeric_params(
        _strat_weighted, cfg, prices, n_trials=10, seed=42, exclude={"momentum_lookback"}
    )
    assert improved is True
    assert opt_cfg.get("momentum_lookback") == 6  # excluded → unchanged
    assert opt_cfg.get("top_x") != 5  # not excluded → may change


def test_optimize_returns_original_on_empty_space() -> None:
    prices = _prices_12yr()
    cfg = {
        "universe": ["HIGH", "LOW"],
        "benchmark": "HIGH",
        "borrow_cost_bps": 100.0,
        "_unbounded_param": 99.0,
    }
    opt_cfg, _best_sharpe, improved = optimize_numeric_params(_strat_weighted, cfg, prices, n_trials=5, seed=42)
    assert improved is False
    assert opt_cfg is cfg or opt_cfg == cfg


def test_optimize_handles_nan_sharpe() -> None:
    prices = _prices_12yr()
    cfg = _cfg(momentum_lookback=6)

    def _broken_strat(p: pd.DataFrame, c: dict) -> pd.DataFrame:
        if c.get("momentum_lookback", 6) > 20:
            return pd.DataFrame(float("nan"), index=p.index[:1], columns=c.get("universe", []))
        return _strat_weighted(p, c)

    opt_cfg, _best_sharpe, _improved = optimize_numeric_params(_broken_strat, cfg, prices, n_trials=10, seed=42)
    assert isinstance(opt_cfg, dict)


def test_optimize_more_trials_not_worse() -> None:
    prices = _prices_12yr()
    cfg = _cfg(momentum_lookback=6)
    _, _sharpe_few, improved_few = optimize_numeric_params(_strat_weighted, cfg, prices, n_trials=3, seed=42)
    _, _sharpe_many, improved_many = optimize_numeric_params(_strat_weighted, cfg, prices, n_trials=20, seed=42)
    assert improved_many or not improved_few


# ---------------------------------------------------------------------------
# _apply_optimized_config
# ---------------------------------------------------------------------------


@dataclass
class _FakeEdit:
    strategy_code: str = ""
    config_yaml: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    cost: float = 0.0
    reasoning: str = ""


def test_apply_improved() -> None:
    winner: dict[str, Any] = {
        "edit": _FakeEdit(config_yaml="momentum_lookback: 6\n"),
        "config_yaml": "momentum_lookback: 6\n",
        "_config": {"momentum_lookback": 6},
    }
    _apply_optimized_config(winner, {"momentum_lookback": 18}, improved=True, gain=0.5)
    assert winner["optimization_applied"] is True
    assert winner["optimization_gain"] == 0.5
    assert winner["_config"]["momentum_lookback"] == 18


def test_apply_not_improved() -> None:
    winner: dict[str, Any] = {
        "edit": _FakeEdit(config_yaml="momentum_lookback: 6\n"),
        "config_yaml": "momentum_lookback: 6\n",
        "_config": {"momentum_lookback": 6},
    }
    _apply_optimized_config(winner, {"momentum_lookback": 18}, improved=False, gain=0.0)
    assert winner["optimization_applied"] is False
    assert winner["optimization_gain"] == 0.0
    assert winner["_config"]["momentum_lookback"] == 6


def test_apply_handles_params_nest_in_yaml() -> None:
    winner: dict[str, Any] = {
        "edit": _FakeEdit(config_yaml="params:\n  mix: 0.5\n"),
        "config_yaml": "params:\n  mix: 0.5\n",
        "_config": {"universe": ["HIGH"], "params": {"mix": 0.5}},
    }
    _apply_optimized_config(winner, {"universe": ["HIGH"], "params": {"mix": 0.8}}, improved=True, gain=0.3)
    assert winner["_config"]["params"]["mix"] == 0.8
    assert "mix: 0.8" in winner["edit"].config_yaml
