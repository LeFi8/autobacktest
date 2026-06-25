"""Tests for the cheap in-sample pre-screen (Phase 3)."""

from unittest.mock import patch

import numpy as np
import pandas as pd

from autobacktest.config import settings
from autobacktest.evaluator.evaluate import evaluate_strategy_detailed
from autobacktest.evaluator.report import EvaluationReport


def _make_prices() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2026-01-01", freq="B")
    n = len(dates)
    rng = np.random.default_rng(42)
    a_ret = rng.normal(0.002, 0.01, n)  # strong positive drift
    b_ret = rng.normal(0.0002, 0.01, n)
    t_ret = rng.normal(-0.002, 0.015, n)  # strong negative drift
    return pd.DataFrame(
        {
            "ASSET": 100.0 * np.exp(np.cumsum(a_ret)),
            "BENCH": 100.0 * np.exp(np.cumsum(b_ret)),
            "TOXIC": 100.0 * np.exp(np.cumsum(t_ret)),
        },
        index=dates,
    )


def _bad_strategy(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Long-only strategy that goes all-in on the toxic asset."""
    monthly = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly.index
    universe = config.get("universe", [])
    weights = pd.DataFrame(0.0, index=idx, columns=universe)
    weights["TOXIC"] = 1.0
    return weights


def _good_strategy(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Long-only strategy that goes all-in on the good asset."""
    monthly = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly.index
    universe = config.get("universe", [])
    weights = pd.DataFrame(0.0, index=idx, columns=universe)
    weights["ASSET"] = 1.0
    return weights


def _config() -> dict:
    return {
        "universe": ["ASSET"],
        "benchmark": "BENCH",
        "borrow_cost_bps": 100.0,
        "max_drawdown_limit": 0.5,
        "turnover_limit": 5.0,
        "params": {},
    }


def _bad_config() -> dict:
    return {
        "universe": ["TOXIC"],
        "benchmark": "BENCH",
        "borrow_cost_bps": 100.0,
        "max_drawdown_limit": 0.5,
        "turnover_limit": 5.0,
        "params": {},
    }


def test_prescreen_rejects_bad_candidate() -> None:
    """Bad in-sample candidate -> battery skipped, report flagged, gate rejects."""
    prices = _make_prices()
    bench = prices["BENCH"].pct_change().fillna(0.0)

    with (
        patch.object(settings, "enable_cheap_prescreen", True),
        patch.object(settings, "prescreen_sharpe_floor", 0.0),
        patch.object(settings, "prescreen_return_floor", 0.0),
    ):
        report, _ = evaluate_strategy_detailed(
            "bad_strat",
            _bad_strategy,
            _bad_config(),
            start_date="2015-01-01",
            end_date="2026-01-01",
            _prices=prices,
            _bench_returns=bench,
        )

    assert isinstance(report, EvaluationReport)
    assert report.prescreen_rejected is True
    assert report.is_accepted is False
    assert report.regime_passed is False
    assert report.walk_forward_metrics == []
    assert report.holdout_metrics.sharpe_ratio == 0.0
    assert "Pre-screen rejected" in (report.rejection_reason or "")


def test_good_candidate_runs_full_battery() -> None:
    """Good in-sample candidate -> full battery runs, prescreen_rejected=False."""
    prices = _make_prices()
    bench = prices["BENCH"].pct_change().fillna(0.0)

    with (
        patch.object(settings, "enable_cheap_prescreen", True),
        patch.object(settings, "prescreen_sharpe_floor", 0.0),
        patch.object(settings, "prescreen_return_floor", 0.0),
    ):
        report, _ = evaluate_strategy_detailed(
            "good_strat",
            _good_strategy,
            _config(),
            start_date="2015-01-01",
            end_date="2026-01-01",
            _prices=prices,
            _bench_returns=bench,
        )

    assert report.prescreen_rejected is False
    assert len(report.walk_forward_metrics) > 0
    assert report.holdout_metrics.sharpe_ratio != 0.0 or report.regime_passed is True


def test_prescreen_off_identical_to_baseline() -> None:
    """When enable_cheap_prescreen=False, behavior is identical to today."""
    prices = _make_prices()
    bench = prices["BENCH"].pct_change().fillna(0.0)

    with patch.object(settings, "enable_cheap_prescreen", False):
        report, _ = evaluate_strategy_detailed(
            "good_strat",
            _good_strategy,
            _config(),
            start_date="2015-01-01",
            end_date="2026-01-01",
            _prices=prices,
            _bench_returns=bench,
        )

    assert report.prescreen_rejected is False
    assert len(report.walk_forward_metrics) > 0


def test_prescreen_default_off() -> None:
    """Default config has prescreen disabled -- no short-circuit."""
    prices = _make_prices()
    bench = prices["BENCH"].pct_change().fillna(0.0)

    report, _ = evaluate_strategy_detailed(
        "bad_strat",
        _bad_strategy,
        _bad_config(),
        start_date="2015-01-01",
        end_date="2026-01-01",
        _prices=prices,
        _bench_returns=bench,
    )

    # Even with a bad strategy, prescreen is off -> full battery runs
    assert report.prescreen_rejected is False
    assert len(report.walk_forward_metrics) > 0
