import numpy as np
import pandas as pd

from autobacktest.evaluator.evaluate import evaluate_strategy_detailed
from autobacktest.evaluator.report import EvaluationReport
from autobacktest.strategy.config_schema import StrategyConfig


def _make_prices() -> pd.DataFrame:
    dates = pd.date_range("2015-01-01", "2026-01-01", freq="B")
    n = len(dates)
    rng = np.random.default_rng(42)
    # Asset returns
    a_ret = rng.normal(0.0005, 0.01, n)
    b_ret = rng.normal(0.0002, 0.01, n)
    prices = pd.DataFrame(
        {
            "ASSET": 100.0 * np.exp(np.cumsum(a_ret)),
            "BENCH": 100.0 * np.exp(np.cumsum(b_ret)),
        },
        index=dates,
    )
    return prices


def generate_signals_mock(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Always allocate 100% to ASSET
    monthly_last = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly_last.index
    universe = config.get("universe", [])
    weights = pd.DataFrame(0.0, index=idx, columns=universe)
    weights["ASSET"] = 1.0
    return weights


def test_characterization_evaluate_detailed() -> None:
    prices = _make_prices()
    bench_returns = prices["BENCH"].pct_change().fillna(0.0)

    config_data = {
        "universe": ["ASSET"],
        "benchmark": "BENCH",
        "momentum_lookback": 12,
        "max_drawdown_limit": 0.5,
        "turnover_limit": 5.0,
        "borrow_cost_bps": 100.0,
        "params": {},
    }
    config = StrategyConfig(**config_data)

    report, returns = evaluate_strategy_detailed(
        strategy_name="toy_strat",
        generate_signals_fn=generate_signals_mock,
        config=config,
        start_date="2015-01-01",
        end_date="2026-01-01",
        _prices=prices,
        _bench_returns=bench_returns,
    )

    assert isinstance(report, EvaluationReport)
    assert report.strategy_name == "toy_strat"
    assert report.in_sample_metrics.sharpe_ratio is not None
    assert isinstance(returns, pd.Series)
    assert not returns.empty
