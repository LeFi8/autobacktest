from unittest.mock import MagicMock

import pandas as pd

from autobacktest.evaluator.evaluate import evaluate_strategy_detailed


def test_cache_hit_on_whitespace_and_comment_changes():
    # 1. Create dummy data
    dates = pd.date_range("2010-01-01", "2025-01-01", freq="D")
    prices = pd.DataFrame({"SPY": [10.0] * len(dates)}, index=dates)
    bench_returns = pd.Series([0.0] * len(dates), index=dates)

    # 2. Setup mock strategy generate_signals_fn
    mock_signals_fn = MagicMock(return_value=pd.DataFrame({"SPY": [1.0] * len(dates)}, index=dates))

    config = {
        "universe": ["SPY"],
        "benchmark": "SPY",
    }
    _eval_cache = {}

    code_v1 = """
# Original strategy code
def generate_signals(prices, config):
    return prices
"""

    code_v2 = """
def generate_signals(prices, config):
    # Added some comments here
    return prices  # Inline comment
"""

    # First run (should populate cache)
    report_1, _returns_1 = evaluate_strategy_detailed(
        "dummy_strategy",
        mock_signals_fn,
        config,
        _prices=prices,
        _bench_returns=bench_returns,
        _eval_cache=_eval_cache,
        _strategy_code=code_v1,
    )
    call_count_after_first = mock_signals_fn.call_count

    # Second run with different comments/whitespaces (should hit cache)
    report_2, _returns_2 = evaluate_strategy_detailed(
        "dummy_strategy",
        mock_signals_fn,
        config,
        _prices=prices,
        _bench_returns=bench_returns,
        _eval_cache=_eval_cache,
        _strategy_code=code_v2,
    )
    # Call count should NOT have increased (cache hit!)
    assert mock_signals_fn.call_count == call_count_after_first
    assert report_1.observed_sharpe == report_2.observed_sharpe
