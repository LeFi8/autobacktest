from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from autobacktest.strategy.config_schema import StrategyConfig


def test_strategy_config_due_diligence_fields():
    # Verify defaults
    config = StrategyConfig(universe=["SPY"])
    assert config.cscv_blocks == 10
    assert config.regime_benchmark is None

    # Verify customized values
    config_custom = StrategyConfig(universe=["SPY"], cscv_blocks=6, regime_benchmark="TLT")
    assert config_custom.cscv_blocks == 6
    assert config_custom.regime_benchmark == "TLT"

    # Verify validation (cscv_blocks must be >= 4)
    with pytest.raises(ValueError):
        StrategyConfig(universe=["SPY"], cscv_blocks=3)


def test_cscv_pbo_length_safety():
    from autobacktest.evaluator.report import EvaluationReport, WindowReport
    from autobacktest.orchestrator import _deflate

    # 1. Setup mock report and ledger
    report = EvaluationReport(
        strategy_name="test_strat",
        dataset_hash="hash_1",
        gates_passed={},
        is_accepted=False,
        rejection_reason=None,
        holdout_metrics=WindowReport("2023-01-01", "2023-01-10", 0.1, 0.1, 1.0, 1.0, 0.05, 0.1),
        in_sample_metrics=WindowReport("2023-01-01", "2023-01-10", 0.1, 0.1, 1.0, 1.0, 0.05, 0.1),
        walk_forward_metrics=[],
        regime_drawdowns={},
        regime_passed=True,
        mc_sharpe_5th=1.0,
        mc_sharpe_50th=1.2,
        mc_sharpe_95th=1.5,
        observed_sharpe=1.2,
        effective_trials=1,
        deflated_sharpe=0.9,
    )

    # Extremely short returns (5 days)
    short_returns = pd.Series([0.001] * 5, index=pd.date_range("2023-01-01", periods=5))

    ledger = MagicMock()
    # Mock historical returns matrix returns empty dataframe (meaning only candidate is used)
    ledger.fetch_historical_returns.return_value = (pd.DataFrame(), [])

    # Run _deflate with cscv_blocks=10. Since 5 < 20 (2 * cscv_blocks), it should skip and set PBO to None
    with patch("autobacktest.evaluator.cscv.calculate_pbo") as mock_calc_pbo:
        _deflate(report, short_returns, ledger, cscv_blocks=10)
        assert report.pbo is None
        mock_calc_pbo.assert_not_called()
