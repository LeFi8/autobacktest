from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from autobacktest.gate import GateResult
from autobacktest.llm.base import AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import run_optimization
from tests.test_holdout_cap import _make_canned_report
from tests.test_orchestrator_e2e import BASELINE_STRATEGY, STRATEGY_CONFIG, _make_fake_provider, _make_synthetic_prices


@pytest.mark.slow
@pytest.mark.usefixtures("mock_validate_candidate_pass")
def test_resume_optimization_picks_up_correctly(
    project_root_with_lessons: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    # 1. Patch evaluation to always return valid report
    def patched_evaluate(*_args, **_kwargs):
        return _make_canned_report(sharpe=1.2)

    monkeypatch.setattr("autobacktest.orchestrator.evaluate_strategy_detailed", patched_evaluate)
    monkeypatch.setattr("autobacktest.orchestrator.max_config_similarity", lambda *_: 0.0)
    monkeypatch.setattr("autobacktest.orchestrator.check_returns_correlation", lambda *_: (True, 0.0))
    monkeypatch.setattr("autobacktest.orchestrator.select", lambda *_a, **_kw: GateResult(accepted=True))
    monkeypatch.setattr("autobacktest.orchestrator.confirm", lambda *_a, **_kw: GateResult(accepted=True))

    edit = AgentEdit(
        strategy_code=BASELINE_STRATEGY,
        config_yaml=STRATEGY_CONFIG,
        reasoning="Dummy edit.",
        raw_response="{}",
    )
    provider = MockProvider(response=edit)

    # 2. Run first optimization for 2 iterations (saves iteration 0, 1, 2)
    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        result_1 = run_optimization(
            program_path=project_root_with_lessons / "program.md",
            strategy_name="toy",
            iterations=2,
            provider=provider,
            run_dir=project_root_with_lessons / "runs",
            strategies_dir=project_root_with_lessons / "strategies",
            configs_dir=project_root_with_lessons / "configs",
            repo_path=project_root_with_lessons,
            start_date="2013-01-01",
            end_date="2025-01-01",
        )

    assert result_1.n_committed == 2
    saved_run_id = result_1.run_id

    # 3. Resume the optimization for a total of 4 iterations
    # It should pick up starting at k=3 and run iterations 3 and 4!
    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        result_2 = run_optimization(
            program_path=project_root_with_lessons / "program.md",
            strategy_name="toy",
            iterations=4,
            provider=provider,
            run_dir=project_root_with_lessons / "runs",
            strategies_dir=project_root_with_lessons / "strategies",
            configs_dir=project_root_with_lessons / "configs",
            repo_path=project_root_with_lessons,
            start_date="2013-01-01",
            end_date="2025-01-01",
            resume=saved_run_id,
        )

    # In the resumed optimization, it ran 2 iterations (k=3, 4) and successfully committed both!
    assert result_2.n_committed == 2
    assert result_2.run_id == saved_run_id


@pytest.mark.slow
@pytest.mark.usefixtures("mock_validate_candidate_pass")
def test_resume_reconstructs_holdout_net_returns(
    project_root_with_lessons: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    # We want a mock evaluate that returns a report with holdout_net_returns populated!
    idx = pd.date_range("2021-01-01", periods=3)
    expected_holdout_returns = pd.Series([0.05, -0.02, 0.03], index=idx, name="holdout")

    def patched_evaluate(*_args, **_kwargs):
        report, returns = _make_canned_report(sharpe=1.2)
        report.holdout_net_returns = expected_holdout_returns
        return report, returns

    monkeypatch.setattr("autobacktest.orchestrator.evaluate_strategy_detailed", patched_evaluate)
    monkeypatch.setattr("autobacktest.orchestrator.max_config_similarity", lambda *_: 0.0)
    monkeypatch.setattr("autobacktest.orchestrator.check_returns_correlation", lambda *_: (True, 0.0))
    monkeypatch.setattr("autobacktest.orchestrator.select", lambda *_a, **_kw: GateResult(accepted=True))
    monkeypatch.setattr("autobacktest.orchestrator.confirm", lambda *_a, **_kw: GateResult(accepted=True))

    edit = AgentEdit(
        strategy_code=BASELINE_STRATEGY,
        config_yaml=STRATEGY_CONFIG,
        reasoning="Dummy edit.",
        raw_response="{}",
    )
    provider = MockProvider(response=edit)

    # Run first optimization for 1 iteration
    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        result_1 = run_optimization(
            program_path=project_root_with_lessons / "program.md",
            strategy_name="toy",
            iterations=1,
            provider=provider,
            run_dir=project_root_with_lessons / "runs",
            strategies_dir=project_root_with_lessons / "strategies",
            configs_dir=project_root_with_lessons / "configs",
            repo_path=project_root_with_lessons,
            start_date="2013-01-01",
            end_date="2025-01-01",
        )

    saved_run_id = result_1.run_id

    # Resume the optimization run for a total of 2 iterations
    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        result_2 = run_optimization(
            program_path=project_root_with_lessons / "program.md",
            strategy_name="toy",
            iterations=2,
            provider=provider,
            run_dir=project_root_with_lessons / "runs",
            strategies_dir=project_root_with_lessons / "strategies",
            configs_dir=project_root_with_lessons / "configs",
            repo_path=project_root_with_lessons,
            start_date="2013-01-01",
            end_date="2025-01-01",
            resume=saved_run_id,
        )

    # Verification: final incumbent report should carry reconstructed holdout returns
    assert result_2.final_report.holdout_net_returns is not None
    pd.testing.assert_series_equal(
        result_2.final_report.holdout_net_returns,
        expected_holdout_returns,
        check_names=False,
    )
