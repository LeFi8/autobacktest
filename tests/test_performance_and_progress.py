"""Tests for database index creation and orchestrator progress bar integration."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

from autobacktest.ledger.store import LedgerStore
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import run_optimization


def test_sqlite_index_creation(tmp_path: Path) -> None:
    """Verifies that indexes are successfully created on the attempts table."""
    db_file = tmp_path / "ledger.db"
    store = LedgerStore(db_file)
    store.close()

    # Query sqlite_master for index details
    conn = sqlite3.connect(str(db_file))
    rows = conn.execute(
        """
        SELECT name, tbl_name
        FROM sqlite_master
        WHERE type = 'index' AND tbl_name = 'attempts'
        """
    ).fetchall()
    conn.close()

    index_names = {row[0] for row in rows}
    assert "idx_attempts_run_id" in index_names
    assert "idx_attempts_strategy_name" in index_names
    assert "idx_attempts_dataset_hash" in index_names


@patch("autobacktest.orchestrator.Progress")
def test_orchestrator_progress_bar_interaction(
    mock_progress_cls: MagicMock,
    tmp_path: Path,
) -> None:
    """Verifies that the orchestrator uses rich.progress.Progress to track iterations."""
    # 1. Setup mock progress instances
    mock_progress_instance = MagicMock()
    mock_progress_cls.return_value.__enter__.return_value = mock_progress_instance

    mock_task_id = 999
    mock_progress_instance.add_task.return_value = mock_task_id

    # 2. Setup minimum mocks to run orchestrator with mock provider for 2 iterations

    # Create dummy strategy and config in tmp_path
    strategies_dir = tmp_path / "strategies"
    configs_dir = tmp_path / "configs"
    strategies_dir.mkdir()
    configs_dir.mkdir()

    strat_file = strategies_dir / "simple.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
""",
        encoding="utf-8",
    )

    conf_file = configs_dir / "simple.yaml"
    conf_file.write_text(
        """
universe:
  - SPY
benchmark: SPY
""",
        encoding="utf-8",
    )

    program_file = tmp_path / "program.md"
    program_file.write_text(
        """# Objective
Test progress bar.

# Constraints
None.
""",
        encoding="utf-8",
    )

    # Mock evaluate_strategy_detailed, preflight, and GitLedger to avoid heavy processes and Git commands
    with (
        patch("autobacktest.orchestrator.evaluate_strategy_detailed") as mock_evaluate,
        patch("autobacktest.orchestrator.preflight") as mock_preflight,
        patch("autobacktest.orchestrator.GitLedger") as mock_git_ledger_cls,
    ):
        mock_git_ledger = MagicMock()
        mock_git_ledger.repo_root = tmp_path
        mock_git_ledger.create_run_branch.return_value = "mock-branch"
        mock_git_ledger._repo.head.commit.hexsha = "mock-sha"
        mock_git_ledger.commit_strategy.return_value = "mock-sha"
        mock_git_ledger_cls.return_value = mock_git_ledger

        import pandas as pd

        dummy_returns = pd.Series([0.001] * 5, index=pd.date_range("2025-01-01", periods=5))

        mock_report = MagicMock()
        mock_report.dataset_hash = "abc"
        mock_report.observed_sharpe = 1.5
        mock_report.deflated_sharpe = 1.4
        mock_report.effective_trials = 1
        mock_report.holdout_metrics.sharpe_ratio = 1.5
        mock_report.holdout_metrics.max_drawdown = 0.05
        mock_report.holdout_metrics.turnover = 0.2
        mock_report.in_sample_metrics.sharpe_ratio = 1.5
        mock_report.in_sample_metrics.max_drawdown = 0.05
        mock_report.in_sample_metrics.turnover = 0.2
        mock_report.regime_passed = True
        mock_report.holdout_net_returns = dummy_returns
        mock_report.to_json.return_value = "{}"

        mock_evaluate.return_value = (mock_report, dummy_returns)

        mock_preflight_res = MagicMock()
        mock_preflight_res.passed = True
        mock_preflight.return_value = mock_preflight_res

        # 3. Call run_optimization for 2 iterations
        provider = MockProvider()
        run_optimization(
            program_path=program_file,
            strategy_name="simple",
            iterations=2,
            provider=provider,
            run_dir=tmp_path / "runs",
            strategies_dir=strategies_dir,
            configs_dir=configs_dir,
            repo_path=tmp_path,
        )

    # 4. Assert Progress bar was initialized and advanced 2 times plus once for baseline or failures
    assert mock_progress_cls.called
    assert mock_progress_instance.add_task.called
    assert mock_progress_instance.add_task.call_args[1]["total"] == 2

    # Verify advance calls happened inside the loops
    assert mock_progress_instance.update.call_count == 2
    for call in mock_progress_instance.update.call_args_list:
        assert call[0][0] == mock_task_id
        assert call[1]["advance"] == 1
        assert "Incumbent Sharpe" in call[1]["description"]
