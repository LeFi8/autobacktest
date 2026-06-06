"""Tests for graceful shutdown behavior of the optimization loop on KeyboardInterrupt."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import git
import numpy as np
import pandas as pd
import pytest

from autobacktest.gate import TargetMetric
from autobacktest.llm.base import AgentContext, AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import OrchestratorResult, run_optimization


def _make_synthetic_prices() -> pd.DataFrame:
    dates = pd.bdate_range(start="2013-01-01", end="2025-01-01")
    n = len(dates)
    rng = np.random.default_rng(42)
    high_returns = rng.normal(0.001, 0.002, n)
    low_returns = rng.normal(0.0001, 0.002, n)
    prices = pd.DataFrame(
        {
            "HIGH": 100.0 * np.exp(np.cumsum(high_returns)),
            "LOW": 100.0 * np.exp(np.cumsum(low_returns)),
        },
        index=dates,
    )
    return prices


BASELINE_STRATEGY = """\
import pandas as pd
from typing import Any

def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    monthly_last = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly_last.index
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    weights["LOW"] = 1.0
    return weights
"""

IMPROVED_STRATEGY = """\
import pandas as pd
from typing import Any

def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    monthly_last = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly_last.index
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    weights["HIGH"] = 1.0
    return weights
"""

STRATEGY_CONFIG = """\
universe:
  - HIGH
  - LOW
benchmark: HIGH
momentum_lookback: 12
max_drawdown_limit: 0.50
turnover_limit: 5.0
"""

IMPROVED_CONFIG = """\
universe:
  - HIGH
  - LOW
benchmark: HIGH
momentum_lookback: 1
max_drawdown_limit: 0.30
turnover_limit: 10.0
"""

PROGRAM_MD = """\
# Objective
Maximize returns.

# Constraints
Max drawdown 50%.
"""


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    strat_dir = tmp_path / "strategies"
    cfg_dir = tmp_path / "configs"
    run_dir = tmp_path / "runs"
    strat_dir.mkdir()
    cfg_dir.mkdir()
    run_dir.mkdir()

    (strat_dir / "toy.py").write_text(BASELINE_STRATEGY, encoding="utf-8")
    (cfg_dir / "toy.yaml").write_text(STRATEGY_CONFIG, encoding="utf-8")
    (tmp_path / "program.md").write_text(PROGRAM_MD, encoding="utf-8")

    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    repo.index.add(["strategies/toy.py", "configs/toy.yaml"])
    repo.index.commit("initial: baseline toy strategy")

    return tmp_path


def _make_fake_provider(synthetic_prices: pd.DataFrame) -> object:
    class FakeProvider:
        def get_prices(
            self,
            tickers: list[str],
            _start: str,
            _end: str,
            _interval: str = "1d",
        ) -> pd.DataFrame:
            available = [t for t in tickers if t in synthetic_prices.columns]
            if not available:
                first_col = synthetic_prices.columns[0]
                return synthetic_prices[[first_col]].rename(columns={first_col: tickers[0]})
            return synthetic_prices[available]

    return FakeProvider()


def test_graceful_shutdown_during_optimization(project_root: Path) -> None:
    """If KeyboardInterrupt is raised in the loop after baseline completes, it returns OrchestratorResult."""
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    class InterruptingProvider(MockProvider):
        def generate_edit(self, _context: AgentContext) -> AgentEdit:
            # Raise KeyboardInterrupt when the loop starts generating candidates
            raise KeyboardInterrupt()

    mock_provider = InterruptingProvider()

    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        result = run_optimization(
            program_path=project_root / "program.md",
            strategy_name="toy",
            iterations=3,
            provider=mock_provider,
            run_dir=project_root / "runs",
            strategies_dir=project_root / "strategies",
            configs_dir=project_root / "configs",
            target_metric=TargetMetric.SHARPE,
            repo_path=project_root,
            start_date="2013-01-01",
            end_date="2025-01-01",
            quiet=True,
        )

    assert isinstance(result, OrchestratorResult)
    # The run finished baseline, so baseline_report is populated
    assert result.baseline_report is not None
    assert result.final_report is not None
    # No candidate was accepted because it interrupted immediately
    assert result.n_committed == 0


def test_propagate_interrupt_before_baseline(project_root: Path) -> None:
    """If KeyboardInterrupt is raised during baseline evaluation, it propagates the exception."""
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    mock_provider = MockProvider()

    # Raise KeyboardInterrupt during baseline evaluation
    with (
        patch(
            "autobacktest.evaluator.evaluate.CachedDataProvider",
            return_value=fake_instance,
        ),
        patch(
            "autobacktest.orchestrator.evaluate_strategy_detailed",
            side_effect=KeyboardInterrupt(),
        ),
        pytest.raises(KeyboardInterrupt),
    ):
        run_optimization(
            program_path=project_root / "program.md",
            strategy_name="toy",
            iterations=3,
            provider=mock_provider,
            run_dir=project_root / "runs",
            strategies_dir=project_root / "strategies",
            configs_dir=project_root / "configs",
            target_metric=TargetMetric.SHARPE,
            repo_path=project_root,
            start_date="2013-01-01",
            end_date="2025-01-01",
            quiet=True,
        )
