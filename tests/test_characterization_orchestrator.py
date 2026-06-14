import json
import sqlite3
from pathlib import Path
from unittest.mock import patch

import git
import numpy as np
import pandas as pd
import pytest

from autobacktest.gate import TargetMetric
from autobacktest.llm.base import AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import OrchestratorResult, run_optimization

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
params:
  mock_param: 10.0
"""

IMPROVED_CONFIG = """\
universe:
  - HIGH
  - LOW
benchmark: HIGH
momentum_lookback: 1
max_drawdown_limit: 0.30
turnover_limit: 10.0
params:
  mock_param: 10.0
"""

PROGRAM_MD = """# Objective
Maximize returns.

# Constraints
Drawdown limit 50%.
"""


def _make_prices() -> pd.DataFrame:
    dates = pd.bdate_range("2015-01-01", "2026-01-01")
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


class FakeProvider:
    def __init__(self, prices: pd.DataFrame):
        self.prices = prices

    def get_prices(self, tickers: list[str], _start: str, _end: str, _interval: str = "1d") -> pd.DataFrame:
        available = [t for t in tickers if t in self.prices.columns]
        if not available:
            return self.prices[[self.prices.columns[0]]].rename(columns={self.prices.columns[0]: tickers[0]})
        return self.prices[available]


@pytest.mark.slow
def test_characterization_orchestrator_run(tmp_path: Path, mock_validate_candidate_pass: None) -> None:  # noqa: ARG001
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
    repo.config_writer().set_value("user", "name", "Test").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    repo.index.add(["strategies/toy.py", "configs/toy.yaml"])
    repo.index.commit("initial")

    prices = _make_prices()
    fake_prov = FakeProvider(prices)

    edit = AgentEdit(
        strategy_code=IMPROVED_STRATEGY,
        config_yaml=IMPROVED_CONFIG,
        reasoning="Improvement reasoning",
        raw_response="{}",
    )
    mock_prov = MockProvider(response=edit)

    with patch("autobacktest.evaluator.evaluate.CachedDataProvider", return_value=fake_prov):
        result = run_optimization(
            program_path=tmp_path / "program.md",
            strategy_name="toy",
            iterations=1,
            provider=mock_prov,
            run_dir=run_dir,
            strategies_dir=strat_dir,
            configs_dir=cfg_dir,
            target_metric=TargetMetric.SHARPE,
            repo_path=tmp_path,
            start_date="2015-01-01",
            end_date="2026-01-01",
        )

    assert isinstance(result, OrchestratorResult)
    assert result.n_committed >= 1

    # Check ledger database entries
    ledger_path = run_dir / "ledger.db"
    assert ledger_path.exists()
    conn = sqlite3.connect(str(ledger_path))
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM runs WHERE run_id = ?", (result.run_id,))
    assert cursor.fetchone()[0] == 1

    cursor.execute("SELECT iteration, accepted, committed FROM attempts WHERE run_id = ?", (result.run_id,))
    attempts = cursor.fetchall()
    assert len(attempts) >= 2  # Baseline (0) + iteration 1
    conn.close()

    # Check event log format
    event_log_path = run_dir / result.run_id / "events.jsonl"
    assert event_log_path.exists()
    with event_log_path.open("r") as f:
        events = [json.loads(line) for line in f if line.strip()]
    assert len(events) >= 1
    assert "iteration" in events[0]
