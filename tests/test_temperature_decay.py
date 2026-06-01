from pathlib import Path
from typing import Any

import git
import pytest

from autobacktest.llm.base import AgentContext, AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import run_optimization


class TemperatureTrackingProvider(MockProvider):
    """MockProvider subclass that records temperatures at each invocation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.temperature = 0.7
        self.recorded_temperatures: list[float] = []

    def generate_edit(self, context: AgentContext) -> AgentEdit:
        self.recorded_temperatures.append(self.temperature)
        return super().generate_edit(context)


@pytest.fixture
def mock_project(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Setup mock directories, program file, and commit to a Git repo."""
    prog_file = tmp_path / "program.md"
    prog_file.write_text(
        """# Objective
Optimize momentum logic.

# Constraints
Maximize return and keep turnover low.
""",
        encoding="utf-8",
    )

    strat_dir = tmp_path / "strategies"
    conf_dir = tmp_path / "configs"
    strat_dir.mkdir()
    conf_dir.mkdir()

    strat_file = strat_dir / "momentum.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    if "SPY" in weights.columns:
        weights["SPY"] = 1.0
    return weights
""",
        encoding="utf-8",
    )

    conf_file = conf_dir / "momentum.yaml"
    conf_file.write_text(
        """
universe:
  - SPY
benchmark: SPY
momentum_lookback: 12
params:
  offensive_universe:
    - SPY
""",
        encoding="utf-8",
    )

    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    repo.index.add(["strategies/momentum.py", "configs/momentum.yaml"])
    repo.index.commit("initial: baseline momentum strategy")

    return prog_file, strat_dir, conf_dir, tmp_path


def test_adaptive_temperature_on_continuous_failures(
    mock_project: tuple[Path, Path, Path, Path],
) -> None:
    """Verifies that in explore mode temperature scales continuously based on rolling failures."""
    prog_file, strat_dir, conf_dir, repo_root = mock_project
    provider = TemperatureTrackingProvider()
    provider.temperature = 0.7  # start_temp = 0.7, min_temp = 0.1

    run_optimization(
        program_path=prog_file,
        strategy_name="momentum",
        iterations=3,
        provider=provider,
        run_dir=repo_root / "runs",
        strategies_dir=strat_dir,
        configs_dir=conf_dir,
        repo_path=repo_root,
    )

    # 3 iterations x 3 candidates each = 9 LLM calls. All 3 calls in an
    # iteration share the same temperature (set once per iteration).
    assert len(provider.recorded_temperatures) == 9

    # k=1: rolling history is empty -> failure_rate = 0.6. Temp = 0.1 + 0.6 * 0.6 = 0.46
    assert all(t == pytest.approx(0.46) for t in provider.recorded_temperatures[0:3])

    # k=2: k=1 failed -> rolling history [False] -> failure_rate = 1.0. Temp = 0.7
    assert all(t == pytest.approx(0.7) for t in provider.recorded_temperatures[3:6])

    # k=3: k=1,2 failed -> rolling history [False, False] -> failure_rate = 1.0. Temp = 0.7
    assert all(t == pytest.approx(0.7) for t in provider.recorded_temperatures[6:9])
