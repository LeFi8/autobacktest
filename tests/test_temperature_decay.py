"""Unit tests verifying the dynamic temperature decay schema inside the Orchestrator loop."""

from pathlib import Path

import git
import pytest

from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import run_optimization


class TemperatureTrackingProvider(MockProvider):
    """MockProvider subclass that records temperatures at each invocation."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.temperature = 0.7
        self.recorded_temperatures = []

    def generate_edit(self, context) -> any:
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

    # Write simple mock strategy and config
    strat_file = strat_dir / "momentum.py"
    strat_file.write_text(
        """
import pandas as pd
def generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Always invest in SPY
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

    # Initialize git repo and commit
    repo = git.Repo.init(tmp_path)
    repo.config_writer().set_value("user", "name", "Test User").release()
    repo.config_writer().set_value("user", "email", "test@test.com").release()
    repo.index.add(["strategies/momentum.py", "configs/momentum.yaml"])
    repo.index.commit("initial: baseline momentum strategy")

    return prog_file, strat_dir, conf_dir, tmp_path


def test_temperature_decay_progression(mock_project: tuple[Path, Path, Path, Path]) -> None:
    """Verifies that the sampling temperature decays linearly across multiple iterations."""
    prog_file, strat_dir, conf_dir, repo_root = mock_project
    provider = TemperatureTrackingProvider()

    # Run for 5 iterations: temperatures should be: 0.7, 0.55, 0.4, 0.25, 0.1
    run_optimization(
        program_path=prog_file,
        strategy_name="momentum",
        iterations=5,
        provider=provider,
        run_dir=repo_root / "runs",
        strategies_dir=strat_dir,
        configs_dir=conf_dir,
        repo_path=repo_root,
    )

    # 5 iterations should record exactly 5 temperature states
    assert len(provider.recorded_temperatures) == 5
    expected = [0.7, 0.55, 0.4, 0.25, 0.1]
    assert provider.recorded_temperatures == pytest.approx(expected)

    # Verify that original state is restored at the end
    assert provider.temperature == 0.7


def test_temperature_decay_single_iteration(mock_project: tuple[Path, Path, Path, Path]) -> None:
    """Verifies that temperature remains at start temperature if there is only 1 iteration."""
    prog_file, strat_dir, conf_dir, repo_root = mock_project
    provider = TemperatureTrackingProvider()

    run_optimization(
        program_path=prog_file,
        strategy_name="momentum",
        iterations=1,
        provider=provider,
        run_dir=repo_root / "runs",
        strategies_dir=strat_dir,
        configs_dir=conf_dir,
        repo_path=repo_root,
    )

    assert len(provider.recorded_temperatures) == 1
    assert provider.recorded_temperatures[0] == 0.7
    assert provider.temperature == 0.7
