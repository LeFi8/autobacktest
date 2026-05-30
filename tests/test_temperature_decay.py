"""Unit tests verifying the dynamic temperature decay schema inside the Orchestrator loop."""

from pathlib import Path

import git
import pytest

from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import STUCK_THRESHOLD, run_optimization


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


def test_stuck_temperature_escalation(mock_project: tuple[Path, Path, Path, Path]) -> None:
    """Temperature bumps back toward start_temp after STUCK_THRESHOLD consecutive rejections.

    The MockProvider returns an identity edit (same code, same config) every iteration,
    so all candidates are rejected by the gate.  After STUCK_THRESHOLD rejections the
    temperature should switch from monotonic decay to the escalation formula:
        min(start_temp, min_temp + (start_temp - min_temp) * 0.8)
    """
    prog_file, strat_dir, conf_dir, repo_root = mock_project
    start_temp = 0.7
    min_temp = 0.1
    provider = TemperatureTrackingProvider()
    provider.temperature = start_temp

    # Run enough iterations so we cross STUCK_THRESHOLD (default 5).
    # Iterations STUCK_THRESHOLD+1 ... onwards should all use the escalated temperature.
    n_iter = STUCK_THRESHOLD + 3  # 8 iterations total
    run_optimization(
        program_path=prog_file,
        strategy_name="momentum",
        iterations=n_iter,
        provider=provider,
        run_dir=repo_root / "runs",
        strategies_dir=strat_dir,
        configs_dir=conf_dir,
        repo_path=repo_root,
    )

    assert len(provider.recorded_temperatures) == n_iter

    # First STUCK_THRESHOLD iterations (k=1..5) use normal monotonic decay.
    # consecutive_no_accept < STUCK_THRESHOLD at the START of each of these iterations.
    for i in range(STUCK_THRESHOLD):
        decay_factor = i / (n_iter - 1)
        expected = start_temp - decay_factor * (start_temp - min_temp)
        assert provider.recorded_temperatures[i] == pytest.approx(expected), (
            f"Iteration {i + 1}: expected normal decay {expected:.4f}, "
            f"got {provider.recorded_temperatures[i]:.4f}"
        )

    # Iterations STUCK_THRESHOLD+1 onward should all use the escalated temperature.
    escalated = min(start_temp, min_temp + (start_temp - min_temp) * 0.8)
    for i in range(STUCK_THRESHOLD, n_iter):
        assert provider.recorded_temperatures[i] == pytest.approx(escalated), (
            f"Iteration {i + 1}: expected escalated temp {escalated:.4f}, "
            f"got {provider.recorded_temperatures[i]:.4f}"
        )

    # Original temperature is restored after the run.
    assert provider.temperature == pytest.approx(start_temp)


def test_stuck_counter_resets_on_acceptance() -> None:
    """Counter resets to 0 on acceptance, restoring normal decay for subsequent iterations.

    Direct unit test of the temperature selection state machine: simulate the
    consecutive_no_accept counter incrementing on rejection, resetting on acceptance,
    and verify that temperature follows the correct formula at every step.
    """
    # Unit test: simulate the temperature selection logic directly.
    start_temp = 0.8
    min_temp = 0.1
    total_iters = 12

    def compute_temperature(k: int, consecutive: int) -> float:
        if consecutive >= STUCK_THRESHOLD:
            return min(start_temp, min_temp + (start_temp - min_temp) * 0.8)
        if total_iters > 1:
            decay_factor = (k - 1) / (total_iters - 1)
            return start_temp - decay_factor * (start_temp - min_temp)
        return start_temp

    # Simulate: all-reject for first 6 iterations, accept on 7th, then all-reject again.
    consecutive = 0
    temps = []
    for k in range(1, total_iters + 1):
        temps.append(compute_temperature(k, consecutive))
        if k == 7:
            # acceptance: reset counter
            consecutive = 0
        else:
            consecutive += 1

    # k=1..5 (index 0..4): consecutive was 0,1,2,3,4 — all < 5, normal decay
    for i in range(STUCK_THRESHOLD):
        decay_factor = i / (total_iters - 1)
        expected = start_temp - decay_factor * (start_temp - min_temp)
        assert temps[i] == pytest.approx(expected)

    # k=6 (index 5): consecutive was 5 >= STUCK_THRESHOLD — escalated
    escalated = min(start_temp, min_temp + (start_temp - min_temp) * 0.8)
    assert temps[5] == pytest.approx(escalated)

    # k=7 (index 6): consecutive was still 5 at entry (reset happens AFTER compute)
    assert temps[6] == pytest.approx(escalated)

    # k=8 (index 7): consecutive reset to 0 after k=7 acceptance, so normal decay
    k = 8
    decay_factor = (k - 1) / (total_iters - 1)
    expected_normal = start_temp - decay_factor * (start_temp - min_temp)
    assert temps[7] == pytest.approx(expected_normal)
