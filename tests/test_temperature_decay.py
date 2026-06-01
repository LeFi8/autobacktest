"""Unit tests verifying the dynamic temperature decay schema inside the Orchestrator loop."""

from pathlib import Path
from typing import Any

import git
import pytest

from autobacktest.evaluator.report import EvaluationReport
from autobacktest.gate import GateResult
from autobacktest.llm.base import AgentContext, AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import STUCK_ESCALATION_FACTOR, STUCK_THRESHOLD, run_optimization


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

    # Each iteration records its temperature on the initial call AND each diversity retry.
    # The MockProvider always returns the same (baseline) config, so diversity fires and
    # exhausts MAX_DIVERSITY_RETRIES retries every iteration.
    from autobacktest.orchestrator import MAX_DIVERSITY_RETRIES

    calls_per_iter = 1 + MAX_DIVERSITY_RETRIES
    assert len(provider.recorded_temperatures) == 5 * calls_per_iter

    # The temperature for each iteration's calls must match the expected decay value.
    # Temperature is set once per outer iteration and unchanged during diversity retries.
    expected_per_iter = [0.7, 0.55, 0.4, 0.25, 0.1]
    for i, expected_temp in enumerate(expected_per_iter):
        for j in range(calls_per_iter):
            idx = i * calls_per_iter + j
            assert provider.recorded_temperatures[idx] == pytest.approx(expected_temp), (
                f"Iteration {i + 1}, retry {j}: expected {expected_temp:.3f}, "
                f"got {provider.recorded_temperatures[idx]:.4f}"
            )

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

    # With diversity retry: MockProvider returns the same config as baseline → diversity
    # fires and exhausts MAX_DIVERSITY_RETRIES retries.
    from autobacktest.orchestrator import MAX_DIVERSITY_RETRIES

    calls_per_iter = 1 + MAX_DIVERSITY_RETRIES
    assert len(provider.recorded_temperatures) == calls_per_iter
    for temp in provider.recorded_temperatures:
        assert temp == pytest.approx(0.7)
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

    # MockProvider returns the baseline config → diversity fires and exhausts
    # MAX_DIVERSITY_RETRIES retries each iteration.  Temperature is set once per
    # outer iteration and held constant during intra-iteration retries.
    from autobacktest.orchestrator import MAX_DIVERSITY_RETRIES

    calls_per_iter = 1 + MAX_DIVERSITY_RETRIES
    assert len(provider.recorded_temperatures) == n_iter * calls_per_iter

    # First STUCK_THRESHOLD iterations (k=1..5) use normal monotonic decay.
    # consecutive_no_accept < STUCK_THRESHOLD at the START of each of these iterations.
    for i in range(STUCK_THRESHOLD):
        decay_factor = i / (n_iter - 1)
        expected = start_temp - decay_factor * (start_temp - min_temp)
        for j in range(calls_per_iter):
            idx = i * calls_per_iter + j
            assert provider.recorded_temperatures[idx] == pytest.approx(expected), (
                f"Iteration {i + 1}, retry {j}: expected normal decay {expected:.4f}, "
                f"got {provider.recorded_temperatures[idx]:.4f}"
            )

    # Iterations STUCK_THRESHOLD+1 onward should all use the escalated temperature.
    escalated = min(start_temp, min_temp + (start_temp - min_temp) * 0.8)
    for i in range(STUCK_THRESHOLD, n_iter):
        for j in range(calls_per_iter):
            idx = i * calls_per_iter + j
            assert provider.recorded_temperatures[idx] == pytest.approx(escalated), (
                f"Iteration {i + 1}, retry {j}: expected escalated temp {escalated:.4f}, "
                f"got {provider.recorded_temperatures[idx]:.4f}"
            )

    # Original temperature is restored after the run.
    assert provider.temperature == pytest.approx(start_temp)


def test_stuck_counter_resets_on_acceptance(
    mock_project: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Counter resets to 0 on acceptance, restoring normal decay for subsequent iterations.

    Runs through run_optimization with monkeypatched evaluation, diversity, and gate
    so that: iterations 1-5 are rejected (escalation kicks in at 6), iteration 6 is
    accepted (counter resets), then iterations 7-9 return to normal decay.
    """
    import pandas as pd

    from autobacktest.evaluator.report import WindowReport

    prog_file, strat_dir, conf_dir, repo_root = mock_project
    start_temp = 0.7
    min_temp = 0.1
    n_iter = STUCK_THRESHOLD + 4  # 9 iterations total

    # Canned evaluation response — used for both baseline and all candidate iterations.
    def _make_canned_report(sharpe: float = 1.0) -> tuple[EvaluationReport, pd.Series]:
        window = WindowReport(
            start_date="2020-01-01",
            end_date="2020-12-31",
            annualized_return=0.12,
            annualized_volatility=0.10,
            sharpe_ratio=sharpe,
            sortino_ratio=1.5,
            max_drawdown=0.10,
            turnover=0.5,
            information_ratio=0.4,
        )
        report = EvaluationReport(
            strategy_name="momentum",
            dataset_hash="testhash",
            gates_passed={"max_drawdown": True, "turnover": True, "regimes": True},
            is_accepted=False,
            rejection_reason=None,
            holdout_metrics=window,
            in_sample_metrics=window,
            walk_forward_metrics=[window],
            regime_drawdowns={},
            regime_passed=True,
            mc_sharpe_5th=0.8,
            mc_sharpe_50th=1.0,
            mc_sharpe_95th=1.2,
            observed_sharpe=sharpe,
            effective_trials=1,
            deflated_sharpe=sharpe,
        )
        returns = pd.Series([0.01, -0.005, 0.008], dtype=float)
        return report, returns

    eval_call_count = [0]

    def patched_evaluate(*_args: Any, **_kwargs: Any) -> tuple[EvaluationReport, pd.Series]:
        eval_call_count[0] += 1
        return _make_canned_report(sharpe=1.0)

    monkeypatch.setattr("autobacktest.orchestrator.evaluate_strategy_detailed", patched_evaluate)

    # Patch diversity gates to always pass.
    monkeypatch.setattr("autobacktest.orchestrator.max_config_similarity", lambda *_: 0.0)
    monkeypatch.setattr("autobacktest.orchestrator.check_returns_correlation", lambda *_: (True, 0.0))

    # Patch select() to accept on the STUCK_THRESHOLD+1-th gate call, reject otherwise.
    gate_call_count = [0]

    def patched_select(*_args: Any, **_kwargs: Any) -> GateResult:
        gate_call_count[0] += 1
        if gate_call_count[0] == STUCK_THRESHOLD + 1:
            return GateResult(accepted=True)
        return GateResult(accepted=False, reason="test rejection", failed_gate="target_metric_improvement")

    monkeypatch.setattr("autobacktest.orchestrator.select", patched_select)
    monkeypatch.setattr(
        "autobacktest.orchestrator.confirm",
        lambda *_a, **_kw: GateResult(accepted=True),
    )

    provider = TemperatureTrackingProvider()
    provider.temperature = start_temp

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

    # k=1..5 (index 0..4): consecutive_no_accept was 0,1,2,3,4 — normal decay.
    for i in range(STUCK_THRESHOLD):
        decay_factor = i / (n_iter - 1)
        expected = start_temp - decay_factor * (start_temp - min_temp)
        assert provider.recorded_temperatures[i] == pytest.approx(expected), (
            f"Iteration {i + 1}: expected normal decay {expected:.4f}, got {provider.recorded_temperatures[i]:.4f}"
        )

    # k=6 (index 5): consecutive_no_accept == 5 >= STUCK_THRESHOLD — escalated.
    # This is the iteration that gets accepted, resetting the counter to 0.
    escalated = min(start_temp, min_temp + (start_temp - min_temp) * STUCK_ESCALATION_FACTOR)
    assert provider.recorded_temperatures[STUCK_THRESHOLD] == pytest.approx(escalated)

    # k=7..9 (index 6..8): after acceptance, mode=EXPLOIT — temperature fixed at min_temp.
    for i in range(STUCK_THRESHOLD + 1, n_iter):
        k = i + 1
        assert provider.recorded_temperatures[i] == pytest.approx(min_temp), (
            f"Iteration {k}: expected EXPLOIT min_temp {min_temp:.4f} after acceptance, "
            f"got {provider.recorded_temperatures[i]:.4f}"
        )
