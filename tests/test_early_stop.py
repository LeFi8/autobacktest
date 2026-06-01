"""Unit tests verifying the early-stop logic in the Orchestrator loop.

Early-stop fires when ``consecutive_no_accept`` reaches ``EARLY_STOP_PATIENCE``
(10) and exits the loop before all iterations are consumed.
"""

from pathlib import Path
from typing import Any

import git
import pandas as pd
import pytest

from autobacktest.config import settings
from autobacktest.evaluator.report import EvaluationReport, WindowReport
from autobacktest.gate import GateResult
from autobacktest.llm.base import AgentContext, AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import EARLY_STOP_PATIENCE, run_optimization

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class IterationCountingProvider(MockProvider):
    """Records number of generate_edit calls (outer iteration count)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.call_count = 0

    def generate_edit(self, context: AgentContext) -> AgentEdit:
        self.call_count += 1
        return super().generate_edit(context)


def test_early_stop_fires_before_all_iterations(
    mock_project: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Early-stop exits the loop once consecutive_no_accept reaches EARLY_STOP_PATIENCE.

    With all iterations rejected, the loop must stop at EARLY_STOP_PATIENCE and not
    run the remaining iterations.
    """
    prog_file, strat_dir, conf_dir, repo_root = mock_project

    # Patch evaluation to return a canned passing report.
    monkeypatch.setattr(
        "autobacktest.orchestrator.evaluate_strategy_detailed",
        lambda *_a, **_kw: _make_canned_report(sharpe=1.0),
    )

    # Diversity gates always pass so we reach the accept() gate.
    monkeypatch.setattr("autobacktest.orchestrator.max_config_similarity", lambda *_: 0.0)
    monkeypatch.setattr("autobacktest.orchestrator.check_returns_correlation", lambda *_: (True, 0.0))

    # Gate always rejects — consecutive_no_accept will climb every iteration.
    monkeypatch.setattr(
        "autobacktest.orchestrator.select",
        lambda *_a, **_kw: GateResult(
            accepted=False,
            reason="test rejection",
            failed_gate="target_metric_improvement",
        ),
    )
    monkeypatch.setattr(settings, "n_candidates", 1)

    total_iterations = EARLY_STOP_PATIENCE + 20  # well beyond patience
    provider = IterationCountingProvider()
    # Remove temperature so no-temp branch is exercised.
    provider.temperature = None  # type: ignore[assignment]

    result = run_optimization(
        program_path=prog_file,
        strategy_name="momentum",
        iterations=total_iterations,
        provider=provider,
        run_dir=repo_root / "runs",
        strategies_dir=strat_dir,
        configs_dir=conf_dir,
        repo_path=repo_root,
    )

    # The loop must have stopped at EARLY_STOP_PATIENCE, not at total_iterations.
    assert provider.call_count == EARLY_STOP_PATIENCE, (
        f"Expected exactly {EARLY_STOP_PATIENCE} LLM calls (early stop), got {provider.call_count}"
    )
    # No strategies were committed (all rejected).
    assert result.n_committed == 0


def test_early_stop_counter_resets_on_acceptance(
    mock_project: tuple[Path, Path, Path, Path],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """After an acceptance the consecutive counter resets and the loop continues.

    Schedule: the first acceptance occurs at iteration EARLY_STOP_PATIENCE // 2
    (halfway to the patience threshold). The counter resets to 0, so the loop
    must run for at least EARLY_STOP_PATIENCE more iterations before stopping.
    """
    prog_file, strat_dir, conf_dir, repo_root = mock_project

    monkeypatch.setattr(
        "autobacktest.orchestrator.evaluate_strategy_detailed",
        lambda *_a, **_kw: _make_canned_report(sharpe=1.0),
    )
    monkeypatch.setattr("autobacktest.orchestrator.max_config_similarity", lambda *_: 0.0)
    monkeypatch.setattr("autobacktest.orchestrator.check_returns_correlation", lambda *_: (True, 0.0))

    accept_at = EARLY_STOP_PATIENCE // 2  # iteration 5 (1-indexed)
    gate_call_count = [0]

    def patched_select(*_a: Any, **_kw: Any) -> GateResult:
        gate_call_count[0] += 1
        if gate_call_count[0] == accept_at:
            return GateResult(accepted=True)
        return GateResult(
            accepted=False,
            reason="test rejection",
            failed_gate="target_metric_improvement",
        )

    monkeypatch.setattr("autobacktest.orchestrator.select", patched_select)
    monkeypatch.setattr(
        "autobacktest.orchestrator.confirm",
        lambda *_a, **_kw: GateResult(accepted=True),
    )
    monkeypatch.setattr(settings, "n_candidates", 1)

    # Run with enough iterations that without reset the loop would have stopped
    # at EARLY_STOP_PATIENCE, but with reset it runs longer.
    total_iterations = EARLY_STOP_PATIENCE * 3
    provider = IterationCountingProvider()
    provider.temperature = None  # type: ignore[assignment]

    result = run_optimization(
        program_path=prog_file,
        strategy_name="momentum",
        iterations=total_iterations,
        provider=provider,
        run_dir=repo_root / "runs",
        strategies_dir=strat_dir,
        configs_dir=conf_dir,
        repo_path=repo_root,
    )

    # Acceptance at iteration `accept_at` resets the counter.
    # After the reset, EARLY_STOP_PATIENCE more consecutive rejections are needed.
    # Expected total iterations = accept_at + EARLY_STOP_PATIENCE
    expected_calls = accept_at + EARLY_STOP_PATIENCE
    assert provider.call_count == expected_calls, (
        f"Expected {expected_calls} LLM calls after reset, got {provider.call_count}"
    )
    # Exactly one strategy was committed (the acceptance at iteration accept_at).
    assert result.n_committed == 1
