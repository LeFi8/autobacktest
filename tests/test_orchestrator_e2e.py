"""End-to-end orchestrator tests with offline FakeProvider and scripted MockProvider."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import git
import numpy as np
import pandas as pd
import pytest

from autobacktest.gate import TargetMetric
from autobacktest.llm.base import AgentContext, AgentEdit
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import OrchestratorResult, run_optimization

# ---------------------------------------------------------------------------
# Synthetic data
# ---------------------------------------------------------------------------


def _make_synthetic_prices(start: str = "2013-01-01", end: str = "2025-01-01") -> pd.DataFrame:
    """Create a deterministic synthetic price DataFrame with HIGH and LOW assets.

    HIGH asset: 0.1% daily drift + tiny noise  → strong positive returns
    LOW  asset: 0.01% daily drift + tiny noise → weak positive returns
    Both use very low volatility so Sharpe differences are clear and the
    gate's hard drawdown/turnover constraints are easily satisfied.
    """
    dates = pd.bdate_range(start=start, end=end)
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


# ---------------------------------------------------------------------------
# Toy strategy code strings
# ---------------------------------------------------------------------------

BASELINE_STRATEGY = '''\
import pandas as pd
from typing import Any


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Always allocate 100% to LOW asset.

    Rebalances monthly using the last available business-day price each month.
    Uses groupby to ensure the index dates are always present in prices.index.
    """
    monthly_last = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly_last.index
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    weights["LOW"] = 1.0
    return weights
'''

IMPROVED_STRATEGY = '''\
import pandas as pd
from typing import Any


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Always allocate 100% to HIGH asset.

    Rebalances monthly using the last available business-day price each month.
    Uses groupby to ensure the index dates are always present in prices.index.
    """
    monthly_last = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly_last.index
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    weights["HIGH"] = 1.0
    return weights
'''

# The invalid strategy imports `os`, which is not in ALLOWED_IMPORTS — this
# will be caught by the AST static whitelist check and raise AST_BLOCKED_IMPORT.
INVALID_STRATEGY = '''\
import os
import pandas as pd
from typing import Any


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Bad edit: blocked import."""
    monthly_last = prices.groupby([prices.index.year, prices.index.month]).tail(1)
    idx = monthly_last.index
    weights = pd.DataFrame(0.0, index=idx, columns=prices.columns)
    weights["HIGH"] = 1.0
    return weights
'''

STRATEGY_CONFIG = """\
universe:
  - HIGH
  - LOW
benchmark: HIGH
momentum_lookback: 12
max_drawdown_limit: 0.50
turnover_limit: 5.0
"""

# Improved config differs enough from STRATEGY_CONFIG to pass the diversity gate
# (config similarity < 0.95 threshold).
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
Maximize risk-adjusted returns on the toy universe.

# Constraints
Max drawdown 50%. Turnover limit 5x.
"""


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Set up a minimal project layout with git repo and initial baseline commit."""
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


# ---------------------------------------------------------------------------
# Helper: build a FakeProvider that serves synthetic prices
# ---------------------------------------------------------------------------


def _make_fake_provider(synthetic_prices: pd.DataFrame) -> object:
    """Return a FakeProvider instance whose get_prices() returns synthetic data."""

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
                # Benchmark ticker may not match column names — return first column
                # renamed to the requested ticker so evaluation doesn't fail.
                first_col = synthetic_prices.columns[0]
                return synthetic_prices[[first_col]].rename(columns={first_col: tickers[0]})
            return synthetic_prices[available]

    return FakeProvider()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_e2e_full_run_commits_improved_strategy(project_root: Path) -> None:
    """Full orchestrator run: scripted MockProvider produces a known-good improvement.

    The IMPROVED_STRATEGY (HIGH asset, ~0.1%/day drift) beats the BASELINE
    (LOW asset, ~0.01%/day drift) on Sharpe, so the gate should accept it on
    the first iteration.  Iterations 2 and 3 re-submit the same edit; once
    IMPROVED_STRATEGY is already incumbent, the gate compares against itself
    and may or may not accept (no regression) — either outcome is fine.
    """
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    scripted_edit = AgentEdit(
        strategy_code=IMPROVED_STRATEGY,
        config_yaml=IMPROVED_CONFIG,
        reasoning="Switch allocation to HIGH asset for better risk-adjusted returns",
        raw_response="{}",
    )
    mock_provider = MockProvider(response=scripted_edit)

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
        )

    # --- Result shape ---
    assert isinstance(result, OrchestratorResult)
    assert result.run_id.startswith("toy-")
    assert result.branch.startswith("autobacktest/toy-")

    # At least iteration 1 should have been accepted (HIGH beats LOW on Sharpe)
    assert result.n_committed >= 1

    # Final report Sharpe should be clearly higher than LOW asset (~0.01%/day drift);
    # HIGH asset has ~0.1%/day drift so its annualized Sharpe is >> 1.0.
    assert result.final_report.observed_sharpe > 1.0

    # Verify the committed file actually contains the IMPROVED (HIGH) strategy code.
    repo = git.Repo(project_root)
    committed_code = repo.git.show(f"{result.branch}:strategies/toy.py")
    assert 'weights["HIGH"] = 1.0' in committed_code, (
        f"Expected HIGH allocation in committed strategy, got:\n{committed_code}"
    )

    # --- events.jsonl ---
    events_path = project_root / "runs" / result.run_id / "events.jsonl"
    assert events_path.exists(), f"events.jsonl not found at {events_path}"
    raw = events_path.read_text(encoding="utf-8").strip()
    lines = [ln for ln in raw.split("\n") if ln]
    assert len(lines) == 3, f"Expected 3 events, got {len(lines)}: {lines}"
    for line in lines:
        event = json.loads(line)
        assert "iteration" in event, f"Missing 'iteration' key in event: {event}"
        assert "timestamp" in event, f"Missing 'timestamp' key in event: {event}"

    # --- ledger.db ---
    ledger_path = project_root / "runs" / "ledger.db"
    assert ledger_path.exists(), "ledger.db not found"
    conn = sqlite3.connect(ledger_path)
    try:
        rows = conn.execute("SELECT COUNT(*) FROM attempts").fetchone()
        assert rows is not None
        # baseline (iteration 0) + up to 3 iteration attempts = at least 2
        assert rows[0] >= 2, f"Expected ≥2 attempts in ledger, got {rows[0]}"
    finally:
        conn.close()

    # --- git commits ---
    repo = git.Repo(project_root)
    run_branch = result.branch
    branch_commits = list(repo.iter_commits(run_branch))
    # Initial commit + at least one accepted-edit commit
    assert len(branch_commits) >= 2, f"Expected ≥2 commits on {run_branch}, got {len(branch_commits)}"

    # --- MockProvider call count ---
    # Iter 1: EXPLORE, 1 call, accepted → EXPLOIT
    # Iters 2-3: EXPLOIT, 1 call each (no diversity gates)
    expected_calls = 3
    assert len(mock_provider.calls) == expected_calls


def test_e2e_validation_failure_continues(project_root: Path) -> None:
    """When MockProvider returns an AST-invalid strategy, loop continues without crash.

    The INVALID_STRATEGY contains `import os` which is not in ALLOWED_IMPORTS.
    The orchestrator's preflight check catches this before touching real files,
    so no commits are made and each event records validation.passed == False.
    """
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    scripted_edit = AgentEdit(
        strategy_code=INVALID_STRATEGY,
        config_yaml=STRATEGY_CONFIG,
        reasoning="Bad edit with blocked import",
        raw_response="{}",
    )
    mock_provider = MockProvider(response=scripted_edit)

    with patch(
        "autobacktest.evaluator.evaluate.CachedDataProvider",
        return_value=fake_instance,
    ):
        result = run_optimization(
            program_path=project_root / "program.md",
            strategy_name="toy",
            iterations=2,
            provider=mock_provider,
            run_dir=project_root / "runs",
            strategies_dir=project_root / "strategies",
            configs_dir=project_root / "configs",
            target_metric=TargetMetric.SHARPE,
            repo_path=project_root,
            start_date="2013-01-01",
            end_date="2025-01-01",
        )

    # No commits since validation always fails
    assert result.n_committed == 0

    # 2 events written (one per iteration)
    events_path = project_root / "runs" / result.run_id / "events.jsonl"
    assert events_path.exists()
    raw = events_path.read_text(encoding="utf-8").strip()
    lines = [ln for ln in raw.split("\n") if ln]
    assert len(lines) == 2, f"Expected 2 events, got {len(lines)}"

    # Each event must record validation failure
    for line in lines:
        event = json.loads(line)
        assert "validation" in event, f"Missing 'validation' key in event: {event}"
        assert event["validation"] is not None, "validation should not be None"
        assert event["validation"]["passed"] is False, f"Expected validation.passed=False, got: {event['validation']}"

    # MockProvider was still called 2 times (edit was generated before validation)
    assert len(mock_provider.calls) == 2


def test_orchestrator_fail_fast_on_non_retryable_error(project_root: Path) -> None:
    """When a non-retryable LLMError is raised, the orchestrator immediately aborts."""
    from autobacktest.llm.base import LLMError
    from autobacktest.llm.mock_provider import MockProvider

    class FailingProvider(MockProvider):
        def generate_edit(self, _context: AgentContext) -> AgentEdit:
            raise LLMError(provider="mock", model="m", detail="Non-retryable config error", retryable=False)

    provider = FailingProvider()
    with pytest.raises(LLMError) as exc_info:
        run_optimization(
            program_path=project_root / "program.md",
            strategy_name="toy",
            iterations=5,
            provider=provider,
            run_dir=project_root / "runs",
            strategies_dir=project_root / "strategies",
            configs_dir=project_root / "configs",
            repo_path=project_root,
        )
    assert "Non-retryable config error" in str(exc_info.value)


@patch("autobacktest.orchestrator.sleep")
def test_orchestrator_continues_on_retryable_error(_mock_sleep: MagicMock, project_root: Path) -> None:
    """When a retryable LLMError is raised, the orchestrator logs and continues."""
    from autobacktest.llm.base import LLMError
    from autobacktest.llm.mock_provider import MockProvider

    class RetryableProvider(MockProvider):
        def generate_edit(self, _context: AgentContext) -> AgentEdit:
            raise LLMError(provider="mock", model="m", detail="Transient timeout", retryable=True)

    provider = RetryableProvider()
    # Since all iterations will fail with a retryable error, zero LLM calls will succeed.
    # Therefore, it should eventually raise a RuntimeError at the end of the loop, not on the first iteration.
    with pytest.raises(RuntimeError) as exc_info:
        run_optimization(
            program_path=project_root / "program.md",
            strategy_name="toy",
            iterations=3,
            provider=provider,
            run_dir=project_root / "runs",
            strategies_dir=project_root / "strategies",
            configs_dir=project_root / "configs",
            repo_path=project_root,
        )
    assert "Zero successful LLM calls" in str(exc_info.value)


@patch("autobacktest.orchestrator.sleep")
def test_orchestrator_retries_transient_error_with_backoff(mock_sleep: MagicMock, project_root: Path) -> None:
    """When a retryable LLMError is raised, the orchestrator retries and sleeps with backoff."""
    from autobacktest.llm.base import LLMError
    from autobacktest.llm.mock_provider import MockProvider

    call_count = 0

    class FlakyProvider(MockProvider):
        def generate_edit(self, _context: AgentContext) -> AgentEdit:
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise LLMError(provider="mock", model="m", detail="Transient timeout", retryable=True)
            return AgentEdit(
                strategy_code=BASELINE_STRATEGY,
                config_yaml=STRATEGY_CONFIG,
                reasoning="Succeeded on retry",
                raw_response="{}",
            )

    provider = FlakyProvider()

    # We expect 1 iteration.
    # On the first iteration, call 1 fails (transient retry 1), call 2 fails (transient retry 2), call 3 succeeds.
    run_optimization(
        program_path=project_root / "program.md",
        strategy_name="toy",
        iterations=1,
        provider=provider,
        run_dir=project_root / "runs",
        strategies_dir=project_root / "strategies",
        configs_dir=project_root / "configs",
        repo_path=project_root,
        start_date="2013-01-01",
        end_date="2025-01-01",
    )

    # Calls 1-2: transient LLMError → retried with backoff.
    # Call 3: succeeds, returns STRATEGY_CONFIG (identical to baseline).
    # Diversity gate fires: STRATEGY_CONFIG == baseline → retry up to MAX_DIVERSITY_RETRIES.
    # Calls 4 to (3 + MAX_DIVERSITY_RETRIES): each returns STRATEGY_CONFIG → diversity rejected.
    from autobacktest.orchestrator import MAX_DIVERSITY_RETRIES

    assert call_count == 3 + MAX_DIVERSITY_RETRIES
    # time.sleep called twice with exponential backoff: 2.0 ** 1 = 2.0s, and 2.0 ** 2 = 4.0s
    assert mock_sleep.call_count == 2
    mock_sleep.assert_any_call(2.0)
    mock_sleep.assert_any_call(4.0)


def test_exploit_mode_skips_diversity_gates(project_root: Path) -> None:
    """After acceptance, EXPLOIT mode should skip both diversity gates."""
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    scripted_edit = AgentEdit(
        strategy_code=IMPROVED_STRATEGY,
        config_yaml=IMPROVED_CONFIG,
        reasoning="Switch to HIGH",
        raw_response="{}",
    )
    mock_provider = MockProvider(response=scripted_edit)

    diversity_config_calls: list[int] = []
    diversity_returns_calls: list[int] = []

    import autobacktest.orchestrator as orch_mod

    original_max_config_sim = orch_mod.max_config_similarity
    original_check_returns = orch_mod.check_returns_correlation

    def tracking_max_config_sim(*args: object, **kwargs: object) -> object:
        diversity_config_calls.append(1)
        return original_max_config_sim(*args, **kwargs)  # type: ignore[arg-type]

    def tracking_check_returns(*args: object, **kwargs: object) -> object:
        diversity_returns_calls.append(1)
        return original_check_returns(*args, **kwargs)  # type: ignore[arg-type]

    with (
        patch("autobacktest.orchestrator.max_config_similarity", side_effect=tracking_max_config_sim),
        patch("autobacktest.orchestrator.check_returns_correlation", side_effect=tracking_check_returns),
        patch("autobacktest.evaluator.evaluate.CachedDataProvider", return_value=fake_instance),
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
        )

    # Iter 1 is in EXPLORE mode → diversity gates called
    # Iters 2-3 are in EXPLOIT mode → diversity gates NOT called
    assert result.n_committed >= 1
    from autobacktest.orchestrator import MAX_DIVERSITY_RETRIES

    # Diversity config gate called only for iter 1 (EXPLORE)
    # After acceptance, mode=EXPLOIT, no diversity calls
    assert len(diversity_config_calls) <= 1 + MAX_DIVERSITY_RETRIES  # iter 1 only
    # In iters 2-3 (EXPLOIT), diversity gates not called
    # Total calls should be from iter 1 only
    assert len(diversity_returns_calls) <= 1


def test_mode_logged_in_events(project_root: Path) -> None:
    """Each event in events.jsonl should contain a 'mode' key."""
    synthetic_prices = _make_synthetic_prices()
    fake_instance = _make_fake_provider(synthetic_prices)

    scripted_edit = AgentEdit(
        strategy_code=IMPROVED_STRATEGY,
        config_yaml=IMPROVED_CONFIG,
        reasoning="Switch to HIGH",
        raw_response="{}",
    )
    mock_provider = MockProvider(response=scripted_edit)

    with patch("autobacktest.evaluator.evaluate.CachedDataProvider", return_value=fake_instance):
        result = run_optimization(
            program_path=project_root / "program.md",
            strategy_name="toy",
            iterations=2,
            provider=mock_provider,
            run_dir=project_root / "runs",
            strategies_dir=project_root / "strategies",
            configs_dir=project_root / "configs",
            target_metric=TargetMetric.SHARPE,
            repo_path=project_root,
            start_date="2013-01-01",
            end_date="2025-01-01",
        )

    events_path = project_root / "runs" / result.run_id / "events.jsonl"
    raw = events_path.read_text(encoding="utf-8").strip()
    for line in raw.split("\n"):
        if not line:
            continue
        event = json.loads(line)
        assert "mode" in event, f"Missing 'mode' key in event: {event}"
        assert event["mode"] in ("explore", "exploit"), f"Invalid mode value: {event['mode']}"
