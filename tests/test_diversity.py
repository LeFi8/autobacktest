"""Unit and integration tests for the diversity gate module."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import git
import numpy as np
import pandas as pd
import pytest

from autobacktest.gate import TargetMetric
from autobacktest.llm.base import AgentContext, AgentEdit, LLMProvider
from autobacktest.llm.mock_provider import MockProvider
from autobacktest.orchestrator import run_optimization
from autobacktest.strategy.diversity import (
    check_returns_correlation,
    config_similarity,
    extract_config_fingerprint,
    max_config_similarity,
)
from tests.test_orchestrator_e2e import (
    BASELINE_STRATEGY,
    IMPROVED_CONFIG,
    IMPROVED_STRATEGY,
    STRATEGY_CONFIG,
    _make_fake_provider,
    _make_synthetic_prices,
)

# ---------------------------------------------------------------------------
# Sequence provider — returns different edits on successive calls
# ---------------------------------------------------------------------------


class SequenceProvider(LLMProvider):
    """Returns edits from a list in order; repeats the last on overflow."""

    def __init__(self, responses: list[AgentEdit]) -> None:
        self.responses = responses
        self.calls: list[AgentContext] = []
        self.temperature: float = 1.0

    @property
    def provider_name(self) -> str:
        return "sequence"

    def generate_edit(self, context: AgentContext) -> AgentEdit:
        self.calls.append(context)
        idx = min(len(self.calls) - 1, len(self.responses) - 1)
        return self.responses[idx]


# ---------------------------------------------------------------------------
# Test config YAML strings
# ---------------------------------------------------------------------------

HAA_CONFIG_A = """\
universe:
  - SPY
  - IWM
  - VEA
  - VWO
benchmark: SPY
momentum_lookback: 12
max_drawdown_limit: 0.15
turnover_limit: 2.0
canary_hysteresis: 0.02
"""

HAA_CONFIG_B = """\
universe:
  - SPY
  - IWM
  - VEA
  - VWO
benchmark: SPY
momentum_lookback: 6
max_drawdown_limit: 0.20
turnover_limit: 3.0
canary_hysteresis: 0.03
"""

HAA_CONFIG_C = """\
universe:
  - SPY
  - IWM
benchmark: SPY
momentum_lookback: 12
max_drawdown_limit: 0.15
turnover_limit: 2.0
"""

# ---------------------------------------------------------------------------
# Fixture: minimal project for integration tests
# ---------------------------------------------------------------------------

PROGRAM_MD = """\
# Objective
Maximize Sharpe.

# Constraints
None.
"""


@pytest.fixture
def project_root(tmp_path: Path) -> Path:
    """Set up a minimal project directory with git repo and baseline strategy."""
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


# ===================================================================
# Config similarity tests (7 tests)
# ===================================================================


class TestConfigSimilarity:
    """Unit tests for config_fingerprint, config_similarity, max_config_similarity."""

    def test_identical_configs_similarity_1(self) -> None:
        """Same config YAML twice → similarity ~1.0."""
        fp1 = extract_config_fingerprint(HAA_CONFIG_A)
        fp2 = extract_config_fingerprint(HAA_CONFIG_A)
        assert config_similarity(fp1, fp2) == pytest.approx(1.0)
        assert max_config_similarity(HAA_CONFIG_A, [HAA_CONFIG_A]) == pytest.approx(1.0)

    def test_different_configs_low_similarity(self) -> None:
        """Completely different universes and params → low similarity."""
        fp_a = extract_config_fingerprint(HAA_CONFIG_A)

        # Config with disjoint assets and completely different params
        different = """\
universe:
  - DBC
  - GLD
benchmark: GLD
momentum_lookback: 1
max_drawdown_limit: 0.30
turnover_limit: 5.0
canary_hysteresis: 0.05
"""
        fp_b = extract_config_fingerprint(different)
        sim = config_similarity(fp_a, fp_b)
        assert sim < 0.5

    def test_hysteresis_variation_detected(self) -> None:
        """Small numeric variation in canary_hysteresis is detected (not 1.0)."""
        fp_a = extract_config_fingerprint(HAA_CONFIG_A)  # hysteresis=0.02
        fp_b = extract_config_fingerprint(HAA_CONFIG_B)  # hysteresis=0.03
        sim = config_similarity(fp_a, fp_b)
        assert sim < 1.0

    def test_universe_change_detected(self) -> None:
        """Different universe sets produce different similarity than same universe."""
        fp_same = extract_config_fingerprint(HAA_CONFIG_A)

        # Same universe as A → Jaccard = 1.0 for universe
        fp_same_u = extract_config_fingerprint(HAA_CONFIG_A)
        sim_same = config_similarity(fp_same, fp_same_u)

        # Smaller universe (only 2 assets) → Jaccard = 0.5 for universe
        fp_diff = extract_config_fingerprint(HAA_CONFIG_C)
        sim_diff = config_similarity(fp_same, fp_diff)

        assert sim_diff < sim_same

    def test_empty_config(self) -> None:
        """Empty or whitespace YAML → no crash, empty fingerprint."""
        fp_empty = extract_config_fingerprint("")
        assert fp_empty.numeric_params == {}
        assert fp_empty.set_fields == {}

        fp_ws = extract_config_fingerprint("   \n\n  ")
        assert fp_ws.numeric_params == {}
        assert fp_ws.set_fields == {}

        # max_config_similarity with empty candidate against real history
        sim = max_config_similarity("", [HAA_CONFIG_A])
        assert 0.0 <= sim <= 1.0

    def test_missing_params_neutral(self) -> None:
        """Config missing keys from another config still computes similarity."""
        config_full = """\
universe: [SPY]
momentum_lookback: 12
max_drawdown_limit: 0.15
turnover_limit: 2.0
extra_param: 99
"""
        config_partial = """\
universe: [SPY]
momentum_lookback: 12
max_drawdown_limit: 0.15
"""
        sim = max_config_similarity(config_full, [config_partial])
        assert 0.0 <= sim <= 1.0
        assert sim < 1.0  # missing extra_param means not identical

    def test_unknown_param_degenerate_range_no_inflation(self) -> None:
        """Unknown param with degenerate range (hi==lo) does NOT inflate similarity."""
        # Two configs that differ on known params but share an unknown param with same value
        candidate = """\
universe: [SPY, IWM]
momentum_lookback: 12
max_drawdown_limit: 0.15
turnover_limit: 2.0
vol_target: 0.12
"""
        historical = """\
universe: [SPY, IWM]
momentum_lookback: 6
max_drawdown_limit: 0.20
turnover_limit: 3.0
vol_target: 0.12
"""
        sim_with = max_config_similarity(candidate, [historical])

        # Same configs without vol_target
        candidate_no = """\
universe: [SPY, IWM]
momentum_lookback: 12
max_drawdown_limit: 0.15
turnover_limit: 2.0
"""
        historical_no = """\
universe: [SPY, IWM]
momentum_lookback: 6
max_drawdown_limit: 0.20
turnover_limit: 3.0
"""
        sim_without = max_config_similarity(candidate_no, [historical_no])

        # The degenerate-range param must NOT push cosine toward 1.0
        assert abs(sim_with - sim_without) < 0.001, f"Inflation detected: with={sim_with:.4f} without={sim_without:.4f}"


# ===================================================================
# Returns correlation tests (4 tests)
# ===================================================================


class TestReturnsCorrelation:
    """Unit tests for check_returns_correlation."""

    def test_returns_identical_correlation(self) -> None:
        """Identical return series → fails threshold (corr=1.0 > 0.90)."""
        dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
        rng = np.random.default_rng(42)
        returns = pd.Series(rng.normal(0.001, 0.02, 500), index=dates, name="candidate")

        hist = pd.DataFrame({1: returns.values}, index=dates)

        passed, max_corr = check_returns_correlation(returns, hist)
        assert not passed
        assert max_corr > 0.999

    def test_returns_independent_correlation(self) -> None:
        """Largely uncorrelated series → passes threshold."""
        dates = pd.bdate_range("2020-01-01", periods=500, freq="B")
        rng = np.random.default_rng(42)
        candidate = pd.Series(rng.normal(0.001, 0.02, 500), index=dates, name="candidate")

        # Historical series is deliberately different
        rng2 = np.random.default_rng(99)
        hist_returns = rng2.normal(-0.0005, 0.03, 500)
        hist = pd.DataFrame({1: hist_returns}, index=dates)

        _passed, max_corr = check_returns_correlation(candidate, hist)
        assert max_corr < 0.90  # should be well below threshold

    def test_returns_short_overlap(self) -> None:
        """Less than 60 overlapping days → passes trivially (can't compute)."""
        dates_a = pd.bdate_range("2020-01-01", periods=30, freq="B")
        dates_b = pd.bdate_range("2025-01-01", periods=30, freq="B")
        rng = np.random.default_rng(42)

        candidate = pd.Series(rng.normal(0.001, 0.02, 30), index=dates_a, name="candidate")
        hist = pd.DataFrame({1: rng.normal(0.001, 0.02, 30)}, index=dates_b)

        passed, max_corr = check_returns_correlation(candidate, hist)
        assert passed
        assert max_corr == 0.0

    def test_returns_empty_series(self) -> None:
        """Empty candidate or empty historical matrix → passes trivially."""
        empty_series = pd.Series(dtype=float, name="candidate")
        empty_matrix = pd.DataFrame()
        nonempty_matrix = pd.DataFrame({1: [0.001, 0.002, 0.003]})

        passed1, corr1 = check_returns_correlation(empty_series, nonempty_matrix)
        assert passed1
        assert corr1 == 0.0

        dates = pd.bdate_range("2020-01-01", periods=100, freq="B")
        rng = np.random.default_rng(42)
        candidate = pd.Series(rng.normal(0.001, 0.02, 100), index=dates, name="candidate")

        passed2, corr2 = check_returns_correlation(candidate, empty_matrix)
        assert passed2
        assert corr2 == 0.0


# ===================================================================
# Integration tests (2 tests)
# ===================================================================


class TestDiversityGateIntegration:
    """Integration tests: diversity gates wired in orchestrator loop."""

    def test_orchestrator_skips_tier1(self, project_root: Path) -> None:
        """Edit with config identical to baseline → Tier 1 rejects before backtest."""
        synthetic_prices = _make_synthetic_prices()
        fake_instance = _make_fake_provider(synthetic_prices)

        # Edit has same config as baseline → 100% config similarity → Tier 1 reject
        same_config_edit = AgentEdit(
            strategy_code=BASELINE_STRATEGY,
            config_yaml=STRATEGY_CONFIG,
            reasoning="Identical to baseline",
            raw_response="{}",
        )
        mock_provider = MockProvider(response=same_config_edit)

        with patch(
            "autobacktest.evaluator.evaluate.CachedDataProvider",
            return_value=fake_instance,
        ):
            result = run_optimization(
                program_path=project_root / "program.md",
                strategy_name="toy",
                iterations=1,
                provider=mock_provider,
                run_dir=project_root / "runs",
                strategies_dir=project_root / "strategies",
                configs_dir=project_root / "configs",
                target_metric=TargetMetric.SHARPE,
                repo_path=project_root,
                start_date="2013-01-01",
                end_date="2025-01-01",
            )

        # No accepted edit (diversity gate rejected it)
        assert result.n_committed == 0

        # Event must contain the diversity rejection
        events_path = project_root / "runs" / result.run_id / "events.jsonl"
        assert events_path.exists()
        events = [json.loads(ln) for ln in events_path.read_text().strip().split("\n") if ln]
        assert len(events) >= 1
        ev = events[0]
        assert "diversity" in ev
        assert ev["diversity"]["tier"] == "config"
        assert ev["diversity"]["passed"] is False

        # No evaluation ran (Tier 1 gate fires before backtest)
        assert ev["evaluation"] is None

    def test_orchestrator_rejects_tier2(self, project_root: Path) -> None:
        """Edit with returns nearly identical to baseline → Tier 2 rejects post-backtest."""
        synthetic_prices = _make_synthetic_prices()
        fake_instance = _make_fake_provider(synthetic_prices)

        # Edit uses BASELINE_STRATEGY code (allocates to LOW) with a different config
        # to pass Tier 1. Since the strategy logic is unchanged, returns are nearly
        # identical to the baseline → Tier 2 rejects.
        same_returns_edit = AgentEdit(
            strategy_code=BASELINE_STRATEGY,
            config_yaml=IMPROVED_CONFIG,
            reasoning="Same strategy, different config",
            raw_response="{}",
        )
        mock_provider = MockProvider(response=same_returns_edit)

        with patch(
            "autobacktest.evaluator.evaluate.CachedDataProvider",
            return_value=fake_instance,
        ):
            result = run_optimization(
                program_path=project_root / "program.md",
                strategy_name="toy",
                iterations=1,
                provider=mock_provider,
                run_dir=project_root / "runs",
                strategies_dir=project_root / "strategies",
                configs_dir=project_root / "configs",
                target_metric=TargetMetric.SHARPE,
                repo_path=project_root,
                start_date="2013-01-01",
                end_date="2025-01-01",
            )

        # No accepted edit (Tier 2 diversity gate rejected)
        assert result.n_committed == 0

        # Event must contain the diversity rejection with tier = returns
        events_path = project_root / "runs" / result.run_id / "events.jsonl"
        assert events_path.exists()
        events = [json.loads(ln) for ln in events_path.read_text().strip().split("\n") if ln]
        assert len(events) >= 1
        ev = events[0]
        assert "diversity" in ev
        assert ev["diversity"]["tier"] == "returns"
        assert ev["diversity"]["passed"] is False

        # Verification that Tier 2 rejection was recorded in the ledger:
        # the failed attempt's config should be available for future diversity checks
        import sqlite3

        ledger_path = project_root / "runs" / "ledger.db"
        conn = sqlite3.connect(ledger_path)
        rows = conn.execute(
            "SELECT iteration, accepted, rejection_reason FROM attempts WHERE run_id = ? ORDER BY id",
            (result.run_id,),
        ).fetchall()
        conn.close()
        assert len(rows) >= 2  # baseline (iter 0) + tier 2 rejection (iter 1)
        assert rows[-1][1] == 0  # accepted=False
        assert rows[-1][2] == "diversity_tier2_returns"  # specific rejection reason

    def test_diversity_retry_succeeds_on_second_proposal(self, project_root: Path) -> None:
        """Tier-1 diversity retry: first proposal is too-similar, second passes.

        The iteration budget must NOT be consumed — the outer loop only advances
        after the diversity gate passes (or retries are exhausted).
        """
        synthetic_prices = _make_synthetic_prices()
        fake_instance = _make_fake_provider(synthetic_prices)

        # First call → same config as baseline (100% similarity → diversity rejected)
        same_config_edit = AgentEdit(
            strategy_code=BASELINE_STRATEGY,
            config_yaml=STRATEGY_CONFIG,
            reasoning="Identical config",
            raw_response="{}",
        )
        # Second call → different config (passes diversity) + improved strategy
        diverse_edit = AgentEdit(
            strategy_code=IMPROVED_STRATEGY,
            config_yaml=IMPROVED_CONFIG,
            reasoning="Diverse config",
            raw_response="{}",
        )

        provider = SequenceProvider(responses=[same_config_edit, diverse_edit])

        with patch(
            "autobacktest.evaluator.evaluate.CachedDataProvider",
            return_value=fake_instance,
        ):
            result = run_optimization(
                program_path=project_root / "program.md",
                strategy_name="toy",
                iterations=1,
                provider=provider,
                run_dir=project_root / "runs",
                strategies_dir=project_root / "strategies",
                configs_dir=project_root / "configs",
                target_metric=TargetMetric.SHARPE,
                repo_path=project_root,
                start_date="2013-01-01",
                end_date="2025-01-01",
            )

        # The second proposal (diverse_edit) should have been accepted.
        # The retry happened within the same iteration, so budget was not wasted.
        assert result.n_committed >= 1
        # Provider was called at least twice (initial + 1 retry)
        assert len(provider.calls) >= 2

    def test_diversity_retry_exhausted_consumes_iteration(self, project_root: Path) -> None:
        """Tier-1 diversity retry: all retries exhausted → iteration budget consumed.

        When every retry still produces a too-similar config, the iteration is
        consumed (event written with retries_exhausted=True) and n_committed stays 0.
        """
        synthetic_prices = _make_synthetic_prices()
        fake_instance = _make_fake_provider(synthetic_prices)

        # Every call returns the same config → always fails diversity
        same_config_edit = AgentEdit(
            strategy_code=BASELINE_STRATEGY,
            config_yaml=STRATEGY_CONFIG,
            reasoning="Identical config every time",
            raw_response="{}",
        )

        provider = SequenceProvider(responses=[same_config_edit])

        with patch(
            "autobacktest.evaluator.evaluate.CachedDataProvider",
            return_value=fake_instance,
        ):
            result = run_optimization(
                program_path=project_root / "program.md",
                strategy_name="toy",
                iterations=1,
                provider=provider,
                run_dir=project_root / "runs",
                strategies_dir=project_root / "strategies",
                configs_dir=project_root / "configs",
                target_metric=TargetMetric.SHARPE,
                repo_path=project_root,
                start_date="2013-01-01",
                end_date="2025-01-01",
            )

        # No commit — all retries failed diversity
        assert result.n_committed == 0

        # Event must record retries_exhausted
        events_path = project_root / "runs" / result.run_id / "events.jsonl"
        events = [json.loads(ln) for ln in events_path.read_text().strip().split("\n") if ln]
        # Find the diversity-rejected event
        diversity_events = [e for e in events if e.get("diversity", {}).get("tier") == "config"]
        assert len(diversity_events) >= 1
        assert diversity_events[0]["diversity"]["retries_exhausted"] is True
        assert diversity_events[0]["diversity"]["passed"] is False

        # Provider was called MAX_DIVERSITY_RETRIES + 1 times for the iteration
        # (1 initial + MAX_DIVERSITY_RETRIES retries = 3 total per iteration)
        from autobacktest.orchestrator import MAX_DIVERSITY_RETRIES

        assert len(provider.calls) >= MAX_DIVERSITY_RETRIES + 1
