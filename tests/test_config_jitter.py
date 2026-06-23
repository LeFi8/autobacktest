import yaml

from autobacktest.strategy.config_jitter import jitter_config
from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.diversity import max_config_similarity

BASE_YAML = """
universe:
  - SPY
  - QQQ
benchmark: SPY
max_drawdown_limit: 0.20
turnover_limit: 2.0
require_dsr_non_degradation: true
enable_holdout_confirmation: true
holdout_min_improvement: 0.0
min_improvement: 0.0
select_min_return_ratio: 0.5
params:
  mom_window: 12
  canary_smoothing_window: 10
  other_param: 4.5
  negative_param: -2.0
  zero_param: 0.0
"""


def test_jitter_basic():
    tried = [BASE_YAML]
    # Similarity to tried should be 1.0 initially
    assert max_config_similarity(BASE_YAML, tried) > 0.999

    # Run jitter
    jittered_yaml, meta = jitter_config(BASE_YAML, tried, 0.95, seed=42)
    assert jittered_yaml is not None
    assert meta["jitter_applied"] is True
    assert meta["attempts"] > 0
    assert meta["final_similarity"] < 0.95

    # Check that it validates
    dict_val = yaml.safe_load(jittered_yaml)
    StrategyConfig.model_validate(dict_val)

    # Check int types are preserved
    assert isinstance(dict_val["params"]["mom_window"], int)
    assert isinstance(dict_val["params"]["canary_smoothing_window"], int)
    assert isinstance(dict_val["params"]["other_param"], float)

    # Check sign preservation
    assert dict_val["params"]["mom_window"] > 0
    assert dict_val["params"]["canary_smoothing_window"] > 0
    assert dict_val["params"]["other_param"] > 0
    assert dict_val["params"]["negative_param"] < 0
    assert dict_val["params"]["zero_param"] != 0.0  # moved from zero


def test_jitter_bounds():
    yaml_with_bounds = """
universe: [SPY]
benchmark: SPY
max_drawdown_limit: 0.20
turnover_limit: 2.0
params:
  custom_limit: 0.45
"""
    tried = [yaml_with_bounds]
    jittered_yaml, _ = jitter_config(yaml_with_bounds, tried, 0.90, seed=123)
    assert jittered_yaml is not None
    dict_val = yaml.safe_load(jittered_yaml)
    # max_drawdown_limit is root
    assert 0.0 <= dict_val["max_drawdown_limit"] <= 1.0


def test_jitter_deterministic():
    tried = [BASE_YAML]
    res1, meta1 = jitter_config(BASE_YAML, tried, 0.95, seed=42)
    res2, meta2 = jitter_config(BASE_YAML, tried, 0.95, seed=42)
    res3, _ = jitter_config(BASE_YAML, tried, 0.95, seed=43)

    assert res1 == res2
    assert meta1 == meta2
    assert res1 != res3


def test_jitter_unreachable_threshold():
    tried = [BASE_YAML]
    # With a threshold of 0.0, it should fail to find any valid jitter and return None
    res, meta = jitter_config(BASE_YAML, tried, 0.0, seed=42, max_attempts=5)
    assert res is None
    assert meta["jitter_applied"] is False
    assert meta["attempts"] == 5


def test_jitter_nested_lists():
    yaml_with_nested = """
universe: [SPY]
benchmark: SPY
max_drawdown_limit: 0.20
turnover_limit: 2.0
params:
  weights: [0.1, 0.2, 0.3]
  scalar_param: 1.5
"""
    tried = [yaml_with_nested]
    # scalar_param is mutated, weights list should be completely untouched
    jittered, _ = jitter_config(yaml_with_nested, tried, 0.95, seed=42)
    assert jittered is not None
    dict_val = yaml.safe_load(jittered)
    assert dict_val["params"]["weights"] == [0.1, 0.2, 0.3]
    assert dict_val["params"]["scalar_param"] != 1.5


def test_jitter_boundary_force_adjustment():
    # Test momentum_lookback pinned at its lower bound (1)
    yaml_at_bound = """
universe: [SPY]
benchmark: SPY
momentum_lookback: 1
max_drawdown_limit: 0.20
turnover_limit: 2.0
params:
  mock_param: 10.0
"""
    tried = [yaml_at_bound]
    jittered, _meta = jitter_config(yaml_at_bound, tried, 0.95, seed=42)
    assert jittered is not None
    dict_val = yaml.safe_load(jittered)
    # Since momentum_lookback is ge=1, it must be forced to move inward (i.e. to 2 or more)
    assert dict_val["momentum_lookback"] >= 2


def test_orchestrator_apply_jitter_exhaustion() -> None:
    """Verifies that Orchestrator._apply_jitter writes a 'jitter_exhausted_proceeding' event on exhaustion."""
    from pathlib import Path
    from unittest import mock
    from unittest.mock import MagicMock

    from autobacktest.llm.base import AgentEdit
    from autobacktest.orchestrator import _OptimizationState

    # Create a mock orchestrator state
    orchestrator = MagicMock(spec=_OptimizationState)
    # Set the attributes used by _apply_jitter
    orchestrator.strategies_dir = Path("dummy_strategies")
    orchestrator.configs_dir = Path("dummy_configs")
    orchestrator.last_importance = None

    # Mock event log
    mock_event_log = MagicMock()
    orchestrator.event_log = mock_event_log

    # Bind the actual method to the mock instance
    orchestrator._apply_jitter = _OptimizationState._apply_jitter.__get__(orchestrator, _OptimizationState)

    # Let's mock settings for diversity/jitter
    with mock.patch("autobacktest.strategy.config_jitter.jitter_config") as mock_jitter:
        mock_jitter.return_value = (None, {"final_similarity": 0.98, "jitter_applied": False, "attempts": 10})

        ev = {
            "config_yaml": BASE_YAML,
            "edit": MagicMock(spec=AgentEdit),
        }

        orchestrator._apply_jitter(k=1, i=0, ev=ev, all_tried=[BASE_YAML])

        # Verify event log call
        mock_event_log.write.assert_called_once_with(
            {
                "event": "jitter_exhausted_proceeding",
                "iteration": 1,
                "candidate_idx": 0,
                "similarity": 0.98,
            }
        )

        # Verify ev was updated with correct status flags
        assert ev["config_yaml"] == BASE_YAML
        assert ev.get("jitter_applied") is False
        assert ev.get("jitter_attempted") is True
