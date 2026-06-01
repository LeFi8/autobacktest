"""Unit tests for Spearman-based parameter importance tracking."""

from __future__ import annotations

import pytest
import yaml

from autobacktest.strategy.parameter_importance import (
    _extract_numeric_params,
    _parse_config_yaml,
    compute_parameter_importance,
    format_importance_lessons,
)


class TestExtractNumericParams:
    def test_skips_non_numeric(self) -> None:
        flat = {"lookback": 20, "name": "test", "tickers": ["A", "B"], "active": True}
        result = _extract_numeric_params(flat)
        assert result == {"lookback": 20.0}

    def test_empty(self) -> None:
        assert _extract_numeric_params({}) == {}

    def test_handles_nan_inf(self) -> None:
        flat = {"a": float("nan"), "b": float("inf"), "c": 42}
        result = _extract_numeric_params(flat)
        # nan/inf are still floats, so they pass the type check
        assert "a" in result
        assert "b" in result
        assert result["c"] == 42.0


class TestParseConfigYaml:
    def test_flat_config(self) -> None:
        raw = "lookback: 20\ntop_x: 5\n"
        result = _parse_config_yaml(raw)
        assert result["lookback"] == 20
        assert result["top_x"] == 5

    def test_with_params(self) -> None:
        raw = """
        universe: [SPY]
        params:
          lookback: 20
          threshold: 0.5
        """
        result = _parse_config_yaml(raw)
        assert result["universe"] == ["SPY"]
        assert result["lookback"] == 20
        assert result["threshold"] == 0.5

    def test_invalid_yaml(self) -> None:
        assert _parse_config_yaml("{{{") == {}

    def test_empty_string(self) -> None:
        assert _parse_config_yaml("") == {}

    def test_params_does_not_override_top_level(self) -> None:
        raw = """
        lookback: 10
        params:
          lookback: 999
        """
        result = _parse_config_yaml(raw)
        assert result["lookback"] == 10  # top-level wins


class TestComputeParameterImportance:
    def test_returns_empty_when_too_few_attempts(self) -> None:
        configs = ["lookback: 10\n", "lookback: 20\n"]
        metrics = [1.0, 1.5]
        result = compute_parameter_importance(configs, metrics, min_attempts=5)
        assert result == {}

    def test_monotonic_positive_correlation(self) -> None:
        configs = [yaml.dump({"lookback": v, "params": {"threshold": 0.5}}) for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        metrics = [0.1 * v for v in range(1, 11)]  # strictly increasing
        result = compute_parameter_importance(configs, metrics, min_attempts=6, p_threshold=0.20)
        assert "lookback" in result
        assert result["lookback"]["rho"] == pytest.approx(1.0, abs=0.01)
        assert result["lookback"]["significant"] is True
        assert result["lookback"]["n"] == 10

    def test_monotonic_negative_correlation(self) -> None:
        configs = [yaml.dump({"lookback": v, "params": {"threshold": 0.5}}) for v in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        metrics = [10 - 0.1 * v for v in range(1, 11)]  # strictly decreasing
        result = compute_parameter_importance(configs, metrics, min_attempts=6, p_threshold=0.20)
        assert "lookback" in result
        assert result["lookback"]["rho"] == pytest.approx(-1.0, abs=0.01)
        assert result["lookback"]["significant"] is True

    def test_no_correlation(self) -> None:
        configs = [yaml.dump({"lookback": v, "params": {"threshold": 0.5}}) for v in range(1, 11)]
        metrics = [0.5] * 10  # constant
        result = compute_parameter_importance(configs, metrics, min_attempts=6, p_threshold=0.20)
        # With constant metric, spearmanr returns nan rho → skipped
        assert "lookback" not in result

    def test_skips_parameter_with_fewer_than_min_attempts(self) -> None:
        configs = [yaml.dump({"lookback": 10, "params": {"rare_param": v}}) for v in range(1, 4)]
        # rare_param only appears 3 times, but we have 10 configs
        configs += [yaml.dump({"lookback": v}) for v in range(4, 11)]
        metrics = [0.1 * v for v in range(1, 11)]
        result = compute_parameter_importance(configs, metrics, min_attempts=6, p_threshold=0.20)
        # lookback should be present (10 obs), rare_param not (3 < 6)
        assert "lookback" in result
        assert "rare_param" not in result

    def test_handles_params_key(self) -> None:
        configs = [yaml.dump({"params": {"momentum_window": v}}) for v in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]]
        metrics = [0.01 * v for v in range(1, 11)]
        result = compute_parameter_importance(configs, metrics, min_attempts=6, p_threshold=0.20)
        assert "momentum_window" in result
        assert result["momentum_window"]["n"] == 10
        assert result["momentum_window"]["significant"] is True

    def test_significance_threshold_filters_weak_correlations(self) -> None:
        # Generate configs with very weak relationship
        configs = [yaml.dump({"lookback": v % 3 + 1}) for v in range(10)]
        metrics = [v * 0.01 for v in range(10)]
        result = compute_parameter_importance(
            configs,
            metrics,
            min_attempts=3,
            p_threshold=0.001,  # very strict
        )
        # With only 3 distinct values and 10 obs, p-value will likely be > 0.001
        if "lookback" in result:
            assert result["lookback"]["significant"] is False


class TestFormatImportanceLessons:
    def test_empty_when_no_importance(self) -> None:
        assert format_importance_lessons({}) == ""

    def test_empty_when_no_significant(self) -> None:
        data = {"p1": {"rho": 0.5, "p_value": 0.5, "n": 10, "significant": False}}
        assert format_importance_lessons(data) == ""

    def test_renders_table(self) -> None:
        data = {
            "lookback": {"rho": 0.85, "p_value": 0.01, "n": 10, "significant": True},
            "threshold": {"rho": -0.72, "p_value": 0.03, "n": 8, "significant": True},
        }
        result = format_importance_lessons(data)
        assert "### Parameter Importance" in result
        assert "lookback" in result
        assert "threshold" in result
        assert "0.850" in result
        assert "-0.720" in result
        assert "0.010" in result
        assert "0.030" in result
        assert "10" in result
        assert "8" in result

    def test_sorts_by_abs_rho_descending(self) -> None:
        data = {
            "p_low": {"rho": 0.3, "p_value": 0.01, "n": 10, "significant": True},
            "p_high": {"rho": 0.9, "p_value": 0.01, "n": 10, "significant": True},
        }
        result = format_importance_lessons(data)
        a_pos = result.index("p_low")
        b_pos = result.index("p_high")
        assert b_pos < a_pos  # p_high (rho=0.9) comes before p_low (rho=0.3)
