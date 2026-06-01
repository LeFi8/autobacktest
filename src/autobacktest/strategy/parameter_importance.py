"""Spearman rank correlation for parameter importance tracking.

Computes the rank correlation between each numeric config parameter and the
target metric across optimization attempts, identifying which parameters most
strongly influence performance.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import yaml
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


def _extract_numeric_params(flat: dict[str, Any]) -> dict[str, float]:
    """Extract numeric-scalar values from a flattened config dict.

    Skips non-numeric types (lists, strings, booleans, None).
    Nested ``params`` are expected to already be merged at the top level.
    """
    params: dict[str, float] = {}
    for key, value in flat.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            params[key] = float(value)
    return params


def _parse_config_yaml(raw: str) -> dict[str, Any]:
    """Parse a YAML config string and flatten nested ``params`` into the top level."""
    try:
        data = yaml.safe_load(raw)
    except yaml.YAMLError:
        return {}

    if not isinstance(data, dict):
        return {}

    flat: dict[str, Any] = dict(data)
    params_sub = flat.pop("params", {})
    if isinstance(params_sub, dict):
        for k, v in params_sub.items():
            if k not in flat:
                flat[k] = v
    return flat


def compute_parameter_importance(
    config_yamls: list[str],
    metric_values: list[float],
    min_attempts: int = 6,
    p_threshold: float = 0.20,
) -> dict[str, dict[str, float]]:
    """Compute Spearman rank correlation for each numeric config parameter.

    Only parameters with at least ``min_attempts`` non-NaN observations and at
    least 3 distinct values are considered.  Results with ``p < p_threshold``
    are flagged as ``significant``.

    Args:
        config_yamls: Raw YAML config strings for all attempts (chronological).
        metric_values: Corresponding target metric values (same order).
        min_attempts: Minimum number of valid observations required.
        p_threshold: P-value threshold for significance.

    Returns:
        Dict mapping parameter name -> ``{"rho", "p_value", "n", "significant"}``.
        Empty dict when there are fewer attempts than ``min_attempts`` overall
        or no numeric parameters were found.
    """
    if len(config_yamls) < min_attempts:
        return {}

    all_params: list[dict[str, float]] = []
    for raw in config_yamls:
        flat = _parse_config_yaml(raw)
        all_params.append(_extract_numeric_params(flat))

    param_names: set[str] = set()
    for p in all_params:
        param_names.update(p.keys())

    results: dict[str, dict[str, float]] = {}
    for name in sorted(param_names):
        values: list[float] = []
        metrics: list[float] = []
        for i in range(len(config_yamls)):
            if name in all_params[i]:
                v = all_params[i][name]
                if not np.isnan(v) and not np.isinf(v):
                    values.append(v)
                    metrics.append(metric_values[i])

        n = len(values)
        if n < min_attempts or len(set(values)) < 3:
            continue

        if len(set(metrics)) < 2:
            continue

        rho, p_value = spearmanr(values, metrics)
        if np.isnan(rho):
            continue

        results[name] = {
            "rho": round(float(rho), 4),
            "p_value": round(float(p_value), 4),
            "n": n,
            "significant": bool(p_value < p_threshold),
        }

    return results


def format_importance_lessons(
    importance: dict[str, dict[str, float]],
) -> str:
    """Format parameter importance results as lessons markdown.

    Returns an empty string when no significant parameters exist.
    """
    significant = {k: v for k, v in importance.items() if v["significant"]}
    if not significant:
        return ""

    lines = [
        "### Parameter Importance",
        "",
        "| Parameter | Rho | P-value | N |",
        "|---|---|---|---|",
    ]
    for name, data in sorted(significant.items(), key=lambda x: -abs(x[1]["rho"])):
        lines.append(f"| {name} | {data['rho']:.3f} | {data['p_value']:.3f} | {data['n']} |")
    lines.append("")

    return "\n".join(lines)
