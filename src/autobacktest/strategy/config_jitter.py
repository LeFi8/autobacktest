"""Configuration jittering system for AutoBacktest.

Determinedly and deterministically mutates numeric configuration parameters
to satisfy the config diversity gate without discarding candidates.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any

import yaml

from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.diversity import KNOWN_RANGES, max_config_similarity

logger = logging.getLogger(__name__)

# Bounded root parameters constraints (from StrategyConfig validation)
ROOT_CONSTRAINTS: dict[str, tuple[float | None, float | None]] = {
    "momentum_lookback": (1.0, None),
    "max_drawdown_limit": (0.0, 1.0),
    "turnover_limit": (0.0001, 10.0),  # gt=0.0 represented as >0.0001 lower bound
    "borrow_cost_bps": (0.0, None),
    "cscv_blocks": (4.0, None),
    "min_improvement": (0.0, None),
    "select_min_return_ratio": (0.0, 1.0),
}


def get_param_name(path: tuple[Any, ...]) -> str:
    """Extract parameter name from config dictionary path."""
    for part in reversed(path):
        if isinstance(part, str):
            return part
    return ""


def find_numeric_leaves(data: Any, path: tuple[Any, ...] = ()) -> list[tuple[tuple[Any, ...], int | float]]:
    """Traverse nested structure and retrieve paths to numeric leaves (excl. bools)."""
    leaves: list[tuple[tuple[Any, ...], int | float]] = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                leaves.append(((*path, k), v))
            elif isinstance(v, (dict, list)):
                leaves.extend(find_numeric_leaves(v, (*path, k)))
    elif isinstance(data, list):
        for idx, v in enumerate(data):
            if isinstance(v, bool):
                continue
            if isinstance(v, (int, float)):
                leaves.append(((*path, idx), v))
            elif isinstance(v, (dict, list)):
                leaves.extend(find_numeric_leaves(v, (*path, idx)))
    return leaves


def set_at_path(data: dict[str, Any], path: tuple[Any, ...], val: Any) -> None:
    """Set value in deep config dictionary using path tuple."""
    curr: Any = data
    for p in path[:-1]:
        curr = curr[p]
    curr[path[-1]] = val


def jitter_config(
    config_yaml: str,
    tried_configs: list[str],
    threshold: float,
    *,
    seed: int,
    max_attempts: int = 12,
    rel_step: float = 0.15,
    importance: dict[str, Any] | None = None,
) -> tuple[str | None, dict[str, Any]]:
    """Mutate config's numeric values until its similarity to tried_configs is below threshold.

    Args:
        config_yaml: Proposed strategy configuration YAML string.
        tried_configs: Historical / batch configurations checked against.
        threshold: Config similarity threshold.
        seed: Random seed for deterministic perturbation.
        max_attempts: Maximum attempts to try mutation.
        rel_step: Base relative perturbation step size.
        importance: Dict mapping param name -> correlation stats.

    Returns:
        tuple[str | None, dict]: (new_config_yaml, metadata_dict)
    """
    rng = random.Random(seed)

    try:
        orig_dict = yaml.safe_load(config_yaml)
    except Exception as e:
        logger.warning("Failed to parse config YAML for jittering: %s", e)
        return None, {"jitter_applied": False, "attempts": 0, "final_similarity": 1.0, "changed_params": {}}

    if not isinstance(orig_dict, dict):
        return None, {"jitter_applied": False, "attempts": 0, "final_similarity": 1.0, "changed_params": {}}

    # Collect mutable paths:
    # 1. Bounded root numeric fields
    root_paths: list[tuple[tuple[Any, ...], int | float]] = []
    for k in ROOT_CONSTRAINTS:
        if k in orig_dict and isinstance(orig_dict[k], (int, float)) and not isinstance(orig_dict[k], bool):
            root_paths.append(((k,), orig_dict[k]))

    # 2. Leaves under "params" key
    params_dict = orig_dict.get("params", {})
    params_paths: list[tuple[tuple[Any, ...], int | float]] = []
    if isinstance(params_dict, dict):
        params_paths = find_numeric_leaves(params_dict, ("params",))

    all_paths = root_paths + params_paths
    if not all_paths:
        return None, {"jitter_applied": False, "attempts": 0, "final_similarity": 1.0, "changed_params": {}}

    last_similarity = 1.0
    for a in range(max_attempts):
        rel_step_attempt = rel_step * (1.0 + a * 0.5)

        mutated_dict = copy.deepcopy(orig_dict)
        changed_params: dict[str, dict[str, float]] = {}

        for path, v in all_paths:
            name = get_param_name(path)

            param_rel_step = rel_step_attempt
            if importance and name in importance:
                rho = importance[name].get("rho", 0.0)
                param_rel_step *= 1.0 + abs(rho)

            # Perturbation scale: sign-preserving relative perturbation for unknown,
            # floor at param_rel_step so zeroes move
            scale = max(abs(v) * param_rel_step, param_rel_step)
            change = rng.uniform(-scale, scale)
            v_new = v + change

            is_int = isinstance(v, int)
            if is_int:
                v_new = round(v_new)

            # Sign-preservation
            if v > 0:
                v_new = max(1 if is_int else 1e-6, v_new)
            elif v < 0:
                v_new = min(-1 if is_int else -1e-6, v_new)

            # Retrieve bounds from KNOWN_RANGES or ROOT_CONSTRAINTS
            min_val: float | None = None
            max_val: float | None = None
            if name in KNOWN_RANGES:
                min_val, max_val = KNOWN_RANGES[name]
            if path[0] in ROOT_CONSTRAINTS:
                rmin, rmax = ROOT_CONSTRAINTS[path[0]]
                if rmin is not None:
                    min_val = rmin if min_val is None else max(min_val, rmin)
                if rmax is not None:
                    max_val = rmax if max_val is None else min(max_val, rmax)

            if min_val is not None:
                v_new = max(min_val, v_new)
            if max_val is not None:
                v_new = min(max_val, v_new)

            if is_int:
                v_new = round(v_new)

            # Force adjustment if no change occurred
            if v_new == v:
                direction = rng.choice([1, -1])
                v_new = v + (1 if is_int else (scale * direction))

                # Re-apply sign-preservation & bounds
                if v > 0:
                    v_new = max(1 if is_int else 1e-6, v_new)
                elif v < 0:
                    v_new = min(-1 if is_int else -1e-6, v_new)

                if min_val is not None:
                    v_new = max(min_val, v_new)
                if max_val is not None:
                    v_new = min(max_val, v_new)

                if is_int:
                    v_new = round(v_new)

            set_at_path(mutated_dict, path, v_new)
            if v_new != v:
                changed_params[name] = {"old": float(v), "new": float(v_new)}

        # Validate against schema
        try:
            StrategyConfig.model_validate(mutated_dict)
        except Exception:
            continue

        mutated_yaml = yaml.safe_dump(mutated_dict, sort_keys=False)
        sim = max_config_similarity(mutated_yaml, tried_configs)
        last_similarity = sim

        if sim < threshold:
            meta = {
                "jitter_applied": True,
                "attempts": a + 1,
                "final_similarity": float(sim),
                "changed_params": changed_params,
            }
            return mutated_yaml, meta

    return None, {
        "jitter_applied": False,
        "attempts": max_attempts,
        "final_similarity": float(last_similarity),
        "changed_params": {},
    }
