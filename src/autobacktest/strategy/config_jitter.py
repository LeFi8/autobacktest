"""Configuration jittering system for AutoBacktest.

Determinedly and deterministically mutates numeric configuration parameters
to satisfy the config diversity gate without discarding candidates.
"""

from __future__ import annotations

import copy
import logging
import random
from typing import Any

import numpy as np
import yaml

from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.diversity import (
    KNOWN_RANGES,
    ConfigFingerprint,
    _align_numeric_vectors,
    _cosine_similarity,
    _jaccard,
    _parse_config_to_flat,
    extract_config_fingerprint,
)

logger = logging.getLogger(__name__)

# Bounded root parameters constraints dynamically derived from StrategyConfig pydantic model
ROOT_CONSTRAINTS: dict[str, tuple[float | None, float | None]] = {}

for name, field in StrategyConfig.model_fields.items():
    if name == "params":
        continue
    min_val, max_val = None, None
    for m in getattr(field, "metadata", []):
        ge = getattr(m, "ge", None)
        gt = getattr(m, "gt", None)
        le = getattr(m, "le", None)
        lt = getattr(m, "lt", None)
        if ge is not None:
            min_val = float(ge)
        elif gt is not None:
            # gt represented as slightly greater than gt value
            min_val = float(gt) + 0.0001
        if le is not None:
            max_val = float(le)
        elif lt is not None:
            # lt represented as slightly less than lt value
            max_val = float(lt) - 0.0001
    if min_val is not None or max_val is not None:
        ROOT_CONSTRAINTS[name] = (min_val, max_val)


def get_param_name(path: tuple[Any, ...]) -> str:
    """Extract parameter name from config dictionary path."""
    for part in reversed(path):
        if isinstance(part, str):
            return part
    return ""


def set_at_path(data: dict[str, Any], path: tuple[Any, ...], val: Any) -> None:
    """Set value in deep config dictionary using path tuple."""
    curr: Any = data
    for p in path[:-1]:
        curr = curr[p]
    curr[path[-1]] = val


def compute_max_similarity(
    cf_candidate: ConfigFingerprint,
    cf_history: list[ConfigFingerprint],
) -> float:
    """Compute maximum similarity against pre-parsed historical config fingerprints."""
    if not cf_history:
        return 0.0

    all_fingerprints = [cf_candidate, *cf_history]
    global_min: dict[str, float] = {}
    global_max: dict[str, float] = {}
    for fp in all_fingerprints:
        for k, v in fp.numeric_params.items():
            if k in KNOWN_RANGES:
                continue
            if k not in global_min or v < global_min[k]:
                global_min[k] = v
            if k not in global_max or v > global_max[k]:
                global_max[k] = v

    max_sim = 0.0
    for cf_h in cf_history:
        va, vb = _align_numeric_vectors(
            cf_candidate.numeric_params,
            cf_h.numeric_params,
            global_min=global_min,
            global_max=global_max,
        )
        num_sim = _cosine_similarity(va, vb)

        all_set_keys = sorted(set(cf_candidate.set_fields) | set(cf_h.set_fields))
        jaccards = [
            _jaccard(
                cf_candidate.set_fields.get(k, set()),
                cf_h.set_fields.get(k, set()),
            )
            for k in all_set_keys
        ]
        set_sim = float(np.mean(jaccards)) if jaccards else 0.5

        sim = 0.7 * num_sim + 0.3 * set_sim
        if sim > max_sim:
            max_sim = sim

    return max_sim


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

    # Pre-parse historical config fingerprints once to optimize performance
    historical_fps = [extract_config_fingerprint(h) for h in tried_configs]

    # Map flat parameters to original paths
    flat = _parse_config_to_flat(config_yaml)
    all_paths: list[tuple[tuple[Any, ...], int | float]] = []
    for k, v in flat.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if k in orig_dict:
                all_paths.append(((k,), v))
            elif "params" in orig_dict and isinstance(orig_dict["params"], dict) and k in orig_dict["params"]:
                all_paths.append((("params", k), v))

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

            scale = max(abs(v) * param_rel_step, param_rel_step)
            change = rng.uniform(-scale, scale)
            v_new = v + change

            is_int = isinstance(v, int)
            if is_int:
                v_new = round(v_new)

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

            # Sign-preservation
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

            # Force adjustment if no change occurred, respecting direction and bounds
            if v_new == v:
                direction = rng.choice([1, -1])
                if is_int and v == 1:
                    direction = 1
                elif is_int and v == -1:
                    direction = -1
                elif min_val is not None and v <= min_val:
                    direction = 1
                elif max_val is not None and v >= max_val:
                    direction = -1

                v_new = v + (direction if is_int else (scale * direction))

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
        cf_candidate = extract_config_fingerprint(mutated_yaml)
        sim = compute_max_similarity(cf_candidate, historical_fps)
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
