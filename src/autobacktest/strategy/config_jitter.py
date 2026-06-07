"""Configuration jittering system for AutoBacktest.

Determinedly and deterministically mutates numeric configuration parameters
to satisfy the config diversity gate without discarding candidates.
"""

from __future__ import annotations

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


# Keys that must not be perturbed during jitter (system / gate parameters)
_EXCLUDE_JITTER_KEYS: frozenset[str] = frozenset(
    {
        "pbo_limit",
        "cscv_embargo_days",
        "adaptive_slippage",
        "slippage_vol_window",
        "slippage_vol_cap",
        "mc_bootstrap_method",
        "cscv_blocks",
        "borrow_cost_bps",
        "min_improvement",
        "select_min_return_ratio",
        "holdout_min_improvement",
        "max_drawdown_limit",
        "turnover_limit",
        "dsr_floor",
    }
)


def _collect_jitter_paths(
    orig_dict: dict[str, Any],
    flat: dict[str, Any],
) -> list[tuple[tuple[Any, ...], int | float]]:
    """Build a list of ``(path, value)`` pairs for all jitterable numeric parameters.

    Only root-level and ``params``-nested numeric keys that are not in
    ``_EXCLUDE_JITTER_KEYS`` are included.

    Args:
        orig_dict: Parsed config dictionary.
        flat: Flat key→value mapping produced by ``_parse_config_to_flat``.

    Returns:
        list of ``(path_tuple, original_value)`` pairs eligible for mutation.
    """
    all_paths: list[tuple[tuple[Any, ...], int | float]] = []
    for k, v in flat.items():
        if k in _EXCLUDE_JITTER_KEYS:
            continue
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            continue
        if k in orig_dict:
            all_paths.append(((k,), v))
        elif "params" in orig_dict and isinstance(orig_dict["params"], dict) and k in orig_dict["params"]:
            all_paths.append((("params", k), v))
    return all_paths


def _get_param_bounds(name: str, path: tuple[Any, ...]) -> tuple[float | None, float | None]:
    """Return ``(min_val, max_val)`` for *name* from known ranges and root schema constraints.

    Args:
        name: Flattened parameter name.
        path: Path tuple pointing to the parameter in the config dict.

    Returns:
        tuple: ``(min_val, max_val)`` — either bound may be ``None`` if unconstrained.
    """
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
    return min_val, max_val


def _apply_sign_and_bounds(
    v_new: float,
    v: int | float,
    is_int: bool,
    min_val: float | None,
    max_val: float | None,
) -> int | float:
    """Clamp *v_new* to preserve the sign of *v* and respect numeric bounds.

    Args:
        v_new: The proposed new value after perturbation.
        v: The original value (used for sign-preservation direction).
        is_int: Whether the parameter is an integer type.
        min_val: Lower bound, or ``None`` if unconstrained.
        max_val: Upper bound, or ``None`` if unconstrained.

    Returns:
        int | float: The clamped value, cast to ``int`` when *is_int* is True.
    """
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
    return v_new


def _force_change(
    v: int | float,
    is_int: bool,
    min_val: float | None,
    max_val: float | None,
    rng: random.Random,
    scale: float,
) -> int | float:
    """Compute a guaranteed non-zero mutation of *v* when the perturbation rounded back.

    Chooses a direction (±1) that respects current bounds, applies a step,
    then re-clamps the result via :func:`_apply_sign_and_bounds`.

    Args:
        v: The original value.
        is_int: Whether the parameter is an integer type.
        min_val: Lower bound, or ``None`` if unconstrained.
        max_val: Upper bound, or ``None`` if unconstrained.
        rng: Random number generator for direction selection.
        scale: Magnitude of the forced step.

    Returns:
        int | float: A value that differs from *v* after clamping (best-effort).
    """
    direction = rng.choice([1, -1])
    if is_int and v == 1:
        direction = 1
    elif is_int and v == -1:
        direction = -1
    elif min_val is not None and v <= min_val:
        direction = 1
    elif max_val is not None and v >= max_val:
        direction = -1
    step = direction if is_int else scale * direction
    return _apply_sign_and_bounds(v + step, v, is_int, min_val, max_val)


def _attempt_mutation(
    orig_dict: dict[str, Any],
    all_paths: list[tuple[tuple[Any, ...], int | float]],
    rng: random.Random,
    rel_step_attempt: float,
    importance: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, dict[str, float]]] | None:
    """Mutate every numeric parameter in *orig_dict* by one jitter step.

    Applies sign-preservation, numeric bounds, and a forced-change fallback.
    Returns ``None`` when the mutated config fails Pydantic schema validation.

    Args:
        orig_dict: Original parsed config dictionary (not modified in place).
        all_paths: ``(path, original_value)`` pairs from :func:`_collect_jitter_paths`.
        rng: Seeded random number generator.
        rel_step_attempt: Relative perturbation magnitude for this attempt.
        importance: Optional dict mapping param name → correlation stats for adaptive step sizing.

    Returns:
        tuple ``(mutated_dict, changed_params)`` on success, or ``None`` on schema failure.
    """
    import copy

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
        v_new: int | float = v + change
        is_int = isinstance(v, int)

        min_val, max_val = _get_param_bounds(name, path)
        v_new = _apply_sign_and_bounds(v_new, v, is_int, min_val, max_val)

        if v_new == v:
            v_new = _force_change(v, is_int, min_val, max_val, rng, scale)

        set_at_path(mutated_dict, path, v_new)
        if v_new != v:
            changed_params[name] = {"old": float(v), "new": float(v_new)}

    try:
        StrategyConfig.model_validate(mutated_dict)
    except Exception:
        return None

    return mutated_dict, changed_params


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
    _fail_meta = {"jitter_applied": False, "attempts": 0, "final_similarity": 1.0, "changed_params": {}}
    rng = random.Random(seed)

    try:
        orig_dict = yaml.safe_load(config_yaml)
    except Exception as e:
        logger.warning("Failed to parse config YAML for jittering: %s", e)
        return None, _fail_meta

    if not isinstance(orig_dict, dict):
        return None, _fail_meta

    historical_fps = [extract_config_fingerprint(h) for h in tried_configs]
    flat = _parse_config_to_flat(config_yaml)
    all_paths = _collect_jitter_paths(orig_dict, flat)

    if not all_paths:
        return None, _fail_meta

    last_similarity = 1.0
    for a in range(max_attempts):
        rel_step_attempt = rel_step * (1.0 + a * 0.5)
        attempt = _attempt_mutation(orig_dict, all_paths, rng, rel_step_attempt, importance)
        if attempt is None:
            continue
        mutated_dict, changed_params = attempt
        mutated_yaml = yaml.safe_dump(mutated_dict, sort_keys=False)
        cf_candidate = extract_config_fingerprint(mutated_yaml)
        sim = compute_max_similarity(cf_candidate, historical_fps)
        last_similarity = sim
        if sim < threshold:
            return mutated_yaml, {
                "jitter_applied": True,
                "attempts": a + 1,
                "final_similarity": float(sim),
                "changed_params": changed_params,
            }

    return None, {
        "jitter_applied": False,
        "attempts": max_attempts,
        "final_similarity": float(last_similarity),
        "changed_params": {},
    }
