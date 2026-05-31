"""Config fingerprint extraction, similarity scoring, and returns correlation checks.

Provides the diversity gate functions used by the orchestrator to prevent
the LLM optimizer from proposing strategy variants that are too similar
to previously attempted or committed variants.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import yaml

# ---------------------------------------------------------------------------
# Known parameter ranges for min-max normalization
# ---------------------------------------------------------------------------
KNOWN_RANGES: dict[str, tuple[float, float]] = {
    "momentum_lookback": (1.0, 24.0),
    "top_x": (1.0, 10.0),
    "canary_smoothing_window": (1.0, 36.0),
    "canary_hysteresis": (0.0, 0.1),
    "min_canary_period": (1.0, 36.0),
    "offensive_rebalance_months": (1.0, 12.0),
    "max_drawdown_limit": (0.0, 0.5),
    "turnover_limit": (0.1, 10.0),
}


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------
class ConfigFingerprint:
    """Normalised representation of a strategy config for similarity comparison.

    Attributes:
        numeric_params: Dict of param_name → raw numeric value (min-max
            normalised later during pairwise comparison).
        set_fields: Dict of field_name → set of string members (tickers, etc.).
    """

    def __init__(
        self,
        numeric_params: dict[str, float] | None = None,
        set_fields: dict[str, set[str]] | None = None,
    ) -> None:
        self.numeric_params = numeric_params or {}
        self.set_fields = set_fields or {}

    def __repr__(self) -> str:
        return f"ConfigFingerprint(numeric_keys={list(self.numeric_params)}, set_keys={list(self.set_fields)})"


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------
def _parse_config_to_flat(config_yaml: str) -> dict[str, Any]:
    """Parse a YAML config string and flatten nested ``params`` into the root.

    Args:
        config_yaml: Raw YAML string.

    Returns:
        Flat dictionary with all top-level keys plus any keys under ``params``.
    """
    if not config_yaml or not config_yaml.strip():
        return {}

    try:
        data = yaml.safe_load(config_yaml)
    except Exception:
        return {}

    if not isinstance(data, dict):
        return {}

    flat: dict[str, Any] = {}
    for k, v in data.items():
        if k == "params" and isinstance(v, dict):
            flat.update(v)
        else:
            flat[k] = v
    return flat


def _normalize(value: float, lo: float, hi: float) -> float:
    """Min-max normalise *value* into the closed interval [0, 1]."""
    if hi <= lo:
        return 0.5
    clipped = max(lo, min(value, hi))
    return (clipped - lo) / (hi - lo)


# ---------------------------------------------------------------------------
# Fingerprint extraction
# ---------------------------------------------------------------------------
def extract_config_fingerprint(config_yaml: str) -> ConfigFingerprint:
    """Parse *config_yaml* and produce a normalised fingerprint.

    Numeric leaf values (int / float) are normalised to [0, 1] via
    :data:`KNOWN_RANGES`.  Values whose keys are unknown are left raw and
    will be range-normalised during pairwise comparison (see
    :func:`_align_numeric_vectors`).

    List-of-string values are stored as sets for Jaccard comparison.

    Args:
        config_yaml: Raw YAML strategy configuration.

    Returns:
        A :class:`ConfigFingerprint` ready for similarity comparison.
    """
    flat = _parse_config_to_flat(config_yaml)

    numeric: dict[str, float] = {}
    sets: dict[str, set[str]] = {}

    for k, v in flat.items():
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            numeric[k] = float(v)

        elif isinstance(v, list) and all(isinstance(x, str) for x in v):
            sets[k] = set(v)

    return ConfigFingerprint(numeric_params=numeric, set_fields=sets)


# ---------------------------------------------------------------------------
# Pairwise similarity helpers
# ---------------------------------------------------------------------------
def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two equally-sized vectors.

    Returns 0.5 when both vectors are zero or empty.
    """
    if not a or not b:
        return 0.5
    if len(a) != len(b):
        # Pad the shorter vector with 0.5 (neutral)
        max_len = max(len(a), len(b))
        a = list(a) + [0.5] * (max_len - len(a))
        b = list(b) + [0.5] * (max_len - len(b))

    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)

    dot = np.dot(a_arr, b_arr)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)

    if norm_a == 0.0 and norm_b == 0.0:
        return 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0

    return float(dot / (norm_a * norm_b))


def _jaccard(a: set[str], b: set[str]) -> float:
    """Jaccard similarity coefficient for two sets.

    Returns 0.5 when both sets are empty.
    """
    if not a and not b:
        return 0.5
    union = a | b
    if not union:
        return 0.5
    return len(a & b) / len(union)


# ---------------------------------------------------------------------------
# Vector alignment for cross-fingerprint numeric comparison
# ---------------------------------------------------------------------------
def _align_numeric_vectors(
    a: dict[str, float],
    b: dict[str, float],
    global_min: dict[str, float] | None = None,
    global_max: dict[str, float] | None = None,
) -> tuple[list[float], list[float]]:
    """Align two numeric-param dicts into equal-length normalised vectors.

    Keys present in only one dict are filled with 0.5 (neutral similarity).
    Unknown keys (not in :data:`KNOWN_RANGES`) are min-max normalised
    using *global_min* / *global_max* if available, otherwise kept as-is.

    Returns:
        (va, vb) two lists of the same length.
    """
    all_keys = sorted(set(a) | set(b))
    va: list[float] = []
    vb: list[float] = []

    for key in all_keys:
        # Skip unknown params with degenerate global range (hi <= lo means no
        # discriminatory signal across the population — normalising both sides
        # to 0.5 would inflate cosine similarity toward 1.0).
        if (
            key not in KNOWN_RANGES
            and global_min is not None
            and global_max is not None
            and key in global_min
            and global_min[key] >= global_max[key]
        ):
            continue

        av = a.get(key)
        bv = b.get(key)

        a_norm: float | None = None
        b_norm: float | None = None

        for val, dst in [(av, "a"), (bv, "b")]:
            if val is None:
                continue
            if key in KNOWN_RANGES:
                lo, hi = KNOWN_RANGES[key]
                normd = _normalize(val, lo, hi)
            elif global_min is not None and global_max is not None and key in global_min:
                lo = global_min[key]
                hi = global_max[key]
                normd = _normalize(val, lo, hi)
            else:
                normd = val  # raw

            if dst == "a":
                a_norm = normd
            else:
                b_norm = normd

        va.append(a_norm if a_norm is not None else 0.5)
        vb.append(b_norm if b_norm is not None else 0.5)

    return va, vb


# ---------------------------------------------------------------------------
# Public API: config similarity
# ---------------------------------------------------------------------------
def config_similarity(a: ConfigFingerprint, b: ConfigFingerprint) -> float:
    """Weighted similarity between two config fingerprints.

    ``similarity = 0.7 x cosine(numeric) + 0.3 x mean(jaccard of set fields)``

    Returns a float in ``[0, 1]`` where 1.0 means structurally identical.
    """
    # Numeric similarity
    va, vb = _align_numeric_vectors(a.numeric_params, b.numeric_params)
    num_sim = _cosine_similarity(va, vb)

    # Set-field similarity
    all_set_keys = sorted(set(a.set_fields) | set(b.set_fields))
    jaccards: list[float] = []
    for key in all_set_keys:
        sa = a.set_fields.get(key, set())
        sb = b.set_fields.get(key, set())
        jaccards.append(_jaccard(sa, sb))

    set_sim = float(np.mean(jaccards)) if jaccards else 0.5

    return 0.7 * num_sim + 0.3 * set_sim


def max_config_similarity(
    candidate_yaml: str,
    historical_ymls: list[str],
) -> float:
    """Return the maximum config similarity between *candidate_yaml* and any config in *historical_ymls*.

    Args:
        candidate_yaml: The proposed strategy's config YAML.
        historical_ymls: Previously committed (or attempted) config YAML strings.

    Returns:
        Maximum similarity score in ``[0, 1]``.  Returns 0.0 when
        *historical_ymls* is empty.
    """
    if not historical_ymls:
        return 0.0

    cf_candidate = extract_config_fingerprint(candidate_yaml)

    # Build global min/max for unknown-param normalisation across all configs.
    all_fingerprints = [cf_candidate] + [extract_config_fingerprint(h) for h in historical_ymls]
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
    for h in historical_ymls:
        cf_h = extract_config_fingerprint(h)
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


# ---------------------------------------------------------------------------
# Public API: returns correlation
# ---------------------------------------------------------------------------
def check_returns_correlation(
    candidate_returns: pd.Series,
    historical_returns_matrix: pd.DataFrame,
    threshold: float = 0.90,
    min_overlap_days: int = 60,
) -> tuple[bool, float]:
    """Check whether *candidate_returns* is too highly correlated with any column in *historical_returns_matrix*.

    Args:
        candidate_returns: Daily net returns of the current strategy candidate.
        historical_returns_matrix: Each column is the return series of a
            past attempt (aligned by date).
        threshold: Maximum allowed Pearson correlation (default 0.90).
        min_overlap_days: Minimum overlapping trading days required to
            compute a meaningful correlation (default 60).

    Returns:
        ``(passed, max_correlation)`` — *passed* is ``True`` when the
        candidate is sufficiently different from all historical series
        (i.e. every pairwise correlation is ≤ *threshold*).
    """
    if candidate_returns.empty or historical_returns_matrix.empty:
        return True, 0.0

    max_corr = 0.0

    for col in historical_returns_matrix.columns:
        series = historical_returns_matrix[col].dropna()
        if series.empty:
            continue

        combined = pd.concat([candidate_returns, series], axis=1).dropna()
        if len(combined) < min_overlap_days:
            continue

        corr = float(combined.iloc[:, 0].corr(combined.iloc[:, 1]))
        if np.isnan(corr):
            continue

        if corr > max_corr:
            max_corr = corr

    return max_corr <= threshold, max_corr
