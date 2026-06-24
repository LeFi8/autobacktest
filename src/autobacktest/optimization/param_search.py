"""Optuna-based numeric parameter optimization for AutoBacktest.

After the LLM selects a winning strategy structure each iteration, this
module tunes the numeric parameters on in-sample data using Optuna TPE
— zero extra LLM tokens.
"""

from __future__ import annotations

import dataclasses
import logging
from copy import deepcopy
from typing import Any

import pandas as pd
import yaml

from autobacktest.strategy.config_jitter import (
    _EXCLUDE_JITTER_KEYS,
    _get_param_bounds,
)

logger = logging.getLogger(__name__)


def _build_optuna_space(
    flat_config: dict[str, Any],
    exclude: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build an Optuna search space from bounded numeric params in *flat_config*.

    Only parameters with finite (min, max) bounds are included — unbounded
    or non-numeric parameters are skipped.

    Args:
        flat_config: Flat (or model-dumped) strategy configuration dict.
        exclude: Optional set of parameter names to exclude.

    Returns:
        Dict mapping param name to ``{"low", "high", "value", "is_int"}``.
    """
    exclude = exclude or set()
    space: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()

    def _try_add(k: str, v: Any, path_key: str) -> None:
        if k in seen:
            return
        if not isinstance(v, (int, float)) or isinstance(v, bool):
            return
        if k in _EXCLUDE_JITTER_KEYS or k in exclude:
            return
        min_val, max_val = _get_param_bounds(k, (path_key,))
        if min_val is None or max_val is None:
            return
        space[k] = {
            "low": min_val,
            "high": max_val,
            "value": v,
            "is_int": isinstance(v, int) and not isinstance(v, bool),
        }
        seen.add(k)

    for k, v in flat_config.items():
        if k == "params":
            continue
        _try_add(k, v, k)

    params_dict = flat_config.get("params", {})
    if isinstance(params_dict, dict):
        for k, v in params_dict.items():
            _try_add(k, v, "params")

    return space


def optimize_numeric_params(
    generate_signals_fn: Any,
    flat_config: dict[str, Any],
    prices: pd.DataFrame,
    n_trials: int = 20,
    seed: int = 42,
    exclude: set[str] | None = None,
    _obj: str = "sharpe",
) -> tuple[dict[str, Any], float, bool]:
    """Run Optuna TPE optimization on numeric config params (in-sample only).

    Args:
        generate_signals_fn: Strategy signal generation function.
        flat_config: Flat strategy configuration dict.
        prices: Full price DataFrame (in-sample + holdout) — only the
            in-sample portion is used internally.
        n_trials: Number of Optuna trials.
        seed: Random seed for deterministic TPE sampling.
        exclude: Optional set of parameter names to exclude.
        _obj: Objective metric (reserved for future use).

    Returns:
        Tuple of ``(optimized_config, best_sharpe, improved)`` where
        ``improved`` is ``True`` when the best trial's Sharpe exceeds the
        original config's Sharpe.
    """
    import optuna.samplers

    from autobacktest.config import settings
    from autobacktest.evaluator.evaluate import _in_sample_objective
    from autobacktest.evaluator.holdout import (
        partition_holdout_data as _partition_holdout_data,
    )
    from autobacktest.evaluator.walk_forward import (
        generate_walk_forward_windows as _generate_walk_forward_windows,
    )

    search_space = _build_optuna_space(flat_config, exclude=exclude)
    if not search_space:
        return flat_config, 0.0, False

    in_sample_idx, _ = _partition_holdout_data(prices.index, holdout_years=settings.default_holdout_years)
    wf_windows = _generate_walk_forward_windows(in_sample_idx, train_years=5, test_years=1)
    asset_returns = prices.pct_change().fillna(0.0)

    original_sharpe, _, _ = _in_sample_objective(
        generate_signals_fn, flat_config, prices, _wf_windows=wf_windows, _asset_returns=asset_returns
    )
    if pd.isna(original_sharpe):
        original_sharpe = 0.0

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed),
    )

    def _objective(trial: Any) -> float:
        trial_config = deepcopy(flat_config)
        for name, spec in search_space.items():
            if spec["is_int"]:
                val = trial.suggest_int(name, int(spec["low"]), int(spec["high"]))
            else:
                val = trial.suggest_float(name, spec["low"], spec["high"])
            if name in trial_config:
                trial_config[name] = val
            elif (
                "params" in trial_config and isinstance(trial_config["params"], dict) and name in trial_config["params"]
            ):
                trial_config["params"][name] = val
        try:
            sh, _, _ = _in_sample_objective(
                generate_signals_fn, trial_config, prices, _wf_windows=wf_windows, _asset_returns=asset_returns
            )
            if pd.isna(sh):
                return -float("inf")
            return float(sh)
        except Exception:
            return -float("inf")

    study.optimize(_objective, n_trials=n_trials)

    best_sharpe = float(study.best_value)
    best_params = study.best_params
    optimized_config = deepcopy(flat_config)
    for name, val in best_params.items():
        if name in optimized_config:
            optimized_config[name] = val
        elif "params" in optimized_config and isinstance(optimized_config["params"], dict):
            optimized_config["params"][name] = val

    improved = best_sharpe > original_sharpe
    return optimized_config, best_sharpe, improved


def build_optimized_yaml(optimized_config: dict[str, Any]) -> str:
    """Render an optimized flat config dict back to YAML.

    Top-level scalar keys are emitted directly; a non-empty ``params`` sub-dict
    is preserved as a nested block. Shared by :func:`_apply_optimized_config`
    and the orchestrator's re-evaluation so the evaluated YAML and the committed
    YAML are byte-identical.
    """
    yaml_dict: dict[str, Any] = {}
    for k, v in optimized_config.items():
        if k == "params":
            continue
        yaml_dict[k] = v
    params = optimized_config.get("params", {})
    if isinstance(params, dict) and params:
        yaml_dict["params"] = params
    return yaml.safe_dump(yaml_dict, default_flow_style=False, sort_keys=False)


def _apply_optimized_config(
    winner: dict[str, Any],
    optimized_config: dict[str, Any],
    improved: bool,
    gain: float = 0.0,
) -> None:
    """Modify *winner* dict in-place with the optimized config.

    When *improved* is ``True``, the winner's config and edit are updated
    so that :func:`commit_winner` writes the optimized version.  Otherwise
    only the metadata flags are set.

    Args:
        winner: Mutable winner candidate dict with keys ``"edit"``,
            ``"config_yaml"``, ``"_config"``.
        optimized_config: The configuration dict produced by Optuna.
        improved: Whether optimization improved the in-sample Sharpe.
        gain: The in-sample Sharpe improvement (``best - original``).
    """
    if not improved:
        winner["optimization_applied"] = False
        winner["optimization_gain"] = 0.0
        return

    new_yaml = build_optimized_yaml(optimized_config)

    new_edit = dataclasses.replace(winner["edit"], config_yaml=new_yaml)
    winner["edit"] = new_edit
    winner["config_yaml"] = new_yaml
    winner["_config"] = optimized_config
    winner["optimization_applied"] = True
    winner["optimization_gain"] = gain
