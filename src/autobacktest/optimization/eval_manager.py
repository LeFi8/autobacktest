"""Parallel evaluation loop management for optimization candidates."""

from __future__ import annotations

import importlib.util
import sys
import uuid
from pathlib import Path
from typing import Any

import pandas as pd

from autobacktest.evaluator.evaluate import _CacheProtocol, evaluate_strategy_detailed
from autobacktest.evaluator.report import EvaluationReport
from autobacktest.strategy.config_schema import StrategyConfig


def load_signals(path: Path) -> Any:
    """Dynamically import generate_signals from a strategy .py file.

    Uses ``importlib.util.spec_from_file_location`` to load the module,
    then evicts it from ``sys.modules`` to prevent namespace pollution
    (only if the module was newly registered).

    Args:
        path: Path to the strategy ``.py`` file.

    Returns:
        Any: The module's ``generate_signals`` function.

    Raises:
        ImportError: When the module cannot be loaded.
        AttributeError: When the module has no ``generate_signals`` function.
    """
    module_name = path.stem
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load strategy module from {path}")
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    finally:
        # Only evict if the module registered itself — avoids clobbering a
        # legitimate stdlib/third-party module that shares the same name.
        if sys.modules.get(module_name) is module:
            sys.modules.pop(module_name)
    if not hasattr(module, "generate_signals"):
        raise AttributeError(f"Strategy module {path} has no generate_signals function")
    return module.generate_signals


def eval_single_candidate(
    strategy_name: str,
    strategy_code: str,
    config_yaml: str,
    strategies_dir: Path,
    configs_dir: Path,
    start_date: str,
    end_date: str,
    eval_cache: _CacheProtocol,
) -> tuple[EvaluationReport | None, pd.Series[Any] | None, dict[str, Any] | None, str | None]:
    """Evaluate one candidate via temp files.

    Writes the candidate code and config to temporary files, loads signals
    dynamically, runs the full walk-forward + holdout evaluation, and cleans
    up.  Uses the provided eval cache to skip redundant evaluations.

    Args:
        strategy_name: The target strategy name.
        strategy_code: Source code of the candidate.
        config_yaml: YAML configuration of the candidate.
        strategies_dir: Path to strategies directory.
        configs_dir: Path to configs directory.
        start_date: Start of the backtest period.
        end_date: End of the backtest period.
        eval_cache: Memoization cache (dict-like) keyed by code+config hash.

    Returns:
        tuple[EvaluationReport | None, pd.Series | None, dict[str, Any] | None, str | None]:
        ``(report, in_sample_returns, flat_config, error_str)``.
        When evaluation fails all four values are ``None``.
    """
    temp_name = f"eval_{uuid.uuid4().hex}"
    temp_py = strategies_dir / f"{temp_name}.py"
    temp_yaml = configs_dir / f"{temp_name}.yaml"
    try:
        temp_py.write_text(strategy_code, encoding="utf-8")
        temp_yaml.write_text(config_yaml, encoding="utf-8")
        candidate_fn = load_signals(temp_py)
        new_config_obj = StrategyConfig.from_yaml(temp_yaml)
        new_config = new_config_obj.model_dump()
        report, returns = evaluate_strategy_detailed(
            strategy_name,
            candidate_fn,
            new_config,
            start_date=start_date,
            end_date=end_date,
            _eval_cache=eval_cache,
            _strategy_code=strategy_code,
        )
        return report, returns, new_config, None
    except Exception as e:
        return None, None, None, str(e)
    finally:
        for p in [temp_py, temp_yaml]:
            if p.exists():
                p.unlink()
