"""CLI command 'init-strategy' implementation."""

from __future__ import annotations

import re
from typing import Any

import typer
import yaml

from autobacktest.config import settings as default_settings
from autobacktest.strategy.config_schema import StrategyConfig


def _validate_strategy_name(name: str | None) -> str:
    """Prompt for a strategy name if not provided and validate it as snake_case.

    Args:
        name: Candidate strategy name (may be ``None`` to trigger interactive prompt).

    Returns:
        str: Normalised snake_case strategy name.

    Raises:
        typer.Exit: When the name fails the snake_case regex.
    """
    if not name:
        name = typer.prompt("Enter a unique name for your strategy (snake_case)")
    strategy_name = re.sub(r"\s+", "_", name.strip().lower())
    if not re.match(r"^[a-z_][a-z0-9_]*$", strategy_name):
        typer.echo("Error: Strategy name must be a valid snake_case Python identifier.")
        raise typer.Exit(code=1)
    return strategy_name


def _confirm_files_overwrite(
    strategy_file: Any, config_file: Any, program_file: Any, strategy_name: str, overwrite: bool
) -> bool:
    """Return ``True`` when the operation should proceed, ``False`` when the user cancels.

    When files already exist and *overwrite* is ``False``, prompts the user interactively.

    Args:
        strategy_file: ``Path`` to the ``.py`` strategy file.
        config_file: ``Path`` to the ``.yaml`` config file.
        program_file: ``Path`` to the ``.md`` program file.
        strategy_name: Display name used in the confirmation prompt.
        overwrite: If ``True``, skip the prompt and proceed.

    Returns:
        bool: ``True`` if the caller should proceed, ``False`` to abort.
    """
    if (strategy_file.exists() or config_file.exists() or program_file.exists()) and not overwrite:
        confirm = typer.confirm(
            f"Strategy files for '{strategy_name}' already exist. Overwrite?",
            default=False,
        )
        if not confirm:
            typer.echo("Operation cancelled.")
            return False
    return True


def _prompt_universe_tickers() -> list[str]:
    """Interactively prompt for asset tickers until at least one is provided.

    Returns:
        list[str]: Uppercase ticker symbols.
    """
    while True:
        universe_raw = typer.prompt("Enter assets universe (comma-separated, e.g. SPY, QQQ, BIL)")
        universe = [t.strip().upper() for t in universe_raw.split(",") if t.strip()]
        if universe:
            return universe
        typer.echo("Error: Universe must contain at least one asset ticker.")


def _prompt_valid_float(prompt: str, default: str, lo: float, hi: float, range_err: str) -> float:
    """Prompt for a float value, looping until it falls within ``[lo, hi]``.

    Args:
        prompt: The prompt text shown to the user.
        default: Default value displayed in the prompt.
        lo: Inclusive lower bound.
        hi: Inclusive upper bound.
        range_err: Error message shown when the value is out of range.

    Returns:
        float: Validated float value.
    """
    while True:
        try:
            val = float(typer.prompt(prompt, default=default))
            if lo <= val <= hi:
                return val
            typer.echo(range_err)
        except ValueError:
            typer.echo("Error: Please enter a valid decimal number.")


def _prompt_valid_int(prompt: str, default: str, min_val: int, range_err: str) -> int:
    """Prompt for an integer value, looping until it is ``>= min_val``.

    Args:
        prompt: The prompt text shown to the user.
        default: Default value displayed in the prompt.
        min_val: Inclusive lower bound.
        range_err: Error message shown when the value is too small.

    Returns:
        int: Validated integer value.
    """
    while True:
        try:
            val = int(typer.prompt(prompt, default=default))
            if val >= min_val:
                return val
            typer.echo(range_err)
        except ValueError:
            typer.echo("Error: Please enter a valid integer.")


def _parse_param_value(raw: str) -> Any:
    """Coerce a raw string to the most specific scalar type possible.

    Attempts boolean keywords first, then ``int``, then ``float``,
    and falls back to the original string.

    Args:
        raw: Raw input string from the user.

    Returns:
        Any: Parsed value (``bool``, ``int``, ``float``, or ``str``).
    """
    lower = raw.lower()
    if lower in ("true", "yes", "on"):
        return True
    if lower in ("false", "no", "off"):
        return False
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _prompt_custom_params(reserved_keys: set[str]) -> dict[str, Any]:
    """Interactively collect custom strategy parameters from the user.

    Loops until the user submits an empty parameter name.  Rejects names
    that collide with reserved ``StrategyConfig`` fields.

    Args:
        reserved_keys: Set of field names already used by ``StrategyConfig``.

    Returns:
        dict[str, Any]: Mapping of parameter name to coerced value.
    """
    custom_params: dict[str, Any] = {}
    while True:
        param_key = typer.prompt("Parameter name (or press Enter to finish)", default="").strip()
        if not param_key:
            break
        if param_key in reserved_keys:
            typer.echo(f"Error: '{param_key}' is a reserved schema field. Choose a different name.")
            continue
        param_val_raw = typer.prompt(f"Value for '{param_key}'")
        custom_params[param_key] = _parse_param_value(param_val_raw)
    return custom_params


def _prompt_advanced_config() -> dict[str, Any]:
    """Interactively prompt for key ``StrategyConfig`` fields.

    Gated behind a single yes/no prompt.  Fields that the user skips
    (empty input for optional values) are omitted so the schema default
    is used.

    Returns:
        dict[str, Any]: Advanced config keys to merge into the config data.
    """
    if not typer.confirm("Do you want to configure advanced strategy parameters?", default=False):
        return {}

    params: dict[str, Any] = {}

    params["borrow_cost_bps"] = _prompt_valid_float(
        "Annualized short borrowing cost (bps)",
        "100.0",
        0.0,
        float("inf"),
        "Error: Borrow cost must be >= 0.",
    )

    params["cscv_blocks"] = _prompt_valid_int(
        "CSCV blocks for PBO calculation",
        "10",
        4,
        "Error: CSCV blocks must be at least 4.",
    )

    pbo_raw = typer.prompt("PBO limit (press Enter for no limit)", default="").strip()
    if pbo_raw:
        try:
            v = float(pbo_raw)
            if 0.0 <= v <= 1.0:
                params["pbo_limit"] = v
            else:
                typer.echo("Error: PBO limit must be between 0.0 and 1.0. Using no limit.")
        except ValueError:
            typer.echo("Error: Invalid number. Using no limit.")

    if typer.confirm("Use adaptive slippage?", default=False):
        params["adaptive_slippage"] = True

    params["min_improvement"] = _prompt_valid_float(
        "Minimum target-metric improvement epsilon",
        "0.0",
        0.0,
        float("inf"),
        "Error: Min improvement must be >= 0.",
    )

    params["select_min_return_ratio"] = _prompt_valid_float(
        "Min fraction of baseline annualized return for select gate",
        "0.5",
        0.0,
        1.0,
        "Error: Must be between 0.0 and 1.0.",
    )

    params["require_dsr_non_degradation"] = typer.confirm("Require DSR non-degradation in select gate?", default=True)

    while True:
        method = typer.prompt("MC bootstrap method", default="stationary")
        if method in ("circular", "stationary"):
            params["mc_bootstrap_method"] = method
            break
        typer.echo("Error: Must be 'circular' or 'stationary'.")

    return params


def _create_program_file(path: Any, name: str, config_data: dict[str, Any]) -> None:
    """Write a boilerplate ``program-{name}.md`` with populated constraints.

    Args:
        path: Output ``Path`` for the program file.
        name: Strategy name (for display).
        config_data: Dict of validated config values to interpolate.
    """
    universe_str = ", ".join(config_data.get("universe", []))
    mdd = config_data.get("max_drawdown_limit", 0.20)
    turnover = config_data.get("turnover_limit", 2.0)
    lookback = config_data.get("momentum_lookback", 12)
    benchmark = config_data.get("benchmark", "SPY")

    content = f"""# Objective

Optimize the **{name}** strategy ({universe_str} universe, benchmark: {benchmark}).

Describe your specific targets here — the LLM optimizer uses this as its north star.

Examples:
- "Increase Sharpe ratio above 1.2 while keeping drawdown below {mdd * 100:.0f}%"
- "Improve risk-adjusted returns; reduce turnover below {turnover:.1f}"
- "Maintain CAGR above 10% with max drawdown under {mdd * 100:.0f}%"

# Constraints

- Maximum drawdown in the holdout period must not exceed {mdd * 100:.0f}%.
- Annualized portfolio turnover must remain below {turnover:.1f}.
- The strategy must pass all regime stress tests (2008 GFC, 2020 COVID, 2022 bear market).
- Monthly rebalancing on the last trading day must be preserved (momentum lookback: {lookback} months).
- Only ``pandas`` and ``numpy`` imports are permitted in the strategy code.
- The ``generate_signals(prices: pd.DataFrame, config: dict) -> pd.DataFrame`` signature must be preserved.

## Strategy Details

Optional: Add background, references, or design notes here.

**Asset universe:** {universe_str}
**Benchmark:** {benchmark}

---

**Tip:** Run ``uv run autobacktest run --program {path.name} --strategy {name} --iterations 5`` to start optimizing.
"""
    path.write_text(content, encoding="utf-8")


def init_strategy_impl(
    name: str | None,
    overwrite: bool,
    settings_obj: Any = default_settings,
) -> None:
    """Interactively scaffold a new strategy with Pydantic-validated boilerplate.

    Prompts for universe tickers, benchmark, risk limits, and custom parameters,
    generates validated ``configs/{name}.yaml`` and ``strategies/{name}.py`` files.

    Args:
        name: Strategy name in snake_case. Prompts interactively if ``None``.
        overwrite: Overwrite existing files without prompting.
        settings_obj: Settings object (injected for testability).
    """
    strategy_name = _validate_strategy_name(name)

    strategies_dir = settings_obj.strategies_dir
    configs_dir = settings_obj.configs_dir
    project_root = strategies_dir.parent
    strategy_file = strategies_dir / f"{strategy_name}.py"
    config_file = configs_dir / f"{strategy_name}.yaml"
    program_file = project_root / f"program-{strategy_name}.md"

    if not _confirm_files_overwrite(strategy_file, config_file, program_file, strategy_name, overwrite):
        raise typer.Exit(code=0)

    typer.echo("\n--- Strategy Configuration Setup Wizard ---\n")

    universe = _prompt_universe_tickers()
    benchmark = typer.prompt("Enter benchmark asset ticker", default="SPY").strip().upper()
    mdd = _prompt_valid_float(
        "Max drawdown limit (0.0 to 1.0)",
        "0.20",
        0.0,
        1.0,
        "Error: Drawdown limit must be between 0.0 and 1.0.",
    )
    turnover = _prompt_valid_float(
        "Annualized turnover limit (e.g. 2.0)",
        "2.0",
        1e-9,
        float("inf"),
        "Error: Turnover limit must be greater than 0.0.",
    )
    momentum_lookback = _prompt_valid_int(
        "Momentum score lookback window (months)",
        "12",
        1,
        "Error: Momentum lookback must be at least 1.",
    )

    advanced_params = _prompt_advanced_config()

    reserved_keys = set(StrategyConfig.model_fields.keys()) - {"params"}
    custom_params: dict[str, Any] = {}
    if typer.confirm("\nDo you want to define custom strategy parameters?", default=False):
        custom_params = _prompt_custom_params(reserved_keys)

    try:
        config_data: dict[str, Any] = {
            "universe": universe,
            "benchmark": benchmark,
            "momentum_lookback": momentum_lookback,
            "max_drawdown_limit": mdd,
            "turnover_limit": turnover,
            **advanced_params,
            "params": custom_params,
        }
        validated_config = StrategyConfig(**config_data)
    except Exception as e:
        typer.echo(f"\nConfiguration validation failed: {e}")
        raise typer.Exit(code=1) from None

    strategies_dir.mkdir(parents=True, exist_ok=True)
    configs_dir.mkdir(parents=True, exist_ok=True)

    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(validated_config.model_dump(), f, default_flow_style=False, sort_keys=False)

    py_boilerplate = f'''"""{strategy_name} strategy — generated by autobacktest init-strategy."""

from typing import Any

import numpy as np
import pandas as pd


def generate_signals(prices: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Generate portfolio allocation weights.

    Args:
        prices: Daily close prices DataFrame (DatetimeIndex).
        config: Strategy configuration dictionary.

    Returns:
        pd.DataFrame: Weights DataFrame indexed by rebalance dates.
    """
    universe = config.get("universe", [])
    cash_asset = config.get("params", {{}}).get("cash_asset", "BIL")

    if prices.empty:
        return pd.DataFrame(0.0, index=pd.DatetimeIndex([]), columns=prices.columns)

    available = set(prices.columns)
    if cash_asset not in available:
        raise ValueError(f"Cash asset {{cash_asset}} not in price data")

    start = prices.index.min()
    end = prices.index.max()
    rebalance_dates = pd.date_range(start=start, end=end, freq="BME", tz=prices.index.tz).intersection(prices.index)
    weights = pd.DataFrame(0.0, index=rebalance_dates, columns=prices.columns)

    for date in rebalance_dates:
        valid_assets = [t for t in universe if t in available]
        if valid_assets:
            w = 1.0 / len(valid_assets)
            for asset in valid_assets:
                weights.loc[date, asset] = w

    return weights
'''
    strategy_file.write_text(py_boilerplate, encoding="utf-8")

    _create_program_file(program_file, strategy_name, config_data)

    typer.echo(f"\n[Success] Strategy '{strategy_name}' initialized!")
    typer.echo(f"  Config:   {config_file.resolve()}")
    typer.echo(f"  Strategy: {strategy_file.resolve()}")
    typer.echo(f"  Program:  {program_file.resolve()}")
