"""CLI command 'init-strategy' implementation."""

from __future__ import annotations

import re
from typing import Any

import typer
import yaml

from autobacktest.config import settings as default_settings
from autobacktest.strategy.config_schema import StrategyConfig


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
    if not name:
        name = typer.prompt("Enter a unique name for your strategy (snake_case)")

    strategy_name = re.sub(r"\s+", "_", name.strip().lower())
    if not re.match(r"^[a-z_][a-z0-9_]*$", strategy_name):
        typer.echo("Error: Strategy name must be a valid snake_case Python identifier.")
        raise typer.Exit(code=1)

    strategies_dir = settings_obj.strategies_dir
    configs_dir = settings_obj.configs_dir
    strategy_file = strategies_dir / f"{strategy_name}.py"
    config_file = configs_dir / f"{strategy_name}.yaml"

    if (strategy_file.exists() or config_file.exists()) and not overwrite:
        confirm = typer.confirm(
            f"Strategy files for '{strategy_name}' already exist. Overwrite?",
            default=False,
        )
        if not confirm:
            typer.echo("Operation cancelled.")
            raise typer.Exit(code=0)

    typer.echo("\n--- Strategy Configuration Setup Wizard ---\n")

    while True:
        universe_raw = typer.prompt(
            "Enter assets universe (comma-separated, e.g. SPY, QQQ, BIL)",
        )
        universe = [t.strip().upper() for t in universe_raw.split(",") if t.strip()]
        if len(universe) > 0:
            break
        typer.echo("Error: Universe must contain at least one asset ticker.")

    benchmark = typer.prompt("Enter benchmark asset ticker", default="SPY").strip().upper()

    while True:
        try:
            mdd = float(typer.prompt("Max drawdown limit (0.0 to 1.0)", default="0.20"))
            if 0.0 <= mdd <= 1.0:
                break
            typer.echo("Error: Drawdown limit must be between 0.0 and 1.0.")
        except ValueError:
            typer.echo("Error: Please enter a valid decimal number.")

    while True:
        try:
            turnover = float(typer.prompt("Annualized turnover limit (e.g. 2.0)", default="2.0"))
            if turnover > 0.0:
                break
            typer.echo("Error: Turnover limit must be greater than 0.0.")
        except ValueError:
            typer.echo("Error: Please enter a valid decimal number.")

    while True:
        try:
            momentum_lookback = int(typer.prompt("Momentum score lookback window (months)", default="12"))
            if momentum_lookback >= 1:
                break
            typer.echo("Error: Momentum lookback must be at least 1.")
        except ValueError:
            typer.echo("Error: Please enter a valid integer.")

    reserved_keys = set(StrategyConfig.model_fields.keys()) - {"params"}
    custom_params: dict[str, Any] = {}
    if typer.confirm("\nDo you want to define custom strategy parameters?", default=False):
        while True:
            param_key = typer.prompt("Parameter name (or press Enter to finish)", default="").strip()
            if not param_key:
                break

            if param_key in reserved_keys:
                typer.echo(f"Error: '{param_key}' is a reserved schema field. Choose a different name.")
                continue

            param_val_raw = typer.prompt(f"Value for '{param_key}'")

            lower = param_val_raw.lower()
            param_val: Any
            if lower in ("true", "yes", "on"):
                param_val = True
            elif lower in ("false", "no", "off"):
                param_val = False
            else:
                try:
                    param_val = int(param_val_raw)
                except ValueError:
                    try:
                        param_val = float(param_val_raw)
                    except ValueError:
                        param_val = param_val_raw

            custom_params[param_key] = param_val

    try:
        config_data: dict[str, Any] = {
            "universe": universe,
            "benchmark": benchmark,
            "momentum_lookback": momentum_lookback,
            "max_drawdown_limit": mdd,
            "turnover_limit": turnover,
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

    typer.echo(f"\n[Success] Strategy '{strategy_name}' initialized!")
    typer.echo(f"  Config:   {config_file.resolve()}")
    typer.echo(f"  Strategy: {strategy_file.resolve()}")
