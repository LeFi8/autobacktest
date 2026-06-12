"""CLI command 'init-strategy' implementation."""

from __future__ import annotations

import re
from typing import Any

import typer
import yaml

from autobacktest.config import settings as default_settings
from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.templates import TEMPLATE_REGISTRY, render_program_template, render_strategy_source


def _validate_strategy_name(name: str | None) -> str:
    """Prompt for a strategy name if not provided and validate it as snake_case.

    Prints a visible warning when the name is modified by normalization.

    Args:
        name: Candidate strategy name (may be ``None`` to trigger interactive prompt).

    Returns:
        str: Normalised snake_case strategy name.

    Raises:
        typer.Exit: When the name fails the snake_case regex.
    """
    if not name:
        name = typer.prompt("Enter a unique name for your strategy (snake_case)")

    original = name.strip()
    strategy_name = re.sub(r"\s+", "_", original.lower())

    if strategy_name != original:
        typer.secho(
            f"Note: Strategy name '{original}' normalized to '{strategy_name}' (Python snake_case convention).",
            fg=typer.colors.YELLOW,
        )

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


def _print_end_summary(
    strategy_name: str,
    template: str,
    config_data: dict[str, Any],
    cash_asset: str,
    config_file: Any,
    strategy_file: Any,
    program_file: Any,
) -> None:
    """Print a formatted summary after successful strategy initialization."""
    typer.echo(f"\n[Success] Strategy '{strategy_name}' initialized!")
    typer.echo(f"  Template:     {template}")
    typer.echo(f"  Universe:     {', '.join(config_data.get('universe', []))}")
    typer.echo(f"  Benchmark:    {config_data.get('benchmark', 'SPY')}")
    typer.echo(f"  Cash Asset:   {cash_asset}")
    typer.echo(f"  Max DD:       {config_data.get('max_drawdown_limit', 0.20) * 100:.0f}%")
    typer.echo(f"  Turnover:     {config_data.get('turnover_limit', 2.0)}")
    typer.echo(f"  Lookback:     {config_data.get('momentum_lookback', 12)} months")
    typer.echo("  Files:")
    typer.echo(f"    Config:     {config_file.resolve()}")
    typer.echo(f"    Strategy:   {strategy_file.resolve()}")
    typer.echo(f"    Program:    {program_file.resolve()}")


def init_strategy_impl(
    name: str | None,
    overwrite: bool,
    silent_universe: str | None = None,
    silent_benchmark: str = "SPY",
    silent_max_drawdown: float = 0.20,
    silent_turnover: float = 2.0,
    silent_lookback: int = 12,
    silent_template: str = "equal-weight",
    silent_cash_asset: str = "BIL",
    settings_obj: Any = default_settings,
) -> None:
    """Scaffold a new strategy with Pydantic-validated boilerplate.

    When *silent_universe* is provided the function runs in fully non-interactive
    mode — no prompts are displayed and unspecified values use schema defaults.
    When *silent_universe* is ``None`` the interactive wizard is shown (with any
    other ``silent_*`` values pre-filling prompt defaults).

    Args:
        name: Strategy name in snake_case. Prompts interactively if ``None``.
        overwrite: Overwrite existing files without prompting.
        silent_universe: Comma-separated tickers. When provided, triggers silent mode.
        silent_benchmark: Benchmark ticker (default ``"SPY"``).
        silent_max_drawdown: Max drawdown limit (default ``0.20``).
        silent_turnover: Annualized turnover limit (default ``2.0``).
        silent_lookback: Momentum lookback months (default ``12``).
        silent_template: Strategy template key (default ``"equal-weight"``).
        silent_cash_asset: Cash/risk-free asset ticker (default ``"BIL"``).
        settings_obj: Settings object (injected for testability).
    """
    strategy_name = _validate_strategy_name(name)
    silent = silent_universe is not None

    if silent_template not in TEMPLATE_REGISTRY:
        valid = ", ".join(sorted(TEMPLATE_REGISTRY))
        typer.echo(f"Error: Unknown template '{silent_template}'. Valid options: {valid}")
        raise typer.Exit(code=1)

    strategies_dir = settings_obj.strategies_dir
    strategy_dir = strategies_dir / strategy_name
    strategy_file = strategy_dir / "strategy.py"
    config_file = strategy_dir / "config.yaml"
    program_file = strategy_dir / "program.md"

    if not _confirm_files_overwrite(strategy_file, config_file, program_file, strategy_name, overwrite):
        raise typer.Exit(code=0)

    custom_params: dict[str, Any]
    if silent:
        assert silent_universe is not None
        universe = [t.strip().upper() for t in silent_universe.split(",") if t.strip()]
        if not universe:
            typer.echo("Error: Universe must contain at least one asset ticker.")
            raise typer.Exit(code=1)
        benchmark = silent_benchmark.strip().upper()
        mdd = silent_max_drawdown
        turnover = silent_turnover
        momentum_lookback = silent_lookback
        template = silent_template
        cash_asset = silent_cash_asset.strip().upper()
        advanced_params: dict[str, Any] = {}
        custom_params = {}
    else:
        typer.echo("\n--- Strategy Configuration Setup Wizard ---\n")
        universe = _prompt_universe_tickers()
        benchmark = typer.prompt("Enter benchmark asset ticker", default=silent_benchmark).strip().upper()
        mdd = _prompt_valid_float(
            "Max drawdown limit (0.0 to 1.0)",
            str(silent_max_drawdown),
            0.0,
            1.0,
            "Error: Drawdown limit must be between 0.0 and 1.0.",
        )
        turnover = _prompt_valid_float(
            "Annualized turnover limit (e.g. 2.0)",
            str(silent_turnover),
            1e-9,
            float("inf"),
            "Error: Turnover limit must be greater than 0.0.",
        )
        momentum_lookback = _prompt_valid_int(
            "Momentum score lookback window (months)",
            str(silent_lookback),
            1,
            "Error: Momentum lookback must be at least 1.",
        )
        cash_asset = typer.prompt("Cash/risk-free asset ticker", default=silent_cash_asset).strip().upper()
        template = (
            typer.prompt(
                "Strategy template",
                default=silent_template,
            )
            .strip()
            .lower()
        )
        if template not in TEMPLATE_REGISTRY:
            valid = ", ".join(sorted(TEMPLATE_REGISTRY))
            typer.echo(f"Error: Unknown template '{template}'. Valid options: {valid}")
            raise typer.Exit(code=1)
        advanced_params = _prompt_advanced_config()
        reserved_keys = set(StrategyConfig.model_fields.keys()) - {"params"}
        custom_params = {}
        if typer.confirm("\nDo you want to define custom strategy parameters?", default=False):
            custom_params = _prompt_custom_params(reserved_keys)

    custom_params["cash_asset"] = cash_asset

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

    strategy_dir.mkdir(parents=True, exist_ok=True)

    with config_file.open("w", encoding="utf-8") as f:
        yaml.safe_dump(validated_config.model_dump(), f, default_flow_style=False, sort_keys=False)

    universe_str = ", ".join(universe)

    py_source = render_strategy_source(strategy_name, cash_asset, template)
    strategy_file.write_text(py_source, encoding="utf-8")

    program_md = render_program_template(
        template, strategy_name, universe_str, benchmark, mdd, turnover, momentum_lookback
    )
    program_file.write_text(program_md, encoding="utf-8")

    _print_end_summary(strategy_name, template, config_data, cash_asset, config_file, strategy_file, program_file)
