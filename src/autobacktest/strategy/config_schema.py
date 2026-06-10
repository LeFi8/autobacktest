"""Pydantic v2 strategy configuration schema validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrategyConfig(BaseModel):
    """Pydantic v2 strategy configuration model.

    Extra fields at the root are **forbidden** — strategy-specific custom
    parameters must be placed in the ``params`` dictionary.  This prevents
    the LLM from injecting arbitrary top-level keys (e.g. overriding
    ``turnover_limit`` or ``max_drawdown_limit``).
    """

    model_config = ConfigDict(extra="forbid")

    universe: list[str] = Field(..., min_length=1, description="List of asset tickers")
    benchmark: str = Field("SPY", description="Benchmark index ticker")
    momentum_lookback: int = Field(12, ge=1, description="Momentum lookback window (months)")
    max_drawdown_limit: float = Field(0.20, ge=0.0, le=1.0, description="Max permitted drawdown")
    turnover_limit: float = Field(2.0, gt=0.0, description="Max annualized turnover")
    borrow_cost_bps: float = Field(100.0, ge=0.0, description="Annualized short borrowing cost in bps")
    cscv_blocks: int = Field(10, ge=4, description="Number of blocks to partition returns for CSCV PBO calculation")
    pbo_limit: float | None = Field(None, ge=0.0, le=1.0, description="PBO ceiling select gate limit")
    cscv_embargo_days: int = Field(5, ge=0, description="CSCV block embargo days")
    adaptive_slippage: bool = Field(False, description="Use volatility-adaptive slippage")
    slippage_vol_window: int = Field(21, ge=1, description="Volatility window for adaptive slippage")
    slippage_vol_cap: float = Field(3.0, ge=1.0, description="Volatility cap multiplier for adaptive slippage")
    mc_bootstrap_method: str = Field("stationary", description="Bootstrap method: circular or stationary")
    regime_benchmark: str | None = Field(
        None, description="Alternative benchmark index for launch-regime timing haircut"
    )
    params: dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")
    min_improvement: float = Field(0.0, ge=0.0, description="Minimum target-metric improvement epsilon for select gate")
    select_min_return_ratio: float = Field(
        0.5, ge=0.0, le=1.0, description="Min fraction of baseline annualized return for select gate"
    )
    require_dsr_non_degradation: bool = Field(
        True, description="If True (default), selection gate enforces DSR non-degradation"
    )
    holdout_min_improvement: float = Field(0.0, description="Tolerance for holdout DSR non-degradation in confirm gate")
    enable_holdout_confirmation: bool = Field(
        True,
        description="If True, select-passing candidates are confirmed on holdout before commit",
    )
    dsr_floor: float | None = Field(
        None, description="Optional absolute DSR floor (unused by gate currently, reserved)"
    )
    metric_return_tradeoff: float = Field(
        0.0,
        ge=0.0,
        description="Maximum acceptable target metric reduction per 1pp (0.01) increase in annualized return. "
        "Example: 0.1 reduces the required metric by 0.5 for a 5pp return improvement.",
    )
    metric_floor: float | None = Field(
        None,
        description="Absolute metric floor. Candidates below this value are always rejected. "
        "Unit follows select_compare_metric: DSR when 'deflated' (default), raw target_metric "
        "(Sharpe/Sortino/IR) when 'raw'. If the baseline is already below this floor, a warning "
        "is emitted at setup time.",
    )
    select_compare_metric: str = Field(
        "deflated",
        description="Metric used for the select-gate improvement comparison. "
        "'deflated' uses Deflated Sharpe Ratio (robust, overfit-adjusted). "
        "'raw' uses the in-sample target_metric (Sharpe/Sortino/Information Ratio). "
        "Unit of min_improvement follows this setting.",
    )
    select_improvement_tol: float = Field(
        0.02,
        ge=0.0,
        description="Non-negative tolerance for the select-gate improvement comparison. "
        "A candidate is accepted when its comparison metric >= incumbent's metric minus this "
        "tolerance, so near-ties are not rejected.",
    )

    @field_validator("select_compare_metric")
    @classmethod
    def validate_select_compare_metric(cls, v: str) -> str:
        if v not in ("deflated", "raw"):
            raise ValueError("select_compare_metric must be either 'deflated' or 'raw'")
        return v

    @field_validator("mc_bootstrap_method")
    @classmethod
    def validate_mc_bootstrap_method(cls, v: str) -> str:
        """Verify that mc_bootstrap_method is circular or stationary."""
        if v not in ("circular", "stationary"):
            raise ValueError("mc_bootstrap_method must be either 'circular' or 'stationary'")
        return v

    @model_validator(mode="after")
    def validate_no_collisions(self) -> "StrategyConfig":
        # Get all known top-level fields and exclude 'params' (Finding 11)
        top_level_keys = set(self.__class__.model_fields.keys())
        top_level_keys.discard("params")

        colliding = top_level_keys.intersection(self.params.keys())
        if colliding:
            raise ValueError(f"Keys in 'params' collide with top-level schema fields: {colliding}")
        return self

    @field_validator("universe")
    @classmethod
    def validate_universe_nonempty(cls, v: list[str]) -> list[str]:
        """Verify that universe is not empty and tickers are valid."""
        if not v:
            raise ValueError("universe list must contain at least one ticker")
        for ticker in v:
            if not ticker or not isinstance(ticker, str):
                raise ValueError("Tickers in universe must be non-empty strings")
        return v

    @field_validator("benchmark")
    @classmethod
    def validate_benchmark_nonempty(cls, v: str) -> str:
        """Verify that benchmark ticker is valid."""
        if not v or not v.strip():
            raise ValueError("benchmark must be a non-empty string")
        return v.strip()

    @classmethod
    def from_yaml(cls, path: Path) -> "StrategyConfig":
        """Load strategy configuration from a YAML file."""
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return cls.model_validate(data)

    def to_flat_dict(self) -> dict[str, Any]:
        """Return a flat dictionary representing the configuration."""
        res = self.model_dump()
        params = res.get("params", {})
        if isinstance(params, dict):
            for k, v in params.items():
                if k not in res:
                    res[k] = v
        return res

    @classmethod
    def constraints_text(cls) -> str:
        """Render schema field constraints and default values as text."""
        from pydantic_core import PydanticUndefined

        lines = []
        for name, field in cls.model_fields.items():
            ann = field.annotation
            if ann is None:
                t_name = "None"
            elif hasattr(ann, "__name__"):
                t_name = ann.__name__
            else:
                t_name = str(ann).replace("typing.", "")

            constraints = []
            for m in getattr(field, "metadata", []):
                for op, symbol in [("ge", ">="), ("gt", ">"), ("le", "<="), ("lt", "<")]:
                    val = getattr(m, op, None)
                    if val is not None:
                        constraints.append(f"{symbol} {val}")
                min_len = getattr(m, "min_length", None)
                if min_len is not None:
                    constraints.append(f"min_length={min_len}")

            if constraints:
                j = ", "
                constraints_str = f", {j.join(constraints)}"
            else:
                constraints_str = ""

            default_val = field.default
            default_str = "" if default_val is PydanticUndefined else f" (default: {default_val})"

            lines.append(f"- {name}: {t_name}{constraints_str}{default_str}")
        return "\n".join(lines)
