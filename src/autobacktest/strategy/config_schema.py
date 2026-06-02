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
    turnover_limit: float = Field(2.0, gt=0.0, le=10.0, description="Max annualized turnover (capped at 10x)")
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
