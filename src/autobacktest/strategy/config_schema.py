"""Pydantic v2 strategy configuration schema validation."""

from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrategyConfig(BaseModel):
    """Pydantic v2 strategy configuration model.

    Enforces schemas with extra fields at root allowed (for custom strategy usage),
    redirecting strategy-specific custom parameters to the `params` dictionary,
    while preventing collisions with standard schema fields.
    """

    model_config = ConfigDict(extra="allow")

    universe: list[str] = Field(..., min_length=1, description="List of asset tickers in strategy universe")
    benchmark: str = Field("SPY", description="Benchmark index ticker")
    momentum_lookback: int = Field(12, ge=1, description="Momentum score lookback window")
    max_drawdown_limit: float = Field(0.20, ge=0.0, le=1.0, description="Max permitted drawdown in holdout")
    turnover_limit: float = Field(2.0, gt=0.0, description="Max permitted annualized turnover rate")
    params: dict[str, Any] = Field(default_factory=dict, description="Strategy-specific parameters")

    @model_validator(mode="after")
    def validate_no_collisions(self) -> "StrategyConfig":
        # Get all top-level keys dynamically except 'params' (Finding 11)
        top_level_keys = set(self.__class__.model_fields.keys())
        if self.model_extra:
            top_level_keys.update(self.model_extra.keys())
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
