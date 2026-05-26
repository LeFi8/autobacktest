"""Strategy registry configurations, contracts, validation, and pre-flight tests."""

from autobacktest.strategy.config_schema import StrategyConfig
from autobacktest.strategy.contract import validate_output, validate_signature
from autobacktest.strategy.validator import ValidationResult, preflight

__all__ = [
    "StrategyConfig",
    "ValidationResult",
    "preflight",
    "validate_output",
    "validate_signature",
]
