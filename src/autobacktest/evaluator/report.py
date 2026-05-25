"""Performance and evaluation reports structure."""

import json
from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class WindowReport:
    """Performance metrics for a specific walk-forward or holdout window."""

    start_date: str
    end_date: str
    annualized_return: float
    annualized_volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    turnover: float
    information_ratio: float = 0.0


@dataclass
class EvaluationReport:
    """Full deterministic backtest evaluation report."""

    strategy_name: str
    dataset_hash: str
    gates_passed: dict[str, bool]
    is_accepted: bool
    rejection_reason: str | None

    # Performance Summaries
    holdout_metrics: WindowReport
    walk_forward_metrics: list[WindowReport]

    # Stress testing & advanced diagnostics
    regime_drawdowns: dict[str, float]
    regime_passed: bool

    # Monte Carlo bootstrap percentiles
    mc_sharpe_5th: float
    mc_sharpe_50th: float
    mc_sharpe_95th: float

    # DSR accounting
    observed_sharpe: float
    effective_trials: int
    deflated_sharpe: float

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary representation."""
        return asdict(self)

    def to_json(self, indent: int = 4) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
