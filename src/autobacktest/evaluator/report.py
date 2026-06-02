"""Performance and evaluation reports structure."""

import json
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd


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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "WindowReport":
        """Reconstruct WindowReport from dictionary representation."""
        return cls(**data)


@dataclass
class EvaluationReport:
    """Full deterministic backtest evaluation report.

    The DSR fields serve two purposes:

    * ``deflated_sharpe`` — in-sample selection DSR, computed on the
      walk-forward aggregate returns.  This is the DSR used by the
      ``select`` gate (always-on non-degradation check).
    * ``holdout_deflated_sharpe`` — holdout confirmation DSR, computed on
      the out-of-sample holdout returns.  Deflated by the number of
      holdout peeks, tracked separately from the iteration count.
    * ``observed_sharpe`` — the annualised Sharpe of the in-sample
      walk-forward aggregate (``in_sample_metrics.sharpe_ratio``).
    """

    strategy_name: str
    dataset_hash: str
    gates_passed: dict[str, bool]
    is_accepted: bool
    rejection_reason: str | None

    # Performance Summaries
    holdout_metrics: WindowReport
    in_sample_metrics: WindowReport
    walk_forward_metrics: list[WindowReport]

    # Stress testing & advanced diagnostics
    regime_drawdowns: dict[str, float]
    regime_passed: bool

    # Monte Carlo bootstrap percentiles
    mc_sharpe_5th: float
    mc_sharpe_50th: float
    mc_sharpe_95th: float

    # DSR accounting (in-sample selection basis)
    observed_sharpe: float
    effective_trials: int
    deflated_sharpe: float

    # Holdout DSR (confirmation basis, deflated by peek count)
    holdout_deflated_sharpe: float = 0.0
    # Raw holdout returns — excluded from serialization; used by _deflate_holdout
    holdout_net_returns: pd.Series | None = field(default=None, repr=False, compare=False)
    # Benchmark returns for charting — excluded from serialization (pop'd in
    # to_dict()).  Will be None after from_json() round-trip.
    benchmark_returns: pd.Series | None = field(default=None, repr=False, compare=False)
    benchmark_ticker: str = "SPY"

    def to_dict(self) -> dict[str, Any]:
        """Convert the report to a dictionary representation."""
        d = asdict(self)
        d.pop("holdout_net_returns", None)
        d.pop("benchmark_returns", None)
        return d

    def to_json(self, indent: int = 4) -> str:
        """Serialize the report to a JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "EvaluationReport":
        """Reconstruct EvaluationReport from dictionary representation."""
        d = dict(data)
        d["holdout_metrics"] = WindowReport.from_dict(d["holdout_metrics"])
        d["in_sample_metrics"] = WindowReport.from_dict(d["in_sample_metrics"])
        d["walk_forward_metrics"] = [WindowReport.from_dict(w) for w in d["walk_forward_metrics"]]
        # Ensure default None for Series fields if not present
        if "holdout_net_returns" not in d:
            d["holdout_net_returns"] = None
        if "benchmark_returns" not in d:
            d["benchmark_returns"] = None
        if "benchmark_ticker" not in d:
            d["benchmark_ticker"] = "SPY"
        return cls(**d)

    @classmethod
    def from_json(cls, json_str: str) -> "EvaluationReport":
        """Reconstruct EvaluationReport from a JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)
