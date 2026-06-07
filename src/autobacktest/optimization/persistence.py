"""Database and git persistence helpers for strategy optimization."""

from __future__ import annotations

from typing import Any

import pandas as pd

from autobacktest.evaluator.cscv import calculate_pbo
from autobacktest.evaluator.deflated_sharpe import calculate_effective_trials, calculate_psr_dsr
from autobacktest.evaluator.report import EvaluationReport
from autobacktest.ledger.store import LedgerStore


def deflate_selection(
    report: EvaluationReport,
    selection_returns: pd.Series[Any],
    ledger: LedgerStore,
    exclude_id: int | None = None,
    cscv_blocks: int = 10,
    embargo_days: int = 5,
) -> None:
    """Deflate the in-sample selection DSR using the ledger's multi-trial history.

    The candidate is deliberately included in both the returns matrix and
    the historical Sharpe list. This is intentionally conservative.
    """
    hist_matrix, hist_sharpes = ledger.fetch_historical_returns(report.dataset_hash, exclude_id=exclude_id)

    if hist_matrix.empty:
        hist_matrix = pd.DataFrame({"candidate": selection_returns})
    else:
        hist_matrix = hist_matrix.copy()
        hist_matrix["candidate"] = selection_returns

    sharpes = list(hist_sharpes) if hist_sharpes is not None else []
    sharpes.append(report.observed_sharpe)

    n = max(1, calculate_effective_trials(hist_matrix))

    report.effective_trials = n
    report.deflated_sharpe = calculate_psr_dsr(selection_returns, sharpes, n)

    # Compute and store PBO (Probability of Backtest Overfitting)
    if len(hist_matrix) >= 2 * cscv_blocks:
        report.pbo = calculate_pbo(hist_matrix, n_blocks=cscv_blocks, embargo_days=embargo_days)
    else:
        report.pbo = None


def deflate_holdout(
    report: EvaluationReport,
    ledger: LedgerStore,
    exclude_id: int | None = None,
) -> None:
    """Deflate ``report.holdout_deflated_sharpe`` by the holdout-peek count.

    Same conservative self-inclusion rationale as ``deflate_selection``.
    """
    hist_matrix, hist_sharpes = ledger.fetch_holdout_history(report.dataset_hash, exclude_id=exclude_id)

    if report.holdout_net_returns is None or report.holdout_net_returns.empty:
        return

    if hist_matrix.empty:
        hist_matrix = pd.DataFrame({"candidate": report.holdout_net_returns})
    else:
        hist_matrix = hist_matrix.copy()
        hist_matrix["candidate"] = report.holdout_net_returns

    sharpes = list(hist_sharpes) if hist_sharpes is not None else []
    sharpes.append(report.holdout_metrics.sharpe_ratio)

    n = max(1, calculate_effective_trials(hist_matrix))

    report.holdout_deflated_sharpe = calculate_psr_dsr(
        report.holdout_net_returns,
        sharpes,
        n,
    )
