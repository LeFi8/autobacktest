"""Combinatorially Symmetric Cross-Validation (CSCV) for Probability of Backtest Overfitting (PBO)."""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def calculate_pbo(returns_matrix: pd.DataFrame, n_blocks: int = 10, embargo_days: int = 0) -> float | None:
    """Calculate the Probability of Backtest Overfitting (PBO) using CSCV.

    Args:
        returns_matrix: DataFrame where each column is the daily net returns of a trial,
            and rows are trading dates.
        n_blocks: Number of blocks to split the returns matrix into (default 10).
        embargo_days: Number of trailing days to drop from each block to avoid boundary autocorrelation.

    Returns:
        float | None: The Probability of Backtest Overfitting (PBO) in [0, 1], or None if uncomputable.
    """
    n_trials = returns_matrix.shape[1]
    n_days = len(returns_matrix)

    if n_trials <= 1:
        return None

    # Tighten validity check: fallback to embargo_days=0 if embargo consumes too much data
    effective_days = n_days - n_blocks * embargo_days
    if effective_days < 2 * n_blocks:
        embargo_days = 0
        effective_days = n_days
        if effective_days < 2 * n_blocks:
            return None

    # Convert to numpy array for performance
    returns_arr = returns_matrix.values

    # 1. Partition rows into n_blocks contiguous blocks using array_split
    #    to distribute remainder rows evenly (no single block absorbs all excess).
    blocks = np.array_split(returns_arr, n_blocks, axis=0)

    if embargo_days > 0:
        embargoed_blocks = []
        for b in blocks:
            keep = max(0, len(b) - embargo_days)
            embargoed_blocks.append(b[:keep])
        blocks = embargoed_blocks

    # 2. Generate all C(S, S/2) combinations of block splits
    is_size = n_blocks // 2
    block_indices = list(range(n_blocks))
    splits = list(itertools.combinations(block_indices, is_size))

    def get_annualized_sharpe(arr: np.ndarray) -> np.ndarray:
        mean_ret = np.mean(arr, axis=0)
        std_ret = np.std(arr, axis=0, ddof=1)
        # Handle zero-volatility gracefully
        sharpe = np.zeros(arr.shape[1])
        valid = (std_ret > 0.0) & (~np.isnan(std_ret))
        # Ensure we don't divide by zero/nan
        sharpe[valid] = (mean_ret[valid] / std_ret[valid]) * np.sqrt(252.0)
        return sharpe

    overfitted_count = 0
    total_splits = len(splits)

    for is_indices in splits:
        oos_indices = [idx for idx in block_indices if idx not in is_indices]

        # Concatenate blocks to form IS and OOS datasets using numpy
        is_arr = np.concatenate([blocks[idx] for idx in is_indices], axis=0)
        oos_arr = np.concatenate([blocks[idx] for idx in oos_indices], axis=0)

        # Compute IS and OOS Sharpes for all strategies
        is_sharpes = get_annualized_sharpe(is_arr)
        oos_sharpes = get_annualized_sharpe(oos_arr)

        # Winner in IS
        winner_idx = int(np.argmax(is_sharpes))

        # Relative rank of IS-winner in OOS among all strategies
        ranks = rankdata(oos_sharpes) - 1.0
        winner_rank = float(ranks[winner_idx] / (n_trials - 1.0))

        if winner_rank < 0.5:
            overfitted_count += 1

    return float(overfitted_count / total_splits)
