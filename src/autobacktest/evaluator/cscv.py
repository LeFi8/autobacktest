"""Combinatorially Symmetric Cross-Validation (CSCV) for Probability of Backtest Overfitting (PBO)."""

import itertools

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def calculate_pbo(returns_matrix: pd.DataFrame, n_blocks: int = 10) -> float:
    """Calculate the Probability of Backtest Overfitting (PBO) using CSCV.

    Args:
        returns_matrix: DataFrame where each column is the daily net returns of a trial,
            and rows are trading dates.
        n_blocks: Number of blocks to split the returns matrix into (default 10).

    Returns:
        float: The Probability of Backtest Overfitting (PBO) in [0, 1].
    """
    n_trials = returns_matrix.shape[1]
    n_days = len(returns_matrix)

    if n_trials <= 1 or n_days < 2 * n_blocks:
        return 0.0

    # Convert to numpy array for performance
    returns_arr = returns_matrix.values

    # 1. Partition rows into n_blocks contiguous blocks
    block_size = n_days // n_blocks
    blocks = []
    for i in range(n_blocks):
        start_idx = i * block_size
        # The last block gets the remainder rows
        end_idx = (i + 1) * block_size if i < n_blocks - 1 else n_days
        blocks.append(returns_arr[start_idx:end_idx])

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
