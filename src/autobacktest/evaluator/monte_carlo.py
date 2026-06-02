"""Monte Carlo stationary/block bootstrap of return series."""

import numpy as np
import pandas as pd


def run_block_bootstrap(
    returns: pd.Series,
    n_paths: int = 10000,
    block_size: int = 21,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Execute block bootstrap to yield 5th, 50th, 95th Sharpe percentiles.

    Preserves short-term serial correlation by grouping returns into blocks.

    Args:
        returns: Daily net returns series.
        n_paths: Number of bootstrap paths.
        block_size: Size of contiguous return blocks in days.
        seed: Seed for random number generator.

    Returns:
        tuple: (5th_percentile_sharpe, 50th_percentile_sharpe, 95th_percentile_sharpe)
    """
    if returns.empty or len(returns) < block_size:
        return 0.0, 0.0, 0.0

    ret_arr = returns.values
    n_samples = len(ret_arr)

    # Pad the array with the first ``block_size`` values so that every
    # element has an equal chance of being a block start (circular
    # block bootstrap).  This eliminates the end-sampling bias where
    # the final ``block_size - 1`` positions can never start a block.
    pad = min(block_size, n_samples)
    ret_arr = np.concatenate([ret_arr, ret_arr[:pad]])

    n_blocks = int(np.ceil(n_samples / block_size))

    # The padded array has ``n_padded = n_samples + pad`` elements;
    # permissible start indices cover the entire original span.
    max_start = n_samples
    if max_start <= 0:
        return 0.0, 0.0, 0.0

    # Draw random block starting indices across paths in a vectorized matrix
    # using a local RNG generator for reproducibility

    rng = np.random.default_rng(seed)
    starts = rng.integers(0, max_start, size=(n_paths, n_blocks))

    # Construct the array indices mapping
    indices = starts[:, :, np.newaxis] + np.arange(block_size)
    indices = indices.reshape(n_paths, -1)[:, :n_samples]

    # Index into the raw returns array (padded, so all indices are valid)
    boot_returns = ret_arr[indices]

    # Calculate Sharpe Ratio across paths
    means = np.mean(boot_returns, axis=1)
    stds = np.std(boot_returns, axis=1, ddof=1)

    # Avoid zero division errors
    stds = np.where(stds == 0, 1e-8, stds)

    sharpes = np.sqrt(252) * (means / stds)

    # Return percentile boundaries
    p5 = float(np.percentile(sharpes, 5))
    p50 = float(np.percentile(sharpes, 50))
    p95 = float(np.percentile(sharpes, 95))

    return p5, p50, p95
