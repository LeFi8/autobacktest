"""Monte Carlo bootstrap of return series for Sharpe ratio significance.

Supports two bootstrapping methods:

- **circular** — circular block bootstrap (default).  Pads the return
  array with its first ``block_size`` elements so every position has
  equal probability of being a block start.
- **stationary** — Politis-Romano stationary bootstrap with geometrically
  distributed block lengths.

Returns 5th, 50th, and 95th percentile Sharpe ratios along with the
full array of bootstrapped Sharpes for histogram plotting.
"""

import numpy as np
import pandas as pd


def run_block_bootstrap(
    returns: pd.Series,
    n_paths: int = 10000,
    block_size: int = 21,
    seed: int | None = None,
    method: str = "circular",
) -> tuple[float, float, float, np.ndarray]:
    """Execute block bootstrap to yield 5th, 50th, 95th Sharpe percentiles.

    Preserves short-term serial correlation by grouping returns into blocks.

    Args:
        returns: Daily net returns series.
        n_paths: Number of bootstrap paths.
        block_size: Size of contiguous return blocks in days (or expected block size for stationary).
        seed: Seed for random number generator.
        method: Bootstrap method ``"circular"`` (default) or ``"stationary"``.

    Returns:
        tuple: (5th_percentile_sharpe, 50th_percentile_sharpe, 95th_percentile_sharpe, all_sharpes)
    """
    if method not in ("circular", "stationary"):
        raise ValueError(f"Unknown bootstrap method: {method}")

    if returns.empty or len(returns) < block_size:
        return 0.0, 0.0, 0.0, np.array([])

    ret_arr = returns.values
    n_samples = len(ret_arr)

    rng = np.random.default_rng(seed)

    if method == "circular":
        # Pad the array with the first ``block_size`` values so that every
        # element has an equal chance of being a block start (circular
        # block bootstrap).  This eliminates the end-sampling bias where
        # the final ``block_size - 1`` positions can never start a block.
        pad = min(block_size, n_samples)
        ret_arr_padded = np.concatenate([ret_arr, ret_arr[:pad]])

        n_blocks = int(np.ceil(n_samples / block_size))

        # The padded array has ``n_padded = n_samples + pad`` elements;
        # permissible start indices cover the entire original span.
        max_start = n_samples
        if max_start <= 0:
            return 0.0, 0.0, 0.0, np.array([])

        # Draw random block starting indices across paths in a vectorized matrix
        starts = rng.integers(0, max_start, size=(n_paths, n_blocks))

        # Construct the array indices mapping
        indices = starts[:, :, np.newaxis] + np.arange(block_size)
        indices = indices.reshape(n_paths, -1)[:, :n_samples]

        # Index into the raw returns array (padded, so all indices are valid)
        boot_returns = ret_arr_padded[indices]

    else:  # stationary
        # Politis-Romano stationary bootstrap
        # Geometric block lengths: expected block size is block_size
        p = 1.0 / block_size
        restart = rng.random((n_paths, n_samples)) < p
        restart[:, 0] = True  # Always start with a fresh block

        fresh = rng.integers(0, n_samples, size=(n_paths, n_samples))

        idx = np.zeros((n_paths, n_samples), dtype=np.intp)
        idx[:, 0] = fresh[:, 0]
        for t in range(1, n_samples):
            idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % n_samples)

        boot_returns = ret_arr[idx]

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

    return p5, p50, p95, sharpes
