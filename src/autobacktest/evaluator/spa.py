"""Hansen's Superior Predictive Ability (SPA) test for multiple testing.

Calculates Consistent, Upper (conservative), and Lower (liberal) p-values
for a benchmark strategy against a set of alternative strategies.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_hansen_spa(
    benchmark_returns: pd.Series,
    alternative_returns: pd.DataFrame,
    n_paths: int = 1000,
    block_size: int = 21,
    seed: int = 42,
) -> dict[str, float]:
    """Calculate Hansen's Superior Predictive Ability (SPA) test p-values.

    Args:
        benchmark_returns: Daily returns of the benchmark strategy (iteration 0).
        alternative_returns: Daily returns of the alternative strategies (iterations > 0).
        n_paths: Number of stationary bootstrap paths to generate.
        block_size: Expected block size for the Politis-Romano stationary bootstrap.
        seed: Random seed for reproducibility.

    Returns:
        dict: A dictionary containing:
            - 'p_consistent': The consistent p-value (standard test result).
            - 'p_upper': The upper conservative p-value bound.
            - 'p_lower': The lower liberal p-value bound.
            - 't_spa': The observed SPA test statistic.
    """
    if alternative_returns.empty:
        raise ValueError("No alternative returns provided for Hansen SPA test.")
    if benchmark_returns.empty:
        raise ValueError("Benchmark returns are empty.")
    if n_paths < 2:
        raise ValueError("Number of bootstrap paths (n_paths) must be at least 2.")

    # 1. Align dates of benchmark and alternative returns using an inner join
    aligned = pd.concat(
        [benchmark_returns.to_frame("benchmark"), alternative_returns],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) < 2:
        raise ValueError(f"Insufficient aligned trading days ({len(aligned)}) for Hansen SPA test.")

    benchmark_aligned = aligned["benchmark"]
    alternative_aligned = aligned.drop(columns=["benchmark"])

    t_len = len(aligned)

    # 2. Compute active returns d_{k, t} = X_{k, t} - Y_t
    d = alternative_aligned.sub(benchmark_aligned, axis=0).values  # Shape: (T, M)

    # 3. Compute sample mean performance differences \bar{d}_k
    d_bar = np.mean(d, axis=0)  # Shape: (M,)

    # 4. Perform stationary bootstrap on the differences matrix along the time axis
    rng = np.random.default_rng(seed)
    p = 1.0 / block_size
    restart = rng.random((n_paths, t_len)) < p
    restart[:, 0] = True  # Always start with a fresh block

    fresh = rng.integers(0, t_len, size=(n_paths, t_len))
    idx = np.zeros((n_paths, t_len), dtype=np.intp)
    idx[:, 0] = fresh[:, 0]
    for t in range(1, t_len):
        idx[:, t] = np.where(restart[:, t], fresh[:, t], (idx[:, t - 1] + 1) % t_len)

    # Index into d to get shape (n_paths, t_len, M)
    d_boot = d[idx]

    # Compute bootstrap mean difference \bar{d}^*_k
    d_bar_boot = np.mean(d_boot, axis=1)  # Shape: (n_paths, M)

    # 5. Compute bootstrap standard errors \omega_k
    omega = np.std(d_bar_boot, axis=0, ddof=1)  # Shape: (M,)
    omega = np.where(omega == 0, 1e-8, omega)

    # 6. Calculate the actual Hansen SPA test statistic T^{SPA}
    t_spa = max(0.0, float(np.max(d_bar / omega)))

    # 7. Compute Hansen's three centered means
    # A_k = \omega_k * \sqrt{2 * \log \log T}
    # For very small t_len, log(T) can be less than 1, so log(log(T)) can be negative or undefined.
    # Enforce minimum value of 1.0001 for log_t = log(t_len) so log(log_t) is always defined and positive.
    log_t = max(1.0001, np.log(t_len))
    a_k = omega * np.sqrt(2.0 * np.log(log_t))

    # Consistent: \mu_k^c = \bar{d}_k * I({\bar{d}_k >= -a_k})
    mu_c = d_bar * (d_bar >= -a_k)
    # Upper Bound: \mu_k^u = \bar{d}_k
    mu_u = d_bar
    # Lower Bound: \mu_k^l = \max(0, \bar{d}_k)
    mu_l = np.maximum(0.0, d_bar)

    # 8. Calculate bootstrap test statistics for each centering option
    # T^*_i = \max(0, \max_k (\bar{d}^*_{k, b} - \mu_k^i) / \omega_k)
    t_boot_c = np.maximum(0.0, np.max((d_bar_boot - mu_c) / omega, axis=1))
    t_boot_u = np.maximum(0.0, np.max((d_bar_boot - mu_u) / omega, axis=1))
    t_boot_l = np.maximum(0.0, np.max((d_bar_boot - mu_l) / omega, axis=1))

    # 9. Return p-values: fraction of paths where T^*_i >= T^{SPA}
    p_consistent = float(np.mean(t_boot_c >= t_spa))
    p_upper = float(np.mean(t_boot_u >= t_spa))
    p_lower = float(np.mean(t_boot_l >= t_spa))

    # Hansen's p-values must satisfy: p_lower <= p_consistent <= p_upper
    # Due to floating point differences, let's clamp them to be safe
    p_lower = min(p_lower, p_consistent)
    p_upper = max(p_upper, p_consistent)

    return {
        "p_consistent": p_consistent,
        "p_upper": p_upper,
        "p_lower": p_lower,
        "t_spa": t_spa,
    }
