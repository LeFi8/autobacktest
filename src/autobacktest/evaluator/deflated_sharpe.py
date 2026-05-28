"""Deflated Sharpe Ratio (DSR) using hierarchical returns clustering."""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from scipy.stats import norm


def _ledoit_wolf_correlation(returns_matrix: pd.DataFrame) -> pd.DataFrame:
    """Compute the shrunk correlation matrix using Ledoit-Wolf shrinkage to a scaled identity target.

    If the returns_matrix has T <= 1 or fails to compute, gracefully falls back
    to the standard empirical correlation matrix.
    """
    if returns_matrix.empty:
        return pd.DataFrame()

    T, p = returns_matrix.shape
    # Fallback to empirical correlation if insufficient observations or features
    if T <= 1 or p <= 1:
        return returns_matrix.corr().fillna(0.0).clip(-1.0, 1.0)

    try:
        # Standardize returns matrix (fill NaNs and center)
        X = returns_matrix.fillna(0.0).values
        X_centered = X - np.mean(X, axis=0)

        # Empirical covariance S
        S = (X_centered.T @ X_centered) / T

        # Constant variance target F = mu * I
        mu = np.trace(S) / p

        # Misspecification distance d^2 = sum((S - F)^2)
        d2 = np.sum((S - mu * np.eye(p)) ** 2)
        if d2 == 0.0:
            shrunk_cov = S
        else:
            # Estimate b^2 (variance of sample covariance elements)
            X2 = X_centered ** 2
            sum_t_x2_x2 = np.sum(X2.T @ X2)
            sum_S2 = np.sum(S ** 2)
            b2 = (sum_t_x2_x2 - T * sum_S2) / (T ** 2)

            # Shrinkage coefficient delta (clipped at d^2)
            b2 = min(b2, d2)
            delta = b2 / d2

            # Convex combination
            shrunk_cov = (1.0 - delta) * S
            np.fill_diagonal(shrunk_cov, shrunk_cov.diagonal() + delta * mu)

        # Convert covariance to correlation
        diag_cov = np.diag(shrunk_cov)
        std_devs = np.sqrt(diag_cov)
        # Handle zero-variance strategies gracefully
        std_devs[std_devs == 0.0] = 1.0

        shrunk_corr = shrunk_cov / np.outer(std_devs, std_devs)
        shrunk_corr = np.clip(shrunk_corr, -1.0, 1.0)
        np.fill_diagonal(shrunk_corr, 1.0)

        return pd.DataFrame(shrunk_corr, index=returns_matrix.columns, columns=returns_matrix.columns)

    except Exception:
        # Graceful fallback on unexpected error
        return returns_matrix.corr().fillna(0.0).clip(-1.0, 1.0)


def calculate_effective_trials(
    returns_matrix: pd.DataFrame, threshold: float = 0.5
) -> int:
    """Determine the number of independent strategy trials.

    Uses hierarchical clustering to group correlated returns.

    Args:
        returns_matrix: DataFrame of historical daily net returns.
        threshold: Distance cophenetic threshold to merge clusters.

    Returns:
        int: Number of independent clusters (effective trials N).
    """
    if returns_matrix.empty or returns_matrix.shape[1] <= 1:
        return int(max(returns_matrix.shape[1], 1))

    # Compute correlation matrix using Ledoit-Wolf shrinkage to stabilize calculations
    corr = _ledoit_wolf_correlation(returns_matrix)

    # Convert correlation to distance matrix: d_ij = sqrt(0.5 * (1 - rho_ij))
    dist = np.sqrt(0.5 * (1.0 - corr))

    # Force symmetry and zero diagonal using a writeable copy
    dist_val = dist.values.copy()
    np.fill_diagonal(dist_val, 0.0)
    dist_val = (dist_val + dist_val.T) / 2.0

    # Condense distance matrix for scipy
    try:
        condensed = squareform(dist_val, checks=False)
        z_linkage = linkage(condensed, method="complete")
        clusters = fcluster(z_linkage, t=threshold, criterion="distance")
        return len(np.unique(clusters))
    except Exception:
        # Fallback to absolute trials count on cluster failure
        return int(returns_matrix.shape[1])


def calculate_psr_dsr(
    observed_daily_returns: pd.Series,
    historical_sharpes: list[float] | None = None,
    effective_trials: int = 1,
) -> float:
    """Calculate the Deflated Sharpe Ratio (DSR) adjusting for effective trials.

    If effective_trials = 1, this calculates the Probabilistic Sharpe Ratio (PSR).

    Args:
        observed_daily_returns: Net returns series of the current strategy.
        historical_sharpes: Annualized Sharpe ratios of all tested trials.
        effective_trials: Calculated independent trials (N).

    Returns:
        float: DSR value (probability of true Sharpe > 0, bounded in [0, 1]).
    """
    n_days = len(observed_daily_returns)
    if n_days < 2:
        return 0.0

    # Calculate observed daily metrics
    mean = observed_daily_returns.mean()
    std = observed_daily_returns.std(ddof=1)
    if std == 0.0:
        return 0.0

    daily_sharpe = mean / std

    # Compute Skewness and Kurtosis
    skew = observed_daily_returns.skew()
    # Scipy/Pandas returns excess kurtosis (kurtosis - 3). We need absolute kurtosis.
    kurt = observed_daily_returns.kurtosis()
    if np.isnan(skew):
        skew = 0.0
    if np.isnan(kurt):
        kurt = 0.0
    abs_kurt = kurt + 3.0

    # Calculate expected maximum Sharpe ratio under the null hypothesis (SR0)
    sr0 = 0.0
    if effective_trials > 1 and historical_sharpes and len(historical_sharpes) > 1:
        # Calculate standard deviation of Sharpes across trials
        sigma_sr = float(np.std(historical_sharpes, ddof=1))
        if sigma_sr > 0.0:
            euler_gamma = 0.5772156649
            # Expected maximum of N standard normal variables approximation
            z_term_1 = norm.ppf(1.0 - 1.0 / effective_trials)
            z_term_2 = norm.ppf(1.0 - 1.0 / (effective_trials * np.e))
            # Handle infinity bounds in ppf using Extreme Value Theory (EVT)
            # asymptotic approximation
            if np.isinf(z_term_1):
                z_term_1 = np.sqrt(2.0 * np.log(effective_trials))
            if np.isinf(z_term_2):
                z_term_2 = np.sqrt(2.0 * np.log(effective_trials * np.e))

            max_z = (1.0 - euler_gamma) * z_term_1 + euler_gamma * z_term_2
            sr0 = sigma_sr * max_z

    # Compute variance of the daily Sharpe estimate
    # Var(SR) = (1 - skew*SR + (kurt-1)/4 * SR^2) / (T - 1)
    # Adjusted to daily basis for the test statistic denominator
    num_var = 1.0 - skew * daily_sharpe + (abs_kurt - 1.0) / 4.0 * (daily_sharpe**2)
    var_sharpe = max(num_var, 1e-8) / (n_days - 1)
    std_sharpe = np.sqrt(var_sharpe)

    # Convert SR0 to daily basis for comparison
    daily_sr0 = sr0 / np.sqrt(252)

    # Compute test statistic: t = (daily_sharpe - daily_sr0) / std_sharpe
    if std_sharpe == 0.0:
        return 0.0
    t_stat = (daily_sharpe - daily_sr0) / std_sharpe

    # Deflated Sharpe Ratio is the CDF of the standard normal distribution at t_stat
    return float(norm.cdf(t_stat))
