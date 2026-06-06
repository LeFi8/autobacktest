"""Stress testing across macro economic crash regimes."""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REGIMES = {
    "2008_GFC": ("2008-09-01", "2009-03-31", 0.25),  # Max 25% drawdown threshold
    "2020_COVID": ("2020-02-20", "2020-04-30", 0.15),  # Max 15% drawdown threshold
    "2022_BEAR": ("2022-01-01", "2022-12-31", 0.20),  # Max 20% drawdown threshold
}

# Minimum-exposure check thresholds
MAX_CASH_RATIO = 0.80  # >80% cash triggers the warning
MAX_CASH_CONSECUTIVE_DAYS = 10  # sustained for >10 trading days
MIN_TICKERS_FOR_REJECT = 3  # only hard-reject when universe >= 3 tickers


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate the maximum drawdown from an equity curve series."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    return float(abs(drawdowns.min()))


def _check_regime_exposure(
    daily_weights: pd.DataFrame,
    regime_start: str,
    regime_end: str,
    n_tickers: int,
) -> str | None:
    """Check for excessive cash holding during a regime period.

    Returns:
        A warning string if the condition is triggered, or ``None``.
    """
    regime_weights = daily_weights.loc[regime_start:regime_end]
    if regime_weights.empty:
        return None

    exposure = regime_weights.sum(axis=1)
    low_exposure = (exposure < 1.0 - MAX_CASH_RATIO).astype(int)

    # Find the longest consecutive run of low-exposure days
    transitions = np.diff(low_exposure, prepend=0)
    run_starts = np.where(transitions == 1)[0]
    run_ends = np.where(transitions == -1)[0]
    if len(run_starts) > 0:
        # Handle case where the series ends in a low-exposure state
        if len(run_ends) < len(run_starts):
            run_ends = np.append(run_ends, len(low_exposure))
        longest_run = int(np.max(run_ends - run_starts))
    else:
        longest_run = 0

    if longest_run > MAX_CASH_CONSECUTIVE_DAYS:
        msg = (
            f"Regime {regime_start}..{regime_end}: strategy held >{MAX_CASH_RATIO * 100:.0f}% cash "
            f"for {longest_run} consecutive trading days (limit: {MAX_CASH_CONSECUTIVE_DAYS})."
        )
        # Hard-reject if the universe is genuinely multi-asset
        if n_tickers >= MIN_TICKERS_FOR_REJECT:
            return f"[HARD REJECT] {msg}"
        return f"[WARNING] {msg}"
    return None


def evaluate_stress_regimes(
    net_returns: pd.Series,
    daily_weights: pd.DataFrame | None = None,
    n_tickers: int = 0,
) -> tuple[dict[str, float], bool]:
    """Calculate drawdowns during stress regimes and return passed indicator.

    When ``daily_weights`` is provided an additional minimum-exposure check
    is performed — sustained >80% cash positions during a crisis regime
    are flagged and may produce a hard rejection for genuinely multi-asset
    strategies (universe >= 3 tickers).

    Args:
        net_returns: Daily portfolio net returns.
        daily_weights: Optional daily portfolio weights (used for
            minimum-exposure check).
        n_tickers: Number of tickers in the strategy universe (used for
            exposure-reject logic).

    Returns:
        tuple containing:
            - Dict mapping regime name to max drawdown observed (float)
            - Boolean verdict indicating if all thresholds were satisfied
    """
    if net_returns.empty:
        return {}, True

    # Calculate equity curve of the sub-window
    drawdowns = {}
    passed = True
    any_overlap = False

    for name, (start, end, limit) in REGIMES.items():
        regime_ret = net_returns.loc[start:end]

        if not regime_ret.empty:
            any_overlap = True
            # Reconstruct sub-equity curve
            sub_equity = (1.0 + regime_ret).cumprod()
            max_dd = calculate_max_drawdown(sub_equity)
            drawdowns[name] = max_dd
            if max_dd > limit:
                passed = False

            # Minimum-exposure check
            if daily_weights is not None:
                exposure_warning = _check_regime_exposure(daily_weights, start, end, n_tickers)
                if exposure_warning:
                    if exposure_warning.startswith("[HARD REJECT]"):
                        logger.warning("Regime exposure: %s", exposure_warning)
                        passed = False
                    else:
                        logger.warning("Regime exposure: %s", exposure_warning)
        else:
            # If data doesn't overlap the regime, we assume 0 drawdown and pass
            logger.warning(
                "Strategy backtest period does not overlap with stress testing "
                "regime %s (%s to %s). Drawdown scored as 0.0.",
                name,
                start,
                end,
            )

            drawdowns[name] = 0.0

    if not any_overlap:
        logger.warning(
            "Strategy backtest period does not overlap with ANY crash regimes. Regimes verdict is marked as failed."
        )
        passed = False

    return drawdowns, passed


def calculate_regime_haircut(
    benchmark_prices: pd.Series,
    launch_date: pd.Timestamp,
) -> float:
    """Calculate the Liu timing haircut based on benchmark z-score at launch_date.

    Args:
        benchmark_prices: Historical daily prices of the benchmark (e.g. SPY).
        launch_date: The date at which the strategy is launched (start of holdout).

    Returns:
        float: The haircut percentage to apply (e.g., 0.05 * z_score).
    """
    if benchmark_prices.empty or len(benchmark_prices) < 252:
        return 0.0

    # Calculate rolling 252-day returns (price_t / price_{t-252} - 1)
    rolling_returns = benchmark_prices.pct_change(252).dropna()
    if rolling_returns.empty:
        return 0.0

    # Find the rolling return at or closest prior to the launch_date
    historical_returns = rolling_returns.loc[:launch_date]
    if historical_returns.empty:
        return 0.0

    launch_val = historical_returns.iloc[-1]

    # Compute z-score relative to the historical rolling distribution up to launch_date
    mean_val = historical_returns.mean()
    std_val = historical_returns.std(ddof=1)

    z_score = 0.0 if std_val == 0.0 or np.isnan(std_val) else (launch_val - mean_val) / std_val

    if z_score > 0.0:
        return float(0.05 * z_score)
    return 0.0
