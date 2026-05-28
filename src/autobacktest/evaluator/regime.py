"""Stress testing across macro economic crash regimes."""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

REGIMES = {
    "2008_GFC": ("2008-09-01", "2009-03-31", 0.25),  # Max 25% drawdown threshold
    "2020_COVID": ("2020-02-20", "2020-04-30", 0.15),  # Max 15% drawdown threshold
    "2022_BEAR": ("2022-01-01", "2022-12-31", 0.20),  # Max 20% drawdown threshold
}


def calculate_max_drawdown(equity: pd.Series) -> float:
    """Calculate the maximum drawdown from an equity curve series."""
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdowns = (equity - running_max) / running_max
    return float(abs(drawdowns.min()))


def evaluate_stress_regimes(net_returns: pd.Series) -> tuple[dict[str, float], bool]:
    """Calculate drawdowns during stress regimes and return passed indicator.

    Args:
        net_returns: Daily portfolio net returns.

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
