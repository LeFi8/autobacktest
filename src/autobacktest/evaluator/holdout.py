"""Out-of-sample holdout validation guards."""

import pandas as pd


def partition_holdout_data(
    index: pd.DatetimeIndex,
    holdout_years: int = 3,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split DatetimeIndex into in-sample and out-of-sample holdout indices.

    Args:
        index: Full sorted DatetimeIndex.
        holdout_years: Number of years to withhold for holdout.

    Returns:
        tuple containing:
            - In-sample index (pd.DatetimeIndex)
            - Holdout index (pd.DatetimeIndex)
    """
    if index.empty:
        return index, index

    max_date = index.max()
    holdout_cutoff = max_date - pd.DateOffset(years=holdout_years)

    in_sample_idx = index[index < holdout_cutoff]
    holdout_idx = index[index >= holdout_cutoff]

    return in_sample_idx, holdout_idx
