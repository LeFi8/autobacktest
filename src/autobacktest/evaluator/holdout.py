"""Out-of-sample holdout validation guards.

Partitions the price date index into an in-sample training period and an
out-of-sample holdout period.  The holdout is strictly reserved for the
``confirm`` gate — it is never shown to the LLM or used during selection.
Holdout length is configurable via ``AUTOBACKTEST_DEFAULT_HOLDOUT_YEARS``
(default 3 years).
"""

import pandas as pd


def partition_holdout_data(
    index: pd.DatetimeIndex,
    holdout_years: int = 3,
) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Split a sorted DatetimeIndex into in-sample and holdout segments.

    The last ``holdout_years`` of data are reserved as out-of-sample.
    The remainder is used for walk-forward training and testing.

    Args:
        index: Full sorted DatetimeIndex of trading days.
        holdout_years: Number of years to reserve for OOS validation.

    Returns:
        ``(in_sample_index, holdout_index)``.  Either index may be empty
        if the date range is shorter than the holdout period.
    """
    if index.empty:
        return index, index

    max_date = index.max()
    holdout_cutoff = max_date - pd.DateOffset(years=holdout_years)

    in_sample_idx = index[index < holdout_cutoff]
    holdout_idx = index[index >= holdout_cutoff]

    return in_sample_idx, holdout_idx
