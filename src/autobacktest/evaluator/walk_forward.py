"""Walk-forward rolling calendar generator.

Generates date windows for walk-forward validation.  Each window has a
training segment (default 5 years) followed immediately by a testing segment
(default 1 year).  Windows roll forward by ``step_years`` (default 1) to
produce multiple folds.  This avoids lookahead bias by ensuring the test
segment always follows its training segment in calendar time.
"""

import pandas as pd


def generate_walk_forward_windows(
    index: pd.DatetimeIndex,
    train_years: int = 5,
    test_years: int = 1,
    step_years: int = 1,
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """Generate rolling walk-forward train/test date window tuples.

    Produces a list of ``(train_start, train_end, test_start, test_end)``
    windows.  The union of all test segments covers the entire in-sample
    period.  Windows that extend beyond the available data are truncated;
    windows with empty train or test sets are omitted.

    Args:
        index: Full DatetimeIndex of available trading days.
        train_years: Number of years in each training window.
        test_years: Number of years in each testing window.
        step_years: Calendar years to advance per window.

    Returns:
        list of ``(train_start, train_end, test_start, test_end)`` tuples.
        Returns an empty list when the index is empty or too short for at
        least one complete window.
    """
    if index.empty:
        return []

    windows = []
    min_date = index.min()
    max_date = index.max()

    current_train_start = min_date

    while True:
        # Calculate train end date
        current_train_end = current_train_start + pd.DateOffset(years=train_years)

        # Check if train window goes beyond the available data
        if current_train_end >= max_date:
            break

        # Calculate test end date
        current_test_start = current_train_end + pd.Timedelta(days=1)
        current_test_end = current_test_start + pd.DateOffset(years=test_years)

        # Truncate if the test window goes beyond available data
        if current_test_end > max_date:
            current_test_end = max_date

        # Only append if we have a valid test window
        if current_test_start < max_date:
            aligned_train = index[(index >= current_train_start) & (index <= current_train_end)]
            aligned_test = index[(index >= current_test_start) & (index <= current_test_end)]

            if not aligned_train.empty and not aligned_test.empty:
                windows.append(
                    (
                        aligned_train.min(),
                        aligned_train.max(),
                        aligned_test.min(),
                        aligned_test.max(),
                    )
                )

        # Step forward
        current_train_start = current_train_start + pd.DateOffset(years=step_years)

    return windows
