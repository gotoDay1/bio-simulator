"""Loader for a multi-step feed-rate profile CSV (段階的流加/stepwise fed-batch feeding).

Distinct from experimental_data.py's measured-data schema: this CSV defines a
simulation *input* (F_in(t) breakpoints), not measured output to compare against.
The CSV has two required columns: `time` (h) and `feed_rate` (L/h). Rows are sorted
by `time` ascending before building the profile, so the file's row order doesn't
matter. Interpolation between breakpoints is step-hold (piecewise constant) - see
biosim.operation_modes.stepwise_feed for the exact before-first/after-last edge-case
semantics.
"""

from collections.abc import Callable

import pandas as pd

from biosim.exceptions import FeedProfileError
from biosim.operation_modes import stepwise_feed

TIME_COLUMN = "time"
RATE_COLUMN = "feed_rate"


def load_feed_profile_csv(path_or_buffer) -> Callable[[float], float]:
    """Load a multi-step feed-rate profile CSV and build a step-hold F_in(t) callable.

    Raises FeedProfileError if the `time`/`feed_rate` columns are missing or no valid
    rows remain after dropping NaNs. Raises InvalidParameterError (from stepwise_feed)
    for duplicate/non-increasing timestamps or negative feed rates.
    """
    df = pd.read_csv(path_or_buffer)
    missing = [c for c in (TIME_COLUMN, RATE_COLUMN) if c not in df.columns]
    if missing:
        raise FeedProfileError(
            f"Feed profile CSV is missing required column(s) {missing}. "
            f"Found columns {list(df.columns)}."
        )

    df = df[[TIME_COLUMN, RATE_COLUMN]].dropna()
    if df.empty:
        raise FeedProfileError("Feed profile CSV has no valid (time, feed_rate) rows.")

    df = df.sort_values(TIME_COLUMN).reset_index(drop=True)
    return stepwise_feed(df[TIME_COLUMN].tolist(), df[RATE_COLUMN].tolist())
