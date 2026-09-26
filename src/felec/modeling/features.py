"""Feature construction for modeling, each one safe relative to a D-1 cutoff."""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd


def cutoff_instant_for_date(target_date: date, cutoff_hour: int = 12) -> pd.Timestamp:
    """UTC instant of cutoff_hour (Paris wall-clock time) on the day before target_date.

    Constructs the local instant directly (year/month/day/hour) rather than adding a
    fixed duration to midnight, so it stays correct across DST transitions -- adding
    a Timedelta to a tz-aware midnight would cross the skipped/repeated hour on
    transition days and land on the wrong wall-clock time.
    """
    d_minus_1 = target_date - timedelta(days=1)
    local_cutoff = pd.Timestamp(
        year=d_minus_1.year, month=d_minus_1.month, day=d_minus_1.day,
        hour=cutoff_hour, tz="Europe/Paris",
    )
    return local_cutoff.tz_convert("UTC")


def compute_lag_168h(df: pd.DataFrame) -> pd.Series:
    """Consumption exactly 168 hours (1 week) before each row, by UTC instant --
    not a positional shift, so it stays correct across the dataset's known small
    gaps (see PROJECT_NOTES.md's "Known data gaps"). Always safe regardless of the
    cutoff hour: 168 hours before any point on D-1 is always at least 6 full days
    before D-1 itself.
    """
    lagged_index = df.index - pd.Timedelta(hours=168)
    lagged = df["consommation"].reindex(lagged_index)
    lagged.index = df.index
    lagged.name = "lag_168h"
    return lagged
