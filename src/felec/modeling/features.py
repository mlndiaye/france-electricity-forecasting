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

    France's transitions both pivot exactly on the 2:00-2:59 wall-clock hour: on the
    spring-forward day that hour never happens (clocks jump straight from 01:59:59 to
    03:00:00), and on the fall-back day it happens twice (once CEST, once CET). Either
    way, `pd.Timestamp(..., hour=2, tz="Europe/Paris")` resolves silently to one
    arbitrary offset instead of raising -- so cutoff_hour=2 is rejected explicitly.
    The 3:00-3:59 hour is unaffected by both transitions (it's the first, unambiguous
    hour after the spring jump, and the second, unambiguous hour after the fall
    repeat), so it is not guarded.
    """
    if cutoff_hour == 2:
        raise ValueError(
            f"cutoff_hour={cutoff_hour} falls within France's DST transition window "
            "(2:00-2:59 local time can be nonexistent or ambiguous on transition days) "
            "-- choose a different hour."
        )
    d_minus_1 = target_date - timedelta(days=1)
    local_cutoff = pd.Timestamp(
        year=d_minus_1.year, month=d_minus_1.month, day=d_minus_1.day,
        hour=cutoff_hour, tz="Europe/Paris",
    )
    return local_cutoff.tz_convert("UTC")


def compute_lag_168h(df: pd.DataFrame) -> pd.Series:
    """Consumption exactly 168 hours (1 week) before each row, by UTC instant --
    not a positional shift, so it stays correct across the dataset's known small
    gaps (see PROJECT_NOTES.md's "Known data gaps"). 168 hours (7 days) comfortably
    exceeds the largest possible gap (at most ~48h) between any hour of the target
    day D and any cutoff instant on D-1, for any cutoff_hour in [0, 23] outside the
    guarded DST window, so this lag is always safely computable before the cutoff.
    """
    lagged_index = df.index - pd.Timedelta(hours=168)
    lagged = df["consommation"].reindex(lagged_index).set_axis(df.index)
    lagged.name = "lag_168h"
    return lagged
