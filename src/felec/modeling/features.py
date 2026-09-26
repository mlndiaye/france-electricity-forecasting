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


def compute_recent_trend(df: pd.DataFrame, cutoff_hour: int = 12) -> pd.Series:
    """Mean consommation over the 24h window ending at the D-1 cutoff (Paris local
    time), broadcast to every hourly row of the target day D. Safe by construction:
    never references anything after the cutoff, regardless of which hour of D a row
    belongs to -- recovers the "recent conditions" signal a naive 24h-ago lag would
    have given, without that lag's per-target-hour safety problem (see ADR 0004).
    """
    local_dates = df.index.tz_convert("Europe/Paris").normalize()
    unique_dates = local_dates.unique()

    trend_by_date: dict[pd.Timestamp, float] = {}
    for local_midnight in unique_dates:
        cutoff_utc = cutoff_instant_for_date(local_midnight.date(), cutoff_hour)
        window_start_utc = cutoff_utc - pd.Timedelta(hours=23)
        trend_by_date[local_midnight] = df.loc[window_start_utc:cutoff_utc, "consommation"].mean()

    return pd.Series(
        local_dates.map(trend_by_date).to_numpy(), index=df.index, name="recent_trend"
    )


def build_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Hour of day, day of week, and month, all in Paris local time -- the daily/
    weekly demand pattern found in notebooks/exploration.ipynb only makes sense in
    local time, not UTC (same fix already applied there for the same reason).
    Known months in advance either way, so no leakage concern.
    """
    local_index = df.index.tz_convert("Europe/Paris")
    return pd.DataFrame(
        {
            "hour": local_index.hour,
            "day_of_week": local_index.dayofweek,
            "month": local_index.month,
        },
        index=df.index,
    )


FEATURE_COLUMNS = [
    "hour", "day_of_week", "month",
    "lag_168h", "recent_trend",
    "temperature_nationale", "est_ferie",
    "vacances_zone_a", "vacances_zone_b", "vacances_zone_c",
]


def build_features(df: pd.DataFrame, cutoff_hour: int = 12) -> pd.DataFrame:
    """Build the full feature set for modeling from the processed dataset.

    df must be indexed by UTC-aware date_heure, sorted, as produced by
    felec.processing.build_dataset.build_dataset().
    """
    features = build_calendar_features(df)
    features["lag_168h"] = compute_lag_168h(df)
    features["recent_trend"] = compute_recent_trend(df, cutoff_hour=cutoff_hour)
    features["temperature_nationale"] = df["temperature_nationale"]
    features["est_ferie"] = df["est_ferie"]
    features["vacances_zone_a"] = df["vacances_zone_a"]
    features["vacances_zone_b"] = df["vacances_zone_b"]
    features["vacances_zone_c"] = df["vacances_zone_c"]
    return features[FEATURE_COLUMNS]
