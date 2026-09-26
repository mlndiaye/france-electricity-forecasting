from datetime import date

import pandas as pd
import pytest

from felec.modeling.features import (
    FEATURE_COLUMNS,
    build_calendar_features,
    build_features,
    compute_lag_168h,
    compute_recent_trend,
    cutoff_instant_for_date,
)


def test_cutoff_instant_for_date_in_winter_is_utc_plus_1():
    # Noon Paris time in winter (CET, UTC+1) is 11:00 UTC. Target date is one day
    # after the date whose noon we're computing, since the cutoff is on D-1.
    result = cutoff_instant_for_date(date(2024, 2, 2), cutoff_hour=12)
    assert result == pd.Timestamp("2024-02-01T11:00:00", tz="UTC")


def test_cutoff_instant_for_date_in_summer_is_utc_plus_2():
    # Noon Paris time in summer (CEST, UTC+2) is 10:00 UTC.
    result = cutoff_instant_for_date(date(2024, 7, 2), cutoff_hour=12)
    assert result == pd.Timestamp("2024-07-01T10:00:00", tz="UTC")


def test_cutoff_instant_for_date_around_spring_dst_transition():
    # 2024-03-31 is the spring-forward transition in France (02:00 CET -> 03:00
    # CEST). D-1 for target_date=2024-04-01 is 2024-03-31 itself -- the transition
    # day. Noon that day is unambiguous (well after the 2am jump), but only correct
    # if computed by constructing the hour directly rather than adding a fixed
    # duration to midnight, which would cross the skipped hour and land one hour
    # off. Confirmed by reverting to the naive implementation: it returns
    # 2024-03-31T11:00:00Z instead of the correct 2024-03-31T10:00:00Z.
    result = cutoff_instant_for_date(date(2024, 4, 1), cutoff_hour=12)
    assert result == pd.Timestamp("2024-03-31T10:00:00", tz="UTC")


def test_cutoff_instant_for_date_rejects_ambiguous_hour():
    # cutoff_hour=2 lands exactly on the 2:00-2:59 wall-clock hour that's
    # nonexistent on the spring-forward day and ambiguous (occurs twice) on the
    # fall-back day. pd.Timestamp(..., hour=2, tz="Europe/Paris") resolves this
    # silently instead of raising, so the function must guard it explicitly.
    with pytest.raises(ValueError, match="DST transition"):
        cutoff_instant_for_date(date(2024, 4, 1), cutoff_hour=2)


def _hourly_index(n_hours: int, start="2024-01-01") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n_hours, freq="h", tz="UTC")


def test_compute_lag_168h_looks_up_by_time_not_position():
    # 300 hourly rows, value == row number, so "168h before row N" should equal
    # value N-168 whenever that timestamp exists in the index.
    index = _hourly_index(300)
    df = pd.DataFrame({"consommation": range(300)}, index=index)

    result = compute_lag_168h(df)

    assert result.iloc[280] == 112  # 280 - 168
    assert pd.isna(result.iloc[100])  # nothing exists 168h before row 100 (< 168)


def test_compute_lag_168h_is_gap_safe_not_a_positional_shift():
    # Drop row 150 (a gap strictly between row 112 and row 280). A naive
    # `.shift(168)` on the gapped array would look back 168 *rows*, landing one row
    # short of the correct answer because of the missing row in between. A correct
    # time-based lookup is unaffected, since the gap sits outside the [112, 280]
    # window this specific lookup needs.
    index = _hourly_index(300)
    df = pd.DataFrame({"consommation": range(300)}, index=index)
    gapped = df.drop(df.index[150])

    result = compute_lag_168h(gapped)

    target_row = gapped.index[gapped.index == index[280]]
    assert result.loc[target_row[0]] == 112
    # Sanity check that a naive positional shift would have gotten this wrong:
    positional_shift_answer = gapped["consommation"].shift(168).loc[target_row[0]]
    assert positional_shift_answer != 112


def test_compute_recent_trend_averages_the_24h_window_ending_at_cutoff():
    # Cutoff for target date 2024-02-03 is 2024-02-02 at 11:00 UTC (noon Paris,
    # winter). The 24h window ending there is 2024-02-01T12:00 -> 2024-02-02T11:00
    # UTC inclusive. Build a df spanning several days with a known, distinct value
    # per hour so the window average is easy to check by hand.
    index = _hourly_index(24 * 5, start="2024-02-01T00:00")  # 5 days from Feb 1 00:00 UTC
    df = pd.DataFrame({"consommation": range(24 * 5)}, index=index)

    result = compute_recent_trend(df, cutoff_hour=12)

    window = df.loc["2024-02-01T12:00":"2024-02-02T11:00", "consommation"]
    assert len(window) == 24
    expected = window.mean()

    target_rows = result[
        index.tz_convert("Europe/Paris").normalize()
        == pd.Timestamp("2024-02-03", tz="Europe/Paris")
    ]
    assert (target_rows == expected).all()


def test_compute_recent_trend_is_nan_when_window_entirely_missing():
    # The very first day of a dataset has no 24h window before its own cutoff.
    index = _hourly_index(5, start="2024-02-01T00:00")
    df = pd.DataFrame({"consommation": range(5)}, index=index)

    result = compute_recent_trend(df, cutoff_hour=12)

    assert result.isna().all()


def test_build_calendar_features_uses_paris_local_time():
    # 2024-02-01T23:00 UTC is 2024-02-02T00:00 in Paris (winter, UTC+1) -- a
    # different calendar hour/day than the raw UTC value, which is exactly the bug
    # already found and fixed once in notebooks/exploration.ipynb.
    index = pd.DatetimeIndex(["2024-02-01T23:00:00+00:00"])
    df = pd.DataFrame({"consommation": [100]}, index=index)

    result = build_calendar_features(df)

    assert result.iloc[0]["hour"] == 0
    assert result.iloc[0]["day_of_week"] == 4  # Friday, Feb 2 2024


def test_build_features_combines_all_columns():
    index = _hourly_index(24 * 10, start="2024-02-01T00:00")
    df = pd.DataFrame(
        {
            "consommation": range(24 * 10),
            "temperature_nationale": [10.0] * (24 * 10),
            "est_ferie": [False] * (24 * 10),
            "vacances_zone_a": [False] * (24 * 10),
            "vacances_zone_b": [False] * (24 * 10),
            "vacances_zone_c": [False] * (24 * 10),
        },
        index=index,
    )

    result = build_features(df)

    assert list(result.columns) == FEATURE_COLUMNS
    assert len(result) == len(df)
