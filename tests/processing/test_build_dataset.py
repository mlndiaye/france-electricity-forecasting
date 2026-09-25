from unittest.mock import patch

import pandas as pd
import pytest

from felec.processing.build_dataset import (
    aggregate_rte_hourly,
    build_dataset,
    combine_cities_weighted,
    expand_calendar_to_hourly,
)


def test_aggregate_rte_hourly_averages_quarter_hours():
    df = pd.DataFrame(
        {
            "date_heure": [
                "2024-02-01T08:00:00+00:00",
                "2024-02-01T08:15:00+00:00",
                "2024-02-01T08:30:00+00:00",
                "2024-02-01T08:45:00+00:00",
            ],
            "consommation": [51800, 52400, 53100, 53700],
            "prevision_j1": [52000, 52500, 53000, 53500],
            "prevision_j": [51900, 52300, 52900, 53400],
        }
    )

    result = aggregate_rte_hourly(df)

    assert len(result) == 1
    assert result.iloc[0]["consommation"] == 52750.0


def test_combine_cities_weighted_sums_contributions():
    city_frames = {
        "paris": pd.DataFrame({"datetime": ["2024-02-01T08:00"], "temperature": [10.0]}),
        "lyon": pd.DataFrame({"datetime": ["2024-02-01T08:00"], "temperature": [5.0]}),
    }

    with patch(
        "felec.processing.build_dataset.CITIES",
        {"paris": (0, 0, 8_000_000), "lyon": (0, 0, 2_000_000)},
    ):
        result = combine_cities_weighted(city_frames)

    expected = 10.0 * 0.8 + 5.0 * 0.2
    assert result.iloc[0]["temperature_nationale"] == pytest.approx(expected)


def test_combine_cities_weighted_returns_nan_when_all_cities_missing():
    city_frames = {
        "paris": pd.DataFrame({"datetime": ["2024-02-01T08:00"], "temperature": [None]}),
        "lyon": pd.DataFrame({"datetime": ["2024-02-01T08:00"], "temperature": [None]}),
    }

    with patch(
        "felec.processing.build_dataset.CITIES",
        {"paris": (0, 0, 8_000_000), "lyon": (0, 0, 2_000_000)},
    ):
        result = combine_cities_weighted(city_frames)

    assert pd.isna(result.iloc[0]["temperature_nationale"])


def test_expand_calendar_to_hourly_flags_holiday_and_vacation():
    index = pd.to_datetime(
        ["2024-01-01T10:00:00+00:00", "2024-01-02T10:00:00+00:00"], utc=True
    )
    jours_feries = pd.DataFrame({"date": ["2024-01-01"], "nom": ["1er janvier"]})
    vacances = pd.DataFrame(
        {
            "zone": ["Zone A"],
            "date_debut": ["2023-12-22"],
            "date_fin": ["2024-01-07"],
            "description": ["Vacances de Noël"],
        }
    )

    result = expand_calendar_to_hourly(index, jours_feries, vacances)

    # `==` rather than `is`: these columns are pandas bool-dtype (as required
    # by the processed-dataset schema), so scalar access yields numpy.bool_,
    # which is never `is True`/`is False` -- only `==` is meaningful here.
    assert result.iloc[0]["est_ferie"] == True
    assert result.iloc[1]["est_ferie"] == False
    assert result.iloc[0]["vacances_zone_a"] == True
    assert result.iloc[1]["vacances_zone_a"] == True  # still within the Noël range
    assert result.iloc[0]["vacances_zone_b"] == False


@patch("felec.processing.build_dataset.save_processed")
@patch("felec.processing.build_dataset.load_raw")
def test_build_dataset_wires_sources_together(mock_load_raw, mock_save_processed):
    """Thin integration test: proves the orchestrator's wiring, not the math.

    cons_def and tr each contribute a row for a *different* hour, so a
    copy-paste slip that concatenates only one of the two datasets would
    silently drop a row instead of passing. The other assertions catch:
    save_processed never called (validate_processed_dataset raised, or the
    function short-circuited), and a value from the fake input failing to
    reach the output (a wrong merge column/type).
    """

    def fake_load_raw(relative_path):
        if relative_path == "rte/cons_def.parquet":
            return pd.DataFrame(
                {
                    "date_heure": ["2024-02-01T08:00:00+00:00"],
                    "consommation": [50000],
                    "prevision_j1": [51000],
                    "prevision_j": [50500],
                }
            )
        if relative_path == "rte/tr.parquet":
            return pd.DataFrame(
                {
                    "date_heure": ["2024-02-01T09:00:00+00:00"],
                    "consommation": [60000],
                    "prevision_j1": [61000],
                    "prevision_j": [60500],
                }
            )
        if relative_path.startswith("weather/"):
            return pd.DataFrame(
                {
                    "datetime": ["2024-02-01T08:00", "2024-02-01T09:00"],
                    "temperature": [10.0, 11.0],
                }
            )
        if relative_path == "calendar/jours_feries.parquet":
            return pd.DataFrame({"date": ["2024-01-01"], "nom": ["1er janvier"]})
        if relative_path == "calendar/vacances_scolaires.parquet":
            return pd.DataFrame(
                columns=["zone", "date_debut", "date_fin", "description"]
            )
        raise AssertionError(f"unexpected load_raw path: {relative_path}")

    mock_load_raw.side_effect = fake_load_raw

    result = build_dataset()

    # Both cons_def's and tr's row made it through the concat + merges.
    assert len(result) == 2
    assert set(result["consommation"]) == {50000.0, 60000.0}
    assert "temperature_nationale" in result.columns
    assert (result["est_ferie"] == False).all()  # 2024-02-01 is not a holiday

    mock_save_processed.assert_called_once()
    saved_df, saved_filename = mock_save_processed.call_args[0]
    assert len(saved_df) == 2
    assert saved_filename == "dataset.parquet"
