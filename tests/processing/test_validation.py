import pandas as pd
import pandera as pa
import pytest

from felec.processing.validation import validate_processed_dataset


def _valid_row() -> dict:
    return {
        "date_heure": pd.Timestamp("2024-02-01T08:00:00", tz="UTC"),
        "consommation": 52000.0,
        "prevision_j1": 53000.0,
        "temperature_nationale": 8.5,
        "est_ferie": False,
        "vacances_zone_a": False,
        "vacances_zone_b": False,
        "vacances_zone_c": False,
    }


def test_valid_dataset_passes():
    df = pd.DataFrame([_valid_row()])
    validate_processed_dataset(df)  # should not raise


def test_temperature_out_of_range_fails():
    row = _valid_row()
    row["temperature_nationale"] = 80.0
    df = pd.DataFrame([row])

    with pytest.raises(pa.errors.SchemaError):
        validate_processed_dataset(df)


def test_consumption_out_of_range_fails():
    row = _valid_row()
    row["consommation"] = 500.0
    df = pd.DataFrame([row])

    with pytest.raises(pa.errors.SchemaError):
        validate_processed_dataset(df)


def test_naive_datetime_fails():
    row = _valid_row()
    row["date_heure"] = pd.Timestamp("2024-02-01T08:00:00")  # naive, no tz
    df = pd.DataFrame([row])

    with pytest.raises((pa.errors.SchemaError, ValueError)):
        validate_processed_dataset(df)
