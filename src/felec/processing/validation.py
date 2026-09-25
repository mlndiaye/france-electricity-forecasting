"""Pandera schema for the final hourly modeling dataset."""
import datetime

import pandas as pd
import pandera as pa
from pandera import Check, Column, DataFrameSchema
from pandera.engines.pandas_engine import DateTime

processed_dataset_schema = DataFrameSchema(
    {
        # coerce=True: accept any datetime64 unit/precision (pandas's default
        # resolution varies across versions), as long as it is UTC-aware.
        "date_heure": Column(DateTime(tz=datetime.UTC), coerce=True),
        "consommation": Column(pa.Float, Check.in_range(20_000, 100_000), nullable=True),
        "prevision_j1": Column(pa.Float, Check.in_range(20_000, 100_000), nullable=True),
        "temperature_nationale": Column(pa.Float, Check.in_range(-25, 45), nullable=True),
        "est_ferie": Column(pa.Bool),
        "vacances_zone_a": Column(pa.Bool),
        "vacances_zone_b": Column(pa.Bool),
        "vacances_zone_c": Column(pa.Bool),
    },
    strict=False,  # allow extra columns (e.g. prevision_j) without failing
)


def validate_processed_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Validate df against the processed-dataset schema. Raises pandera.errors.SchemaError."""
    return processed_dataset_schema.validate(df)
