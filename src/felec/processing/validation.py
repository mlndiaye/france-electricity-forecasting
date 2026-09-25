"""Pandera schema for the final hourly modeling dataset."""
import datetime

import pandas as pd
import pandera.pandas as pa
from pandera.engines.pandas_engine import DateTime
from pandera.pandas import Check, Column, DataFrameSchema

processed_dataset_schema = DataFrameSchema(
    {
        # coerce=True here only normalizes datetime64 unit/precision (pandas's
        # default resolution varies across versions, e.g. "us" vs "ns"); it does
        # NOT vouch for timezone correctness, since coercion would silently
        # tz_localize a naive timestamp to UTC (treating it as already UTC) or
        # tz_convert a different timezone into UTC. validate_processed_dataset()
        # below rejects naive/non-UTC input explicitly before this schema runs,
        # so by the time this check applies, tz is already guaranteed to be UTC.
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
    """Validate df against the processed-dataset schema. Raises pandera.errors.SchemaError.

    date_heure's tz is checked explicitly first: the schema's coerce=True would
    otherwise silently accept a naive timestamp (tz_localize'd to UTC with no
    shift, i.e. treated as if it already were UTC) or a non-UTC timestamp
    (tz_convert'd into UTC), which would mask a real upstream bug instead of
    catching it.
    """
    tz = df["date_heure"].dt.tz
    if tz is None or str(tz) != "UTC":
        raise pa.errors.SchemaError(
            processed_dataset_schema,
            df,
            f"date_heure must be UTC-aware, got tz={tz}",
        )
    return processed_dataset_schema.validate(df)
