"""Build the final hourly modeling dataset from the raw ingested sources."""
from __future__ import annotations

import pandas as pd

from felec.ingestion.storage import load_raw, save_processed
from felec.ingestion.weather import CITIES
from felec.processing.validation import validate_processed_dataset


def aggregate_rte_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Average sub-hourly RTE readings into one row per hour.

    A flow quantity (MW) is better represented by the mean over the hour than
    by picking a single sub-hourly instant, which would under/over-count
    whatever happened later in that hour.
    Input columns: date_heure (str, ISO), consommation, prevision_j1, prevision_j.
    Output: date_heure floored to the hour (UTC), same columns averaged.
    """
    df = df.copy()
    df["date_heure"] = pd.to_datetime(df["date_heure"], utc=True)
    df["hour"] = df["date_heure"].dt.floor("h")
    return (
        df.groupby("hour", as_index=False)[["consommation", "prevision_j1", "prevision_j"]]
        .mean()
        .rename(columns={"hour": "date_heure"})
    )


def combine_cities_weighted(city_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Combine per-city hourly temperatures into one population-weighted series.

    Each frame in city_frames has columns: datetime (str, ISO, UTC), temperature.
    Weights come from felec.ingestion.weather.CITIES. An hour where every city
    is null becomes NaN (not 0 -- a silent 0degC would be a dangerous bug, not
    a plausible value). An hour where only some cities are missing is not
    reweighted among the remaining ones; this is a known simplification.
    """
    total_population = sum(pop for _, _, pop in CITIES.values())
    weighted_frames = []

    for city, df in city_frames.items():
        _, _, population = CITIES[city]
        weight = population / total_population
        city_df = df[["datetime", "temperature"]].copy()
        city_df["date_heure"] = pd.to_datetime(city_df["datetime"], utc=True)
        city_df["weighted_temperature"] = city_df["temperature"] * weight
        weighted_frames.append(city_df[["date_heure", "weighted_temperature"]])

    stacked = pd.concat(weighted_frames, ignore_index=True)
    return (
        stacked.groupby("date_heure", as_index=False)["weighted_temperature"]
        .sum(min_count=1)
        .rename(columns={"weighted_temperature": "temperature_nationale"})
    )


def expand_calendar_to_hourly(
    date_heure_index: pd.DatetimeIndex,
    jours_feries: pd.DataFrame,
    vacances: pd.DataFrame,
) -> pd.DataFrame:
    """Broadcast daily calendar flags onto an hourly index.

    jours_feries: columns date, nom.
    vacances: columns zone, date_debut, date_fin, description.
    Returns a DataFrame with columns: date_heure, est_ferie, vacances_zone_a,
    vacances_zone_b, vacances_zone_c.
    """
    days = pd.DatetimeIndex(date_heure_index).tz_convert("UTC").normalize()
    ferie_dates = set(pd.to_datetime(jours_feries["date"], utc=True).dt.normalize())

    zone_ranges: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {
        "Zone A": [],
        "Zone B": [],
        "Zone C": [],
    }
    for _, row in vacances.iterrows():
        if row["zone"] in zone_ranges:
            zone_ranges[row["zone"]].append(
                (
                    pd.Timestamp(row["date_debut"], tz="UTC"),
                    pd.Timestamp(row["date_fin"], tz="UTC"),
                )
            )

    def in_any_range(day: pd.Timestamp, ranges: list[tuple[pd.Timestamp, pd.Timestamp]]) -> bool:
        return any(start <= day <= end for start, end in ranges)

    return pd.DataFrame(
        {
            "date_heure": date_heure_index,
            "est_ferie": [d in ferie_dates for d in days],
            "vacances_zone_a": [in_any_range(d, zone_ranges["Zone A"]) for d in days],
            "vacances_zone_b": [in_any_range(d, zone_ranges["Zone B"]) for d in days],
            "vacances_zone_c": [in_any_range(d, zone_ranges["Zone C"]) for d in days],
        }
    )


def build_dataset() -> pd.DataFrame:
    """Load all raw sources, join them into one hourly dataset, validate, and save."""
    cons_def = load_raw("rte/cons_def.parquet")
    tr = load_raw("rte/tr.parquet")
    rte_raw = pd.concat([cons_def, tr], ignore_index=True)
    rte_hourly = aggregate_rte_hourly(rte_raw)

    city_frames = {city: load_raw(f"weather/{city}.parquet") for city in CITIES}
    weather_hourly = combine_cities_weighted(city_frames)

    jours_feries = load_raw("calendar/jours_feries.parquet")
    vacances = load_raw("calendar/vacances_scolaires.parquet")
    calendar_hourly = expand_calendar_to_hourly(
        pd.DatetimeIndex(rte_hourly["date_heure"]), jours_feries, vacances
    )

    dataset = rte_hourly.merge(weather_hourly, on="date_heure", how="left")
    dataset = dataset.merge(calendar_hourly, on="date_heure", how="left")

    dataset = validate_processed_dataset(dataset)
    save_processed(dataset, "dataset.parquet")
    return dataset
