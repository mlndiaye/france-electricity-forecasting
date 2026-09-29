"""Weather forecast connector (Open-Meteo Previous Runs API)."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import requests

from felec.ingestion.storage import save_raw

PREVIOUS_RUNS_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"

# Approximate aire urbaine population (INSEE, ~2021, order of magnitude) used
# only to weight cities into a national temperature signal -- not precise
# census figures. Refine later if the weighting turns out to matter.
CITIES: dict[str, tuple[float, float, int]] = {
    # name: (latitude, longitude, approx_population)
    "paris": (48.8566, 2.3522, 10_700_000),
    "lyon": (45.7640, 4.8357, 2_300_000),
    "marseille": (43.2965, 5.3698, 1_900_000),
    "lille": (50.6292, 3.0573, 1_500_000),
    "toulouse": (43.6047, 1.4442, 1_400_000),
    "bordeaux": (44.8378, -0.5792, 1_200_000),
    "nantes": (47.2184, -1.5536, 1_000_000),
    "strasbourg": (48.5734, 7.7521, 800_000),
}


def fetch_previous_day_forecast(
    latitude: float, longitude: float, start_date: date, end_date: date
) -> pd.DataFrame:
    """Fetch the D-1 forecast temperature for each hour in [start_date, end_date].

    Uses timezone=UTC so returned timestamps are directly comparable to RTE's
    UTC date_heure with no DST conversion. Returns a DataFrame with columns:
    datetime (str, ISO, UTC), temperature (float, nullable -- null before the
    archive's actual start, ~2024-02-01, or when a model run is missing).
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m_previous_day1",
        "models": "meteofrance_arome_france",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "timezone": "UTC",
    }
    response = requests.get(PREVIOUS_RUNS_URL, params=params, timeout=30)
    response.raise_for_status()
    hourly = response.json()["hourly"]
    return pd.DataFrame(
        {
            "datetime": hourly["time"],
            "temperature": hourly["temperature_2m_previous_day1"],
        }
    )


def backfill(start_date: date, end_date: date) -> None:
    """Fetch D-1 forecast temperature for every city, for [start_date, end_date]."""
    for city, (lat, lon, _population) in CITIES.items():
        df = fetch_previous_day_forecast(lat, lon, start_date, end_date)
        save_raw(df, f"weather/{city}.parquet", key_cols=["datetime"])


def refresh() -> None:
    """Re-fetch the last 7 days through tomorrow for every city.

    Includes tomorrow because the Previous Runs API already has tomorrow's
    forecast available live (verified against the real API, ADR 0007) --
    needed by the daily forecast pipeline to predict tomorrow. "Today"/
    "tomorrow" are Paris-local dates, not UTC, so the window's edges align
    with the demand pattern's own local-time convention (see features.py).
    """
    today = datetime.now(UTC).astimezone(ZoneInfo("Europe/Paris")).date()
    backfill(today - timedelta(days=7), today + timedelta(days=1))
