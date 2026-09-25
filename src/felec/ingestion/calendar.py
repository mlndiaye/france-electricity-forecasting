"""Public holidays and school holidays connector."""
from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import requests

from felec.ingestion.storage import save_raw

JOURS_FERIES_URL = "https://calendrier.api.gouv.fr/jours-feries/metropole/{year}.json"
VACANCES_URL = "https://data.education.gouv.fr/api/records/1.0/search/"
VACANCES_DATASET = "fr-en-calendrier-scolaire"
VALID_ZONES = {"Zone A", "Zone B", "Zone C"}


def fetch_jours_feries(year: int) -> pd.DataFrame:
    """Fetch French public holidays for a given year.

    Returns a DataFrame with columns: date (str, YYYY-MM-DD), nom (str).
    """
    response = requests.get(JOURS_FERIES_URL.format(year=year), timeout=10)
    response.raise_for_status()
    data = response.json()
    return pd.DataFrame([{"date": d, "nom": nom} for d, nom in data.items()])


def fetch_vacances_scolaires(annee_scolaire: str) -> pd.DataFrame:
    """Fetch school holidays for a given school year (e.g. "2023-2024").

    The source API returns one record per (zone, location) pair -- the same
    holiday period repeated once per académie within a zone. This collapses
    that down to one row per (zone, date_debut, date_fin, description).
    Columns: zone, date_debut, date_fin, description.
    """
    params = {
        "dataset": VACANCES_DATASET,
        "rows": 1000,
        "refine.annee_scolaire": annee_scolaire,
    }
    response = requests.get(VACANCES_URL, params=params, timeout=10)
    response.raise_for_status()
    records = response.json()["records"]

    rows = [
        {
            "zone": r["fields"]["zones"],
            "date_debut": r["fields"]["start_date"][:10],
            "date_fin": r["fields"]["end_date"][:10],
            "description": r["fields"]["description"],
        }
        for r in records
        if r["fields"]["zones"] in VALID_ZONES
    ]
    df = pd.DataFrame(rows)
    return df.drop_duplicates().reset_index(drop=True)


def backfill(start_year: int, end_year: int) -> None:
    """Fetch public and school holidays for every year in [start_year, end_year]."""
    for year in range(start_year, end_year + 1):
        holidays = fetch_jours_feries(year)
        save_raw(holidays, "calendar/jours_feries.parquet", key_cols=["date"])

    for year in range(start_year, end_year + 1):
        annee_scolaire = f"{year}-{year + 1}"
        vacances = fetch_vacances_scolaires(annee_scolaire)
        save_raw(
            vacances,
            "calendar/vacances_scolaires.parquet",
            key_cols=["zone", "date_debut", "date_fin", "description"],
        )


def refresh() -> None:
    """Re-fetch the current and next year (holidays are published in advance)."""
    current_year = datetime.now(UTC).year
    backfill(current_year, current_year + 1)
