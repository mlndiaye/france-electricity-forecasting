"""HTTP client and pure computations for the dashboard. No Streamlit import --
kept separate from app.py so this logic is unit-testable, and the dashboard
never computes anything modeling-related itself, only what the serving API
(ADR 0010) already returns. See ADR 0011.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import requests


class ApiUnavailableError(Exception):
    """The serving API could not be reached at all (connection refused)."""


class ForecastNotReadyError(Exception):
    """The serving API responded, but with 404 -- the underlying pipeline
    command (ingest predict / ingest quantile-backtest) hasn't run yet.
    """


def _get(base_url: str, path: str, params: dict | None = None) -> list:
    try:
        response = requests.get(f"{base_url}{path}", params=params, timeout=10)
    except requests.exceptions.ConnectionError as exc:
        raise ApiUnavailableError(f"Could not reach the API at {base_url}") from exc
    if response.status_code == 404:
        raise ForecastNotReadyError(response.json().get("detail", "Not found"))
    response.raise_for_status()
    return response.json()


def fetch_latest_forecast(base_url: str) -> pd.DataFrame:
    records = _get(base_url, "/forecast/latest")
    df = pd.DataFrame(records)
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    return df


def fetch_history(base_url: str, start: date | None, end: date | None) -> pd.DataFrame:
    params = {}
    if start is not None:
        params["start"] = start.isoformat()
    if end is not None:
        params["end"] = end.isoformat()
    records = _get(base_url, "/forecast/history", params=params)
    df = pd.DataFrame(records)
    if not df.empty:
        df["date_heure"] = pd.to_datetime(df["date_heure"])
    return df


def compute_interval_coverage(history: pd.DataFrame) -> float:
    inside = (history["consommation"] >= history["q10_pred"]) & (
        history["consommation"] <= history["q90_pred"]
    )
    return float(inside.mean())


def compute_rte_mae(history: pd.DataFrame) -> float:
    return float((history["rte_pred"] - history["consommation"]).abs().mean())
