"""RTE éCO2mix connector (via the ODRÉ Explore v2.1 API)."""
from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import requests

from felec.ingestion.storage import save_raw

RECORDS_URL = (
    "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/{dataset}/records"
)
FIELDS = ["date", "heure", "date_heure", "consommation", "prevision_j1", "prevision_j"]
PAGE_SIZE = 100

CONS_DEF_DATASET = "eco2mix-national-cons-def"
TR_DATASET = "eco2mix-national-tr"

# eco2mix-national-cons-def is republished periodically and always trails
# today by a consolidation lag. This is the boundary verified empirically on
# 2026-09-25 (cons-def's last record matched tr's first record exactly). It
# will drift forward over time -- re-verify if cons-def ever looks stale
# (e.g. its most recent date_heure is unexpectedly old).
CONS_DEF_LAST_DATE = date(2026, 6, 30)


def fetch_records(dataset: str, start_date: date, end_date: date) -> pd.DataFrame:
    """Fetch every record of `dataset` with date_heure in [start_date, end_date], inclusive.

    Paginates through the API (100 records per page, the API's page size).
    Filtering uses date_heure (a real datetime column) rather than date
    (filtering on date returns HTTP 400 on these datasets -- verified).
    Returns a DataFrame with columns: date, heure, date_heure, consommation,
    prevision_j1, prevision_j.
    """
    end_exclusive = end_date + timedelta(days=1)
    where = (
        f"date_heure >= '{start_date.isoformat()}' "
        f"AND date_heure < '{end_exclusive.isoformat()}'"
    )
    rows: list[dict] = []
    offset = 0

    while True:
        response = requests.get(
            RECORDS_URL.format(dataset=dataset),
            params={
                "select": ",".join(FIELDS),
                "where": where,
                "order_by": "date_heure",
                "limit": PAGE_SIZE,
                "offset": offset,
            },
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()["results"]
        if not page:
            break
        rows.extend(page)
        offset += PAGE_SIZE
        if len(page) < PAGE_SIZE:
            break

    return pd.DataFrame(rows, columns=FIELDS)


def backfill(start_date: date, end_date: date) -> None:
    """Fetch RTE data for [start_date, end_date], chaining cons-def and tr.

    cons-def covers the older part of the range (up to CONS_DEF_LAST_DATE),
    tr covers whatever is more recent than that.
    """
    if start_date <= CONS_DEF_LAST_DATE:
        cons_def_end = min(end_date, CONS_DEF_LAST_DATE)
        df = fetch_records(CONS_DEF_DATASET, start_date, cons_def_end)
        save_raw(df, "rte/cons_def.parquet", key_cols=["date_heure"])

    if end_date > CONS_DEF_LAST_DATE:
        tr_start = max(start_date, CONS_DEF_LAST_DATE + timedelta(days=1))
        df = fetch_records(TR_DATASET, tr_start, end_date)
        save_raw(df, "rte/tr.parquet", key_cols=["date_heure"])


def refresh() -> None:
    """Re-fetch the last 5 days through today from the tr dataset."""
    today = datetime.now(UTC).date()
    df = fetch_records(TR_DATASET, today - timedelta(days=5), today)
    save_raw(df, "rte/tr.parquet", key_cols=["date_heure"])
