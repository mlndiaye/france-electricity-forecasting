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

    Paginates through the API (100 records per page, the API's page size) by
    cursoring on date_heure rather than using offset: the API hard-caps
    offset + limit at 10,000 (verified empirically -- InvalidRESTParameterError
    above that), which a naive offset-based loop exceeds for any range longer
    than ~100 days at 15-minute resolution -- i.e. for any real backfill.
    Cursoring instead by "date_heure > <last row's date_heure>" keeps offset
    at 0 always, so it has no such ceiling. This assumes date_heure is unique
    per record, consistent with save_raw's use of date_heure as the dedup key
    for these same files -- if two rows ever tie on date_heure and land split
    across a page boundary, the cursor would silently skip the rest of the
    tied group, so that case is checked for explicitly and raises RuntimeError
    rather than under-counting silently.
    Filtering uses date_heure (a real datetime column) rather than date
    (filtering on date returns HTTP 400 on these datasets -- verified).
    Returns a DataFrame with columns: date, heure, date_heure, consommation,
    prevision_j1, prevision_j.
    """
    end_exclusive = end_date + timedelta(days=1)
    lower_bound = f"'{start_date.isoformat()}'"
    comparator = ">="
    rows: list[dict] = []

    while True:
        where = (
            f"date_heure {comparator} {lower_bound} "
            f"AND date_heure < '{end_exclusive.isoformat()}'"
        )
        response = requests.get(
            RECORDS_URL.format(dataset=dataset),
            params={
                "select": ",".join(FIELDS),
                "where": where,
                "order_by": "date_heure",
                "limit": PAGE_SIZE,
            },
            timeout=30,
        )
        response.raise_for_status()
        page = response.json()["results"]
        if not page:
            break
        rows.extend(page)
        if len(page) >= 2 and page[-1]["date_heure"] == page[-2]["date_heure"]:
            raise RuntimeError(
                f"duplicate date_heure at page boundary: {page[-1]['date_heure']!r} -- "
                "cursor pagination cannot safely advance past tied timestamps"
            )
        lower_bound = f"'{page[-1]['date_heure']}'"
        comparator = ">"
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
