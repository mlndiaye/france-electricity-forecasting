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
    The cursor is inclusive ("date_heure >= <last row's date_heure>"), so a
    full page's last row is always re-requested as (at least) the first row of
    the next page -- deliberately, since that's how ties on date_heure are
    detected and handled: results already come back sorted by date_heure
    (order_by), so re-fetching from the last-seen value and de-duplicating on
    date_heure (keeping the first occurrence) correctly keeps every row that
    shares that timestamp, whether both tied rows landed in the same page or
    split across the boundary. A ">" cursor would silently drop the
    split-across-boundary case forever, with nothing downstream to notice
    (no row-count check exists anywhere in the pipeline) -- see the
    corresponding tests. If more than PAGE_SIZE records ever shared one exact
    date_heure, the cursor would never advance (every row in the page would
    already be seen); that pathological case raises RuntimeError instead of
    looping forever.
    Filtering uses date_heure (a real datetime column) rather than date
    (filtering on date returns HTTP 400 on these datasets -- verified).
    Returns a DataFrame with columns: date, heure, date_heure, consommation,
    prevision_j1, prevision_j.
    """
    end_exclusive = end_date + timedelta(days=1)
    cursor = start_date.isoformat()
    rows: list[dict] = []
    seen_date_heures: set[str] = set()

    while True:
        where = f"date_heure >= '{cursor}' AND date_heure < '{end_exclusive.isoformat()}'"
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

        new_rows = [r for r in page if r["date_heure"] not in seen_date_heures]
        if not new_rows and len(page) == PAGE_SIZE:
            raise RuntimeError(
                f"RTE pagination made no progress at cursor {cursor!r} -- more than "
                f"{PAGE_SIZE} records appear to share the same date_heure, which "
                "breaks this cursor-based pagination's assumption that date_heure "
                "is a usable ordering key."
            )
        seen_date_heures.update(r["date_heure"] for r in new_rows)
        rows.extend(new_rows)

        if len(page) < PAGE_SIZE:
            break
        cursor = page[-1]["date_heure"]

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
