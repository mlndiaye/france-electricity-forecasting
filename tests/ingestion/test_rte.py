from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch

import pandas as pd
import pytest

from felec.ingestion.rte import (
    CONS_DEF_DATASET,
    CONS_DEF_LAST_DATE,
    TR_DATASET,
    backfill,
    fetch_records,
)

ONE_RECORD = {
    "date": "2024-02-01",
    "heure": "00:00",
    "date_heure": "2024-02-01T00:00:00+00:00",
    "consommation": 45000,
    "prevision_j1": 46000,
    "prevision_j": 45500,
}


def _records(n: int, start_index: int = 0) -> list[dict]:
    """n synthetic records with distinct, 15-minute-incrementing date_heure
    values (like the real, ordered API response), starting start_index * 15
    minutes past 2024-02-01T00:00:00Z."""
    base = datetime.fromisoformat("2024-02-01T00:00:00+00:00")
    return [
        {**ONE_RECORD, "date_heure": (base + timedelta(minutes=15 * (start_index + i))).isoformat()}
        for i in range(n)
    ]


@patch("felec.ingestion.rte.requests.get")
def test_fetch_records_paginates_until_short_page(mock_get):
    page_1 = {"results": _records(100, start_index=0)}
    page_2 = {"results": _records(1, start_index=100)}
    mock_get.side_effect = [
        Mock(json=lambda: page_1, raise_for_status=lambda: None),
        Mock(json=lambda: page_2, raise_for_status=lambda: None),
    ]

    result = fetch_records(TR_DATASET, date(2024, 2, 1), date(2024, 2, 1))

    assert len(result) == 101
    assert mock_get.call_count == 2


@patch("felec.ingestion.rte.requests.get")
def test_fetch_records_paginates_by_cursoring_on_date_heure_not_offset(mock_get):
    """Regression test: the ODRE API hard-caps offset + limit at 10,000
    (InvalidRESTParameterError, verified against the live API), which an
    offset-based page loop exceeds for any date range with more than 10,000
    records -- guaranteed for a real multi-year backfill. Pagination must
    advance by cursoring on date_heure instead, so no request should ever
    carry an "offset" param, and each successive page's `where` clause must
    pick up from (inclusive of) the previous page's last row -- inclusive so
    that a tie on date_heure at the boundary is re-fetched and can be
    recognized, rather than silently skipped by a strict ">" cursor.
    """
    page_1_records = _records(100, start_index=0)
    page_2_records = _records(1, start_index=100)
    mock_get.side_effect = [
        Mock(json=lambda: {"results": page_1_records}, raise_for_status=lambda: None),
        Mock(json=lambda: {"results": page_2_records}, raise_for_status=lambda: None),
    ]

    result = fetch_records(TR_DATASET, date(2024, 2, 1), date(2024, 2, 1))

    assert len(result) == 101
    first_params = mock_get.call_args_list[0].kwargs["params"]
    second_params = mock_get.call_args_list[1].kwargs["params"]
    assert "offset" not in first_params
    assert "offset" not in second_params
    assert f"date_heure >= '{page_1_records[-1]['date_heure']}'" in second_params["where"]


@patch("felec.ingestion.rte.requests.get")
def test_fetch_records_dedupes_row_refetched_at_inclusive_cursor_boundary(mock_get):
    """The inclusive ">=" cursor deliberately re-requests page 1's last row as
    (at least) page 2's first row -- this is how a tie on date_heure at the
    boundary would be recognized in the first place. This proves that re-fetch
    is correctly collapsed to a single row (keeping the first-seen, page-1
    copy -- proven via a sentinel field the page-2 copy doesn't share) rather
    than counted twice, while a genuinely new row past the boundary is still
    kept.

    Note: this constructs the re-fetch-of-the-same-row pattern the inclusive
    cursor itself produces, not two independently-tied original API rows (the
    connector assumes date_heure is unique per underlying record -- see this
    module's docstring). See test_fetch_records_dedupes_same_page_tie below
    for two rows genuinely sharing a date_heure.
    """
    page_1_records = _records(100, start_index=0)
    tied_date_heure = page_1_records[-1]["date_heure"]
    refetched_row = {**ONE_RECORD, "date_heure": tied_date_heure, "consommation": -1}
    genuinely_new_date_heure = (
        datetime.fromisoformat(tied_date_heure) + timedelta(minutes=15)
    ).isoformat()
    page_2_records = [
        refetched_row,
        {**ONE_RECORD, "date_heure": genuinely_new_date_heure},
    ]
    mock_get.side_effect = [
        Mock(json=lambda: {"results": page_1_records}, raise_for_status=lambda: None),
        Mock(json=lambda: {"results": page_2_records}, raise_for_status=lambda: None),
    ]

    result = fetch_records(TR_DATASET, date(2024, 2, 1), date(2024, 2, 1))

    # 100 distinct rows from page 1 plus the 1 genuinely new row from page 2 --
    # the re-fetched duplicate must not be double-counted, and the tied row
    # must not be lost either.
    assert len(result) == 101
    assert result["date_heure"].nunique() == 101
    tied_rows = result.loc[result["date_heure"] == tied_date_heure]
    assert len(tied_rows) == 1
    assert tied_rows.iloc[0]["consommation"] != -1  # kept page 1's copy, not the re-fetch
    assert genuinely_new_date_heure in set(result["date_heure"])


@patch("felec.ingestion.rte.requests.get")
def test_fetch_records_dedupes_same_page_tie(mock_get):
    """Regression test for the incremental-dedup fix: two rows sharing a
    date_heure WITHIN THE SAME PAGE (not at a page boundary) must collapse to
    a single row. Filtering has to check each row against the accumulated
    seen-set as it goes; filtering the whole page via a list comprehension and
    only updating the seen-set afterward lets both tied rows in one page pass
    the same stale check and both get added -- silently duplicating the row
    instead of deduping it (this was a real regression caught by review).
    """
    tied_date_heure = _records(1, start_index=10)[0]["date_heure"]
    page = (
        _records(10, start_index=0)
        + [
            {**ONE_RECORD, "date_heure": tied_date_heure},
            {**ONE_RECORD, "date_heure": tied_date_heure},
        ]
        + _records(10, start_index=11)
    )
    mock_get.return_value = Mock(json=lambda: {"results": page}, raise_for_status=lambda: None)

    result = fetch_records(TR_DATASET, date(2024, 2, 1), date(2024, 2, 1))

    assert mock_get.call_count == 1  # page shorter than PAGE_SIZE -- no second request
    assert len(result) == 21  # 10 + 1 (deduped tie) + 10, not 22
    assert len(result.loc[result["date_heure"] == tied_date_heure]) == 1


@patch("felec.ingestion.rte.PAGE_SIZE", 2)
@patch("felec.ingestion.rte.requests.get")
def test_fetch_records_raises_when_more_than_page_size_share_one_timestamp(mock_get):
    """Pathological case: if more than PAGE_SIZE records share the exact same
    date_heure, the inclusive cursor can never move past them -- every row of
    the next page has already been seen -- which would otherwise loop
    forever. Must raise loudly instead. PAGE_SIZE is patched down to 2 so a
    tiny mocked page can represent "a full page, entirely duplicates".
    """
    tied = "2024-02-01T00:00:00+00:00"
    duplicate_page = {"results": [{**ONE_RECORD, "date_heure": tied}] * 2}
    mock_get.side_effect = [
        Mock(json=lambda: duplicate_page, raise_for_status=lambda: None),
        Mock(json=lambda: duplicate_page, raise_for_status=lambda: None),
    ]

    with pytest.raises(RuntimeError, match="no progress"):
        fetch_records(TR_DATASET, date(2024, 2, 1), date(2024, 2, 1))


@patch("felec.ingestion.rte.requests.get")
def test_fetch_records_stops_on_empty_page(mock_get):
    mock_get.return_value = Mock(json=lambda: {"results": []}, raise_for_status=lambda: None)

    result = fetch_records(TR_DATASET, date(2024, 2, 1), date(2024, 2, 1))

    assert result.empty
    assert mock_get.call_count == 1


@patch("felec.ingestion.rte.save_raw")
@patch("felec.ingestion.rte.fetch_records")
def test_backfill_queries_cons_def_only_when_fully_before_boundary(mock_fetch, mock_save):
    mock_fetch.return_value = pd.DataFrame()

    backfill(date(2024, 2, 1), date(2024, 2, 5))

    mock_fetch.assert_called_once_with(CONS_DEF_DATASET, date(2024, 2, 1), date(2024, 2, 5))


@patch("felec.ingestion.rte.save_raw")
@patch("felec.ingestion.rte.fetch_records")
def test_backfill_queries_both_datasets_when_range_straddles_boundary(mock_fetch, mock_save):
    mock_fetch.return_value = pd.DataFrame()
    start = CONS_DEF_LAST_DATE - timedelta(days=2)
    end = CONS_DEF_LAST_DATE + timedelta(days=2)

    backfill(start, end)

    assert mock_fetch.call_count == 2
    mock_fetch.assert_any_call(CONS_DEF_DATASET, start, CONS_DEF_LAST_DATE)
    mock_fetch.assert_any_call(TR_DATASET, CONS_DEF_LAST_DATE + timedelta(days=1), end)


@patch("felec.ingestion.rte.save_raw")
@patch("felec.ingestion.rte.fetch_records")
def test_backfill_queries_tr_only_when_fully_after_boundary(mock_fetch, mock_save):
    mock_fetch.return_value = pd.DataFrame()
    start = CONS_DEF_LAST_DATE + timedelta(days=10)
    end = CONS_DEF_LAST_DATE + timedelta(days=20)

    backfill(start, end)

    mock_fetch.assert_called_once_with(TR_DATASET, start, end)
