from datetime import date, timedelta
from unittest.mock import Mock, patch

import pandas as pd

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


@patch("felec.ingestion.rte.requests.get")
def test_fetch_records_paginates_until_short_page(mock_get):
    page_1 = {"results": [ONE_RECORD] * 100}
    page_2 = {"results": [ONE_RECORD]}
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
    pick up strictly after the previous page's last row.
    """
    page_1_last = {**ONE_RECORD, "date_heure": "2024-02-01T00:00:00+00:00"}
    page_2_first = {**ONE_RECORD, "date_heure": "2024-02-01T00:15:00+00:00"}
    page_1 = {"results": [page_1_last] * 100}
    page_2 = {"results": [page_2_first]}
    mock_get.side_effect = [
        Mock(json=lambda: page_1, raise_for_status=lambda: None),
        Mock(json=lambda: page_2, raise_for_status=lambda: None),
    ]

    result = fetch_records(TR_DATASET, date(2024, 2, 1), date(2024, 2, 1))

    assert len(result) == 101
    first_params = mock_get.call_args_list[0].kwargs["params"]
    second_params = mock_get.call_args_list[1].kwargs["params"]
    assert "offset" not in first_params
    assert "offset" not in second_params
    assert f"date_heure > '{page_1_last['date_heure']}'" in second_params["where"]


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
