from datetime import date
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import requests

from felec.dashboard.data import (
    ApiUnavailableError,
    ForecastNotReadyError,
    compute_interval_coverage,
    compute_rte_mae,
    fetch_history,
    fetch_latest_forecast,
)


@patch("felec.dashboard.data.requests.get")
def test_fetch_latest_forecast_parses_the_response(mock_get):
    mock_get.return_value = Mock(
        status_code=200,
        json=lambda: [
            {"date_heure": "2026-09-30T00:00:00Z", "q10_pred": 100.0, "q50_pred": 110.0},
        ],
        raise_for_status=lambda: None,
    )

    df = fetch_latest_forecast("http://localhost:8000")

    mock_get.assert_called_once_with(
        "http://localhost:8000/forecast/latest", params=None, timeout=10
    )
    assert len(df) == 1
    assert df.iloc[0]["q50_pred"] == 110.0
    assert pd.api.types.is_datetime64_any_dtype(df["date_heure"])


@patch("felec.dashboard.data.requests.get")
def test_fetch_latest_forecast_raises_when_api_unreachable(mock_get):
    mock_get.side_effect = requests.exceptions.ConnectionError()

    with pytest.raises(ApiUnavailableError):
        fetch_latest_forecast("http://localhost:8000")


@patch("felec.dashboard.data.requests.get")
def test_fetch_latest_forecast_raises_when_not_ready(mock_get):
    mock_get.return_value = Mock(status_code=404, json=lambda: {"detail": "No forecast yet"})

    with pytest.raises(ForecastNotReadyError):
        fetch_latest_forecast("http://localhost:8000")


@patch("felec.dashboard.data.requests.get")
def test_fetch_history_passes_start_and_end_as_query_params(mock_get):
    mock_get.return_value = Mock(status_code=200, json=lambda: [], raise_for_status=lambda: None)

    fetch_history("http://localhost:8000", date(2026, 1, 1), date(2026, 1, 31))

    mock_get.assert_called_once_with(
        "http://localhost:8000/forecast/history",
        params={"start": "2026-01-01", "end": "2026-01-31"},
        timeout=10,
    )


@patch("felec.dashboard.data.requests.get")
def test_fetch_history_omits_params_when_not_given(mock_get):
    mock_get.return_value = Mock(status_code=200, json=lambda: [], raise_for_status=lambda: None)

    fetch_history("http://localhost:8000", None, None)

    mock_get.assert_called_once_with(
        "http://localhost:8000/forecast/history", params={}, timeout=10
    )


def test_compute_interval_coverage():
    history = pd.DataFrame(
        {
            "consommation": [100.0, 100.0, 100.0, 100.0],
            "q10_pred": [90.0, 90.0, 90.0, 150.0],  # last row: consommation below q10
            "q90_pred": [110.0, 110.0, 95.0, 160.0],  # third row: consommation above q90
        }
    )

    coverage = compute_interval_coverage(history)

    assert coverage == 0.5  # 2 of 4 rows inside [q10, q90]


def test_compute_rte_mae():
    history = pd.DataFrame(
        {
            "consommation": [100.0, 200.0, 300.0],
            "rte_pred": [110.0, 190.0, 320.0],
        }
    )

    mae = compute_rte_mae(history)

    assert mae == pytest.approx((10 + 10 + 20) / 3)
