from unittest.mock import patch

import pandas as pd
from fastapi.testclient import TestClient

from felec.api.app import app

client = TestClient(app)


def test_health_returns_ok():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@patch("felec.api.app.load_processed")
def test_forecast_latest_returns_the_saved_forecast(mock_load_processed):
    mock_load_processed.return_value = pd.DataFrame(
        {
            "date_heure": pd.to_datetime(["2026-09-30T00:00:00Z", "2026-09-30T01:00:00Z"]),
            "q10_pred": [40000.0, 39000.0],
            "q50_pred": [42000.0, 41000.0],
            "q90_pred": [44000.0, 43000.0],
            "naive_pred": [41500.0, 40500.0],
            "rte_pred": [42200.0, 41200.0],
        }
    )

    response = client.get("/forecast/latest")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 2
    assert body[0]["q10_pred"] == 40000.0
    assert body[0]["q50_pred"] == 42000.0
    mock_load_processed.assert_called_once_with("forecast_latest.parquet")


@patch("felec.api.app.load_processed")
def test_forecast_latest_returns_404_when_no_forecast_yet(mock_load_processed):
    mock_load_processed.side_effect = FileNotFoundError()

    response = client.get("/forecast/latest")

    assert response.status_code == 404
