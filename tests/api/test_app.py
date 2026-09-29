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


def _fake_backtest_and_dataset():
    backtest = pd.DataFrame(
        {
            "date_heure": pd.to_datetime(
                ["2026-01-01T00:00:00Z", "2026-01-01T01:00:00Z", "2026-01-02T00:00:00Z"]
            ),
            "q10_pred": [39000.0, 38000.0, 37000.0],
            "q90_pred": [43000.0, 42000.0, 41000.0],
        }
    )
    dataset = pd.DataFrame(
        {
            "date_heure": pd.to_datetime(
                [
                    "2026-01-01T00:00:00Z",
                    "2026-01-01T01:00:00Z",
                    "2026-01-02T00:00:00Z",
                    "2026-01-03T00:00:00Z",  # not in backtest -- must not appear in the result
                ]
            ),
            "consommation": [42000.0, 41000.0, 40000.0, 39000.0],
            "prevision_j1": [42100.0, 41100.0, 40100.0, 39100.0],
        }
    )
    return backtest, dataset


@patch("felec.api.app.load_processed")
def test_forecast_history_joins_backtest_with_actual_and_rte(mock_load_processed):
    backtest, dataset = _fake_backtest_and_dataset()
    mock_load_processed.side_effect = lambda filename: (
        backtest if filename == "quantile_backtest_results.parquet" else dataset
    )

    response = client.get("/forecast/history")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 3  # the 2026-01-03 dataset row has no backtest match -- excluded
    assert {row["date_heure"][:10] for row in body} == {"2026-01-01", "2026-01-02"}
    first = body[0]
    assert first["consommation"] == 42000.0
    assert first["rte_pred"] == 42100.0
    assert first["q10_pred"] == 39000.0
    assert first["q90_pred"] == 43000.0


@patch("felec.api.app.load_processed")
def test_forecast_history_filters_by_start_and_end(mock_load_processed):
    backtest, dataset = _fake_backtest_and_dataset()
    mock_load_processed.side_effect = lambda filename: (
        backtest if filename == "quantile_backtest_results.parquet" else dataset
    )

    response = client.get("/forecast/history?start=2026-01-02&end=2026-01-02")

    assert response.status_code == 200
    body = response.json()
    assert len(body) == 1
    assert body[0]["date_heure"][:10] == "2026-01-02"


@patch("felec.api.app.load_processed")
def test_forecast_history_rejects_malformed_date(mock_load_processed):
    backtest, dataset = _fake_backtest_and_dataset()
    mock_load_processed.side_effect = lambda filename: (
        backtest if filename == "quantile_backtest_results.parquet" else dataset
    )

    response = client.get("/forecast/history?start=not-a-date")

    assert response.status_code == 400


@patch("felec.api.app.load_processed")
def test_forecast_history_rejects_start_after_end(mock_load_processed):
    backtest, dataset = _fake_backtest_and_dataset()
    mock_load_processed.side_effect = lambda filename: (
        backtest if filename == "quantile_backtest_results.parquet" else dataset
    )

    response = client.get("/forecast/history?start=2026-01-02&end=2026-01-01")

    assert response.status_code == 400


@patch("felec.api.app.load_processed")
def test_forecast_history_returns_404_when_backtest_missing(mock_load_processed):
    mock_load_processed.side_effect = FileNotFoundError()

    response = client.get("/forecast/history")

    assert response.status_code == 404
