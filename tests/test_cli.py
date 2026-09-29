from datetime import UTC, date, datetime
from unittest.mock import patch

import pandas as pd

from felec.cli import daily_forecast, main


@patch("felec.cli.backfill")
def test_main_dispatches_to_backfill(mock_backfill, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ingest", "backfill"])
    main()
    mock_backfill.assert_called_once()


@patch("felec.cli.refresh")
def test_main_dispatches_to_refresh(mock_refresh, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ingest", "refresh"])
    main()
    mock_refresh.assert_called_once()


@patch("felec.cli.build_dataset")
def test_main_dispatches_to_build_dataset(mock_build, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ingest", "build-dataset"])
    main()
    mock_build.assert_called_once()


@patch("felec.cli.backtest")
def test_main_dispatches_to_backtest(mock_backtest, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ingest", "backtest"])
    main()
    mock_backtest.assert_called_once()


@patch("felec.cli.quantile_backtest")
def test_main_dispatches_to_quantile_backtest(mock_quantile_backtest, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ingest", "quantile-backtest"])
    main()
    mock_quantile_backtest.assert_called_once()


@patch("felec.cli.daily_forecast")
def test_main_dispatches_to_daily_forecast(mock_daily_forecast, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ingest", "daily-forecast"])
    main()
    mock_daily_forecast.assert_called_once()


@patch("felec.cli.save_processed")
@patch("felec.cli.predict_next_day")
@patch("felec.cli.build_dataset")
@patch("felec.cli.refresh")
@patch("felec.cli.datetime")
def test_daily_forecast_computes_target_date_from_paris_local_time(
    mock_datetime, mock_refresh, mock_build_dataset, mock_predict, mock_save
):
    """Regression test: near the UTC/Paris day boundary (23:00 UTC = 01:00
    Paris local the next day, during CEST), target_date must be computed from
    the Paris-local date, not the UTC date -- a UTC-based "tomorrow" would
    silently resolve to Paris-local "today" instead.
    """
    mock_datetime.now.return_value = datetime(2026, 9, 29, 23, 0, tzinfo=UTC)
    mock_build_dataset.return_value = pd.DataFrame({"date_heure": [], "consommation": []})
    mock_predict.return_value = pd.DataFrame({"date_heure": []})

    daily_forecast()

    mock_predict.assert_called_once()
    _, kwargs = mock_predict.call_args
    assert kwargs["target_date"] == date(2026, 10, 1)
