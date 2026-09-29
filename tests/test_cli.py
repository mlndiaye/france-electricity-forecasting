from unittest.mock import patch

from felec.cli import main


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
