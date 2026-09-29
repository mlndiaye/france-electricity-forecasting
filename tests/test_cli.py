from datetime import UTC, date, datetime
from unittest.mock import Mock, patch

import pandas as pd

from felec.cli import daily_forecast, main, predict
from felec.modeling.forecast import ForecastResult


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


@patch("felec.cli.predict")
def test_main_dispatches_to_predict(mock_predict, monkeypatch):
    monkeypatch.setattr("sys.argv", ["ingest", "predict"])
    main()
    mock_predict.assert_called_once()


def _fake_forecast_result(**overrides) -> ForecastResult:
    defaults = {
        "predictions": pd.DataFrame({"date_heure": []}),
        "models": {},
        "params": {},
        "cutoff_hour": 12,
        "n_train_rows": 0,
    }
    defaults.update(overrides)
    return ForecastResult(**defaults)


@patch("felec.cli.mlflow")
@patch("felec.cli.save_processed")
@patch("felec.cli.predict_next_day")
@patch("felec.cli.load_processed")
@patch("felec.cli.datetime")
def test_predict_computes_target_date_from_paris_local_time(
    mock_datetime, mock_load_processed, mock_predict, mock_save, mock_mlflow
):
    """Regression test: near the UTC/Paris day boundary (23:00 UTC = 01:00
    Paris local the next day, during CEST), target_date must be computed from
    the Paris-local date, not the UTC date -- a UTC-based "tomorrow" would
    silently resolve to Paris-local "today" instead.
    """
    mock_datetime.now.return_value = datetime(2026, 9, 29, 23, 0, tzinfo=UTC)
    mock_load_processed.return_value = pd.DataFrame({"date_heure": [], "consommation": []})
    mock_predict.return_value = _fake_forecast_result()

    predict()

    mock_predict.assert_called_once()
    _, kwargs = mock_predict.call_args
    assert kwargs["target_date"] == date(2026, 10, 1)


@patch("felec.cli.mlflow")
@patch("felec.cli.save_processed")
@patch("felec.cli.predict_next_day")
@patch("felec.cli.load_processed")
@patch("felec.cli.datetime")
def test_predict_logs_params_metrics_and_models_to_mlflow(
    mock_datetime, mock_load_processed, mock_predict, mock_save, mock_mlflow
):
    mock_datetime.now.return_value = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
    mock_load_processed.return_value = pd.DataFrame({"date_heure": [], "consommation": []})
    mock_predict.return_value = _fake_forecast_result(
        models={0.1: Mock(), 0.5: Mock(), 0.9: Mock()},
        params={"num_leaves": 63, "learning_rate": 0.05},
        cutoff_hour=12,
        n_train_rows=1234,
    )

    predict()

    mock_mlflow.start_run.assert_called_once()
    mock_mlflow.log_metric.assert_called_once_with("n_train_rows", 1234)
    assert mock_mlflow.lightgbm.log_model.call_count == 3


@patch("felec.cli.predict")
@patch("felec.cli.build_dataset")
@patch("felec.cli.refresh")
def test_daily_forecast_calls_refresh_build_dataset_and_predict_in_order(
    mock_refresh, mock_build_dataset, mock_predict
):
    manager = Mock()
    manager.attach_mock(mock_refresh, "refresh")
    manager.attach_mock(mock_build_dataset, "build_dataset")
    manager.attach_mock(mock_predict, "predict")

    daily_forecast()

    assert [call[0] for call in manager.mock_calls] == ["refresh", "build_dataset", "predict"]
