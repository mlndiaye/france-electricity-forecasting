from datetime import UTC, date, datetime
from unittest.mock import Mock, patch

from felec.ingestion.weather import CITIES, backfill, fetch_previous_day_forecast, refresh

FAKE_RESPONSE = {
    "hourly": {
        "time": ["2024-02-01T00:00", "2024-02-01T01:00", "2024-02-01T02:00"],
        "temperature_2m_previous_day1": [8.5, 8.2, 8.3],
    }
}


@patch("felec.ingestion.weather.requests.get")
def test_fetch_previous_day_forecast_parses_response(mock_get):
    mock_get.return_value = Mock(json=lambda: FAKE_RESPONSE, raise_for_status=lambda: None)

    result = fetch_previous_day_forecast(48.85, 2.35, date(2024, 2, 1), date(2024, 2, 1))

    assert list(result["temperature"]) == [8.5, 8.2, 8.3]
    assert list(result["datetime"]) == FAKE_RESPONSE["hourly"]["time"]


@patch("felec.ingestion.weather.requests.get")
def test_fetch_previous_day_forecast_passes_correct_params(mock_get):
    mock_get.return_value = Mock(json=lambda: FAKE_RESPONSE, raise_for_status=lambda: None)

    fetch_previous_day_forecast(48.85, 2.35, date(2024, 2, 1), date(2024, 2, 2))

    _, kwargs = mock_get.call_args
    assert kwargs["params"]["models"] == "meteofrance_arome_france"
    assert kwargs["params"]["hourly"] == "temperature_2m_previous_day1"
    assert kwargs["params"]["timezone"] == "UTC"
    assert kwargs["params"]["start_date"] == "2024-02-01"
    assert kwargs["params"]["end_date"] == "2024-02-02"


@patch("felec.ingestion.weather.save_raw")
@patch("felec.ingestion.weather.fetch_previous_day_forecast")
def test_backfill_fetches_and_saves_every_city(mock_fetch, mock_save):
    mock_fetch.return_value = Mock()

    backfill(date(2024, 2, 1), date(2024, 2, 1))

    assert mock_fetch.call_count == len(CITIES)
    assert mock_save.call_count == len(CITIES)


@patch("felec.ingestion.weather.backfill")
@patch("felec.ingestion.weather.datetime")
def test_refresh_fetches_through_tomorrow(mock_datetime, mock_backfill):
    """Tomorrow's forecast is already available live from the Previous Runs
    API (verified against the real API while designing ADR 0007) -- refresh
    must include it, not stop at today, or the daily forecast pipeline would
    have no weather feature for the day it's trying to predict.
    """
    mock_datetime.now.return_value = datetime(2026, 9, 29, tzinfo=UTC)

    refresh()

    mock_backfill.assert_called_once_with(date(2026, 9, 22), date(2026, 9, 30))
