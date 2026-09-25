from datetime import date
from unittest.mock import Mock, patch

from felec.ingestion.weather import CITIES, backfill, fetch_previous_day_forecast

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
