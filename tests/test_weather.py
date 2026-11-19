"""Unit tests for Open-Meteo weather tool."""

import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from src.weather_tool import get_weather_forecast, get_weather_code_description


def test_weather_code_descriptions():
    """Verify that WMO codes are mapped to clear human-readable strings."""
    assert "Clear" in get_weather_code_description(0)
    assert "Cloudy" in get_weather_code_description(2)
    assert "Rain" in get_weather_code_description(61)
    assert "Thunderstorm" in get_weather_code_description(95)


def test_weather_forecast_live_or_fallback():
    """Verify weather retrieval for a major city."""
    res = get_weather_forecast(city="Goa", days=5)
    assert "status" in res
    assert res["city"] == "Goa"
    if res["status"] == "success":
        assert len(res["daily_forecast"]) == 5
        day1 = res["daily_forecast"][0]
        assert "condition" in day1
        assert "date" in day1
        assert "travel_advice" in day1


def test_weather_forecast_date_range():
    """Verify weather forecast respecting specific upcoming start and end dates."""
    tomorrow = (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
    three_days = (date.today() + timedelta(days=3)).strftime("%Y-%m-%d")

    res = get_weather_forecast(city="Delhi", start_date=tomorrow, end_date=three_days)
    if res["status"] == "success":
        dates = [d["date"] for d in res["daily_forecast"]]
        assert tomorrow in dates
        assert res["is_live_forecast"] is True


def test_weather_forecast_beyond_horizon():
    """Verify honest beyond_horizon status when dates exceed Open-Meteo 16-day limit."""
    future_start = (date.today() + timedelta(days=30)).strftime("%Y-%m-%d")
    future_end = (date.today() + timedelta(days=33)).strftime("%Y-%m-%d")

    res = get_weather_forecast(city="Jaipur", start_date=future_start, end_date=future_end)
    assert res["status"] == "beyond_horizon"
    assert res["is_live_forecast"] is False
    assert "16 days" in res["data_notice"]
    assert len(res["daily_forecast"]) == 4


def test_weather_forecast_mocked_success():
    """Verify parsing logic against a simulated Open-Meteo payload."""
    mock_payload = {
        "daily": {
            "time": ["2026-12-01", "2026-12-02"],
            "weathercode": [0, 61],
            "temperature_2m_max": [31.5, 28.0],
            "temperature_2m_min": [21.0, 22.5],
            "precipitation_probability_max": [5, 75],
            "windspeed_10m_max": [12.0, 18.5],
        }
    }
    with patch("requests.get") as mock_get:
        mock_resp = MagicMock()
        mock_resp.raise_for_status.return_value = None
        mock_resp.json.return_value = mock_payload
        mock_get.return_value = mock_resp

        res = get_weather_forecast(city="Delhi", days=2)
        assert res["status"] == "success"
        assert len(res["daily_forecast"]) == 2
        d1 = res["daily_forecast"][0]
        assert d1["temp_max_c"] == 31.5
        assert d1["temp_min_c"] == 21.0
        assert "Clear" in d1["condition"]
        assert d1["precipitation_probability_pct"] == 5

        d2 = res["daily_forecast"][1]
        assert "Rain" in d2["condition"]
        assert d2["precipitation_probability_pct"] == 75
        assert "umbrella" in d2["travel_advice"].lower()


def test_weather_forecast_timeout_handling():
    """Verify graceful handling when Open-Meteo request times out."""
    import requests
    with patch("requests.get", side_effect=requests.exceptions.Timeout):
        res = get_weather_forecast(city="Jaipur", days=3)
        assert res["status"] == "timeout"
        assert "seasonal_guidance" in res


def test_weather_forecast_unknown_city():
    """Verify error status when coordinates are not available."""
    res = get_weather_forecast(city="UnknownNonExistentCityXYZ")
    assert res["status"] == "error"
