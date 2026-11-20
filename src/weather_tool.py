
"""Weather Tool for the Agentic Travel Planning Assistant.

Queries the free Open-Meteo API using city latitude and longitude to retrieve
daily weather conditions, maximum and minimum temperatures, precipitation
probabilities, and wind speeds.

Respects requested calendar dates, handles the 16-day forecast horizon honestly,
and never generates fake daily forecasts.
"""

import json
import logging
from datetime import datetime, date, timedelta
from typing import Dict, List, Any, Optional

import requests
from langchain_core.tools import tool

import config

logger = logging.getLogger(__name__)

# WMO Weather interpretation codes (WW)
WMO_CODE_MAP = {
    0: "Clear Sky / Sunny",
    1: "Mainly Clear",
    2: "Partly Cloudy",
    3: "Overcast",
    45: "Foggy",
    48: "Depositing Rime Fog",
    51: "Light Drizzle",
    53: "Moderate Drizzle",
    55: "Dense Drizzle",
    61: "Slight Rain",
    63: "Moderate Rain",
    65: "Heavy Rain",
    71: "Slight Snow Fall",
    73: "Moderate Snow Fall",
    75: "Heavy Snow Fall",
    77: "Snow Grains",
    80: "Slight Rain Showers",
    81: "Moderate Rain Showers",
    82: "Violent Rain Showers",
    85: "Slight Snow Showers",
    86: "Heavy Snow Showers",
    95: "Thunderstorm",
    96: "Thunderstorm with Slight Hail",
    99: "Thunderstorm with Heavy Hail",
}

# Standard seasonal climate averages for major Indian destinations.
# These are climate references, not day-specific weather forecasts.
SEASONAL_CLIMATE_DEFAULTS = {
    "goa": {
        "temp_min": 24.0,
        "temp_max": 33.0,
        "condition": "Sunny / Warm Beach Climate",
    },
    "delhi": {
        "temp_min": 18.0,
        "temp_max": 29.0,
        "condition": "Mild / Clear",
    },
    "mumbai": {
        "temp_min": 23.0,
        "temp_max": 32.0,
        "condition": "Humid / Warm Coastal",
    },
    "jaipur": {
        "temp_min": 16.0,
        "temp_max": 28.0,
        "condition": "Sunny / Pleasant Desert Climate",
    },
    "bangalore": {
        "temp_min": 17.0,
        "temp_max": 27.0,
        "condition": "Pleasant / Mild Breeze",
    },
    "hyderabad": {
        "temp_min": 19.0,
        "temp_max": 30.0,
        "condition": "Sunny / Dry",
    },
    "kolkata": {
        "temp_min": 20.0,
        "temp_max": 30.0,
        "condition": "Clear / Moderate Humidity",
    },
    "chennai": {
        "temp_min": 24.0,
        "temp_max": 31.0,
        "condition": "Warm Coastal / Humid",
    },
}


def get_weather_code_description(code: int) -> str:
    """Translate WMO numerical code into a human-readable condition."""
    try:
        return WMO_CODE_MAP.get(int(code), f"Weather Code {code}")
    except (TypeError, ValueError):
        return "Weather condition unavailable"


def _build_travel_advice(
    t_max: Optional[float],
    t_min: Optional[float],
    precip: Optional[float],
) -> str:
    """Generate practical travel advice using available forecast values.

    Missing values are never treated as zero.
    """

    if precip is not None and precip >= 50:
        return "High chance of rain; carry an umbrella or rainwear."

    if t_max is None and t_min is None:
        return (
            "Temperature data unavailable; check local conditions "
            "before travelling."
        )

    if t_max is not None and t_max >= 35:
        return (
            "Hot daytime conditions; stay hydrated and use sun protection."
        )

    if t_max is not None and t_max >= 30:
        if t_min is not None and t_min <= 16:
            return (
                "Warm daytime conditions with cooler mornings; "
                "stay hydrated and carry a light layer."
            )

        return (
            "Warm daytime conditions; stay hydrated and use sun protection."
        )

    if t_max is not None and t_max < 20:
        return "Cool daytime conditions; carry warm clothing."

    if t_min is not None and t_min <= 16:
        return "Cool mornings and evenings; carry a light jacket."

    return (
        "Mild conditions; suitable for outdoor activities, "
        "subject to local weather."
    )


def get_weather_forecast(
    city: str,
    days: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> Dict[str, Any]:
    """Retrieve weather information for the requested calendar dates.

    Open-Meteo's free forecast is used for dates within its available horizon.
    For dates beyond that horizon, typical seasonal climate averages are
    returned with an explicit non-forecast label.

    Args:
        city: Target city name.
        days: Number of days to fetch when dates are not supplied.
        start_date: Optional start date in YYYY-MM-DD format.
        end_date: Optional end date in YYYY-MM-DD format.

    Returns:
        Dictionary containing daily weather information and data-source status.
    """

    norm_city = config.normalize_city_name(city)
    lookup_key = norm_city.lower()

    # 1. Find city coordinates.
    coords = config.CITY_COORDINATES.get(lookup_key)

    if not coords:
        for key, value in config.CITY_COORDINATES.items():
            if key in lookup_key or lookup_key in key:
                coords = value
                break

    if not coords:
        return {
            "status": "error",
            "city": norm_city,
            "message": f"Coordinates not found for city '{norm_city}'.",
            "daily_forecast": [],
            "data_notice": (
                "Location coordinates unavailable for automated "
                "Open-Meteo query."
            ),
        }

    lat, lon = coords
    today = date.today()

    # 2. Parse requested date range.
    try:
        s_date = (
            datetime.strptime(start_date, "%Y-%m-%d").date()
            if start_date
            else today
        )

        e_date = (
            datetime.strptime(end_date, "%Y-%m-%d").date()
            if end_date
            else s_date + timedelta(days=max(days - 1, 0))
        )

    except ValueError:
        return {
            "status": "error",
            "city": norm_city,
            "message": "Invalid date format. Expected YYYY-MM-DD.",
            "daily_forecast": [],
            "data_notice": "Please provide valid travel dates.",
        }

    # Reject reversed date ranges.
    if e_date < s_date:
        return {
            "status": "error",
            "city": norm_city,
            "message": "End date cannot be earlier than start date.",
            "daily_forecast": [],
            "data_notice": "Please provide a valid date range.",
        }

    # Reject trips that have already ended.
    if e_date < today:
        return {
            "status": "error",
            "city": norm_city,
            "message": "Requested travel dates are in the past.",
            "daily_forecast": [],
            "data_notice": (
                "Weather forecasts are unavailable for past travel dates. "
                "Please provide today or a future date."
            ),
        }

    # Open-Meteo free forecast horizon: today + 15 days.
    max_forecast_date = today + timedelta(days=15)

    # 3. Entire requested trip is beyond the live forecast horizon.
    if s_date > max_forecast_date:

        logger.info(
            "Requested dates %s to %s are beyond the live forecast "
            "horizon for %s.",
            s_date,
            e_date,
            norm_city,
        )

        seasonal = SEASONAL_CLIMATE_DEFAULTS.get(
            lookup_key,
            {
                "temp_min": 20.0,
                "temp_max": 30.0,
                "condition": "General Seasonal Climate",
            },
        )

        seasonal_days = []
        current_date = s_date

        while current_date <= e_date:
            seasonal_days.append({
                "date": current_date.strftime("%Y-%m-%d"),
                "condition": (
                    f"Seasonal Average ({seasonal['condition']})"
                ),
                "temp_max_c": seasonal["temp_max"],
                "temp_min_c": seasonal["temp_min"],
                "temp_display": (
                    f"{seasonal['temp_min']}°C – "
                    f"{seasonal['temp_max']}°C (Seasonal Average)"
                ),
                "precipitation_probability_pct": None,
                "wind_speed_kmh": None,
                "is_live_forecast": False,
                "travel_advice": (
                    "Typical seasonal climate averages are shown, "
                    "not a day-specific forecast. Check the weather "
                    "closer to your travel date."
                ),
            })

            current_date += timedelta(days=1)

        return {
            "status": "beyond_horizon",
            "city": norm_city,
            "coordinates": {
                "latitude": lat,
                "longitude": lon,
            },
            "forecast_days_count": len(seasonal_days),
            "daily_forecast": seasonal_days,
            "is_live_forecast": False,
            "data_notice": (
                "Open-Meteo provides live forecasts up to 16 days ahead. "
                f"Trip dates ({s_date} to {e_date}) exceed this horizon; "
                "typical seasonal climate averages are displayed instead "
                "of fabricated daily weather forecasts."
            ),
            "source": (
                "Historical Seasonal Climate Guidance "
                "(Live Forecast Unavailable)"
            ),
        }

    # 4. Prepare Open-Meteo API request.
    params: Dict[str, Any] = {
        "latitude": lat,
        "longitude": lon,
        "daily": [
            "weathercode",
            "temperature_2m_max",
            "temperature_2m_min",
            "precipitation_probability_max",
            "windspeed_10m_max",
        ],
        "timezone": "auto",
    }

    clamped_start = max(s_date, today)
    clamped_end = min(e_date, max_forecast_date)

    if clamped_start <= clamped_end:
        params["start_date"] = clamped_start.strftime("%Y-%m-%d")
        params["end_date"] = clamped_end.strftime("%Y-%m-%d")
    else:
        params["forecast_days"] = min(max(days, 1), 14)

    # 5. Query Open-Meteo API.
    try:
        response = requests.get(
            config.OPEN_METEO_BASE_URL,
            params=params,
            timeout=config.WEATHER_TIMEOUT_SECONDS,
        )

        response.raise_for_status()
        data = response.json()

        daily = data.get("daily", {})

        dates = daily.get("time", [])
        codes = daily.get("weathercode", [])
        max_temps = daily.get("temperature_2m_max", [])
        min_temps = daily.get("temperature_2m_min", [])
        precips = daily.get("precipitation_probability_max", [])
        winds = daily.get("windspeed_10m_max", [])

        forecast_list = []

        for i in range(len(dates)):

            # Preserve missing API values instead of inventing defaults.
            wcode = codes[i] if i < len(codes) else None

            desc = (
                get_weather_code_description(wcode)
                if wcode is not None
                else "Weather condition unavailable"
            )

            t_max = max_temps[i] if i < len(max_temps) else None
            t_min = min_temps[i] if i < len(min_temps) else None
            precip = precips[i] if i < len(precips) else None
            wind = winds[i] if i < len(winds) else None

            forecast_list.append({
                "date": dates[i],
                "condition": desc,
                "temp_max_c": t_max,
                "temp_min_c": t_min,
                "temp_display": (
                    f"{t_min}°C – {t_max}°C"
                    if t_min is not None and t_max is not None
                    else "N/A"
                ),
                "precipitation_probability_pct": precip,
                "wind_speed_kmh": wind,
                "is_live_forecast": True,
                "travel_advice": _build_travel_advice(
                    t_max,
                    t_min,
                    precip,
                ),
            })

        return {
            "status": "success",
            "city": norm_city,
            "coordinates": {
                "latitude": lat,
                "longitude": lon,
            },
            "forecast_days_count": len(forecast_list),
            "daily_forecast": forecast_list,
            "is_live_forecast": True,
            "source": "Open-Meteo Free Forecast API (Live)",
        }

    except requests.exceptions.Timeout:
        logger.warning(
            "Open-Meteo API timed out for %s.",
            norm_city,
        )

        return {
            "status": "timeout",
            "city": norm_city,
            "message": "Open-Meteo weather service request timed out.",
            "daily_forecast": [],
            "is_live_forecast": False,
            "data_notice": (
                "Live weather data is temporarily unavailable because "
                "the weather service timed out. Check again later."
            ),
            "seasonal_guidance": (
                f"Typical seasonal climate reference for {norm_city}: "
                "22°C - 32°C. This is not a daily forecast."
            ),
            "source": "Fallback (API Timeout)",
        }

    except requests.exceptions.RequestException as e:
        logger.warning(
            "Open-Meteo API error for %s: %s",
            norm_city,
            e,
        )

        return {
            "status": "error",
            "city": norm_city,
            "message": f"Weather API error: {str(e)}",
            "daily_forecast": [],
            "is_live_forecast": False,
            "data_notice": (
                "Live weather data is unavailable due to a network "
                "or API error. Check again later."
            ),
            "seasonal_guidance": (
                f"Typical seasonal climate reference for {norm_city}: "
                "20°C - 30°C. This is not a daily forecast."
            ),
            "source": "Fallback (Network Error)",
        }

    except (ValueError, TypeError, KeyError) as e:
        logger.exception(
            "Unexpected weather response format for %s.",
            norm_city,
        )

        return {
            "status": "error",
            "city": norm_city,
            "message": f"Invalid weather API response: {str(e)}",
            "daily_forecast": [],
            "is_live_forecast": False,
            "data_notice": (
                "The weather service returned an unexpected response. "
                "Please check again later."
            ),
            "source": "Fallback (Invalid API Response)",
        }


@tool
def weather_lookup_tool(
    city: str,
    days: int = 5,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> str:
    """Retrieve weather information for a destination city.

    Args:
        city: Destination city name.
        days: Number of forecast days (1 to 10, default 5).
        start_date: Optional departure date YYYY-MM-DD.
        end_date: Optional return date YYYY-MM-DD.

    Returns:
        JSON-formatted string with daily weather conditions, temperatures,
        and precipitation probability where available.
    """

    res = get_weather_forecast(
        city=city,
        days=days,
        start_date=start_date,
        end_date=end_date,
    )

    return json.dumps(res, indent=2)
