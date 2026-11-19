"""Configuration module for the Agentic Travel Planning Assistant.

Defines file paths, dataset locations, API endpoints, city coordinates,
and default settings.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Root directory of the project
BASE_DIR = Path(__file__).resolve().parent

# Check for data directories: first look in BASE_DIR, fallback to travel_dataset_prepared
def resolve_dir(relative_name: str) -> Path:
    p = BASE_DIR / relative_name
    if p.exists():
        return p
    alt = BASE_DIR / "travel_dataset_prepared" / relative_name
    if alt.exists():
        return alt
    p.mkdir(parents=True, exist_ok=True)
    return p

DATA_DIR = resolve_dir("data")
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_DIR = DATA_DIR / "raw"
REFERENCE_DATA_DIR = resolve_dir("reference")
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset file paths
FLIGHTS_CLEAN_CSV = PROCESSED_DATA_DIR / "flights_clean.csv"
HOTELS_CLEAN_CSV = PROCESSED_DATA_DIR / "hotels_clean.csv"
PLACES_CLEAN_CSV = PROCESSED_DATA_DIR / "places_clean.csv"

FLIGHTS_JSON = REFERENCE_DATA_DIR / "flights.json"
HOTELS_JSON = REFERENCE_DATA_DIR / "hotels.json"
PLACES_JSON = REFERENCE_DATA_DIR / "places.json"

# LLM Configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Open-Meteo Weather API
OPEN_METEO_BASE_URL = "https://api.open-meteo.com/v1/forecast"
WEATHER_TIMEOUT_SECONDS = 6

# Coordinates for major Indian travel hubs & destinations (Latitude, Longitude)
CITY_COORDINATES = {
    "delhi": (28.6139, 77.2090),
    "mumbai": (19.0760, 72.8777),
    "bangalore": (12.9716, 77.5946),
    "goa": (15.2993, 74.1240),
    "hyderabad": (17.3850, 78.4867),
    "kolkata": (22.5726, 88.3639),
    "chennai": (13.0827, 80.2707),
    "jaipur": (26.9124, 75.7873),
    "ahmedabad": (23.0225, 72.5714),
    "pune": (18.5204, 73.8567),
    "coorg": (12.3375, 75.8069),
    "alleppey": (9.4981, 76.3388),
    "agra": (27.1767, 78.0081),
    "varanasi": (25.3176, 82.9739),
    "kochi": (9.9312, 76.2673),
    "manali": (32.2432, 77.1892),
    "shimla": (31.1048, 77.1734),
    "udaipur": (24.5854, 73.7125),
    "srinagar": (34.0837, 74.7973),
    "amritsar": (31.6340, 74.8723),
    "visakhapatnam": (17.6868, 83.2185),
    "leh": (34.1526, 77.5771),
}

# Standardized city alias mapping
CITY_ALIASES = {
    "bengaluru": "Bangalore",
    "bangalore": "Bangalore",
    "bombay": "Mumbai",
    "mumbai": "Mumbai",
    "new delhi": "Delhi",
    "delhi": "Delhi",
    "madras": "Chennai",
    "chennai": "Chennai",
    "calcutta": "Kolkata",
    "kolkata": "Kolkata",
    "hyderabad": "Hyderabad",
    "goa": "Goa",
    "jaipur": "Jaipur",
    "ahmedabad": "Ahmedabad",
    "pune": "Pune",
    "coorg": "Coorg",
    "alleppey": "Alleppey",
    "alappuzha": "Alleppey",
    "agra": "Agra",
    "varanasi": "Varanasi",
    "kochi": "Kochi",
    "cochin": "Kochi",
    "udaipur": "Udaipur",
    "amritsar": "Amritsar",
    "visakhapatnam": "Visakhapatnam",
    "vizag": "Visakhapatnam",
    "leh": "Leh",
}

def normalize_city_name(city: str) -> str:
    """Normalize city names to match dataset conventions."""
    if not city or not isinstance(city, str):
        return ""
    cleaned = city.strip().lower()
    return CITY_ALIASES.get(cleaned, city.strip().title())
