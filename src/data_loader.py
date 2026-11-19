"""Data loader module for the Agentic Travel Planning Assistant.

Loads, caches, and provides query interfaces for both the historical
processed CSVs and the project baseline JSON datasets.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

import config

logger = logging.getLogger(__name__)


class DataLoader:
    """Singleton loader for travel datasets with lazy loading and memory caching."""

    _instance: Optional["DataLoader"] = None

    def __init__(self):
        self._flights_df: Optional[pd.DataFrame] = None
        self._flights_json: Optional[List[Dict[str, Any]]] = None
        self._hotels_df: Optional[pd.DataFrame] = None
        self._hotels_json: Optional[List[Dict[str, Any]]] = None
        self._places_df: Optional[pd.DataFrame] = None
        self._places_json: Optional[List[Dict[str, Any]]] = None

    @classmethod
    def get_instance(cls) -> "DataLoader":
        if cls._instance is None:
            cls._instance = DataLoader()
        return cls._instance

    # -------------------------------------------------------------------------
    # FLIGHTS
    # -------------------------------------------------------------------------
    def load_flights_csv(self) -> pd.DataFrame:
        if self._flights_df is None:
            if not config.FLIGHTS_CLEAN_CSV.exists():
                logger.warning(f"Flights CSV not found at {config.FLIGHTS_CLEAN_CSV}")
                self._flights_df = pd.DataFrame()
            else:
                df = pd.read_csv(config.FLIGHTS_CLEAN_CSV)
                # Standardize column names and city casing
                df["source_city"] = df["source_city"].astype(str).str.strip().str.title()
                df["destination_city"] = df["destination_city"].astype(str).str.strip().str.title()
                df["airline"] = df["airline"].astype(str).str.strip()
                df["travel_class"] = df["travel_class"].astype(str).str.strip().str.title()
                df["price_inr"] = pd.to_numeric(df["price_inr"], errors="coerce")
                df["duration_hours"] = pd.to_numeric(df["duration_hours"], errors="coerce")
                df["stops"] = df["stops"].astype(str).str.strip().str.lower()
                self._flights_df = df
        return self._flights_df

    def load_flights_json(self) -> List[Dict[str, Any]]:
        if self._flights_json is None:
            if not config.FLIGHTS_JSON.exists():
                logger.warning(f"Flights JSON not found at {config.FLIGHTS_JSON}")
                self._flights_json = []
            else:
                with open(config.FLIGHTS_JSON, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                normalized = []
                for item in raw:
                    normalized.append({
                        "flight_id": item.get("flight_id", "REF_FLIGHT"),
                        "airline": item.get("airline", "Standard Airline").strip(),
                        "source_city": config.normalize_city_name(item.get("from", "")),
                        "destination_city": config.normalize_city_name(item.get("to", "")),
                        "departure_time": item.get("departure_time", "10:00"),
                        "arrival_time": item.get("arrival_time", "12:30"),
                        "price_inr": float(item.get("price", 4000)),
                        "travel_class": "Economy",
                        "duration_hours": 2.5,
                        "stops": "zero",
                        "days_left": None,
                        "is_reference": True,
                    })
                self._flights_json = normalized
        return self._flights_json

    # -------------------------------------------------------------------------
    # HOTELS
    # -------------------------------------------------------------------------
    def load_hotels_csv(self) -> pd.DataFrame:
        if self._hotels_df is None:
            if not config.HOTELS_CLEAN_CSV.exists():
                logger.warning(f"Hotels CSV not found at {config.HOTELS_CLEAN_CSV}")
                self._hotels_df = pd.DataFrame()
            else:
                df = pd.read_csv(config.HOTELS_CLEAN_CSV)
                df["city"] = df["city"].astype(str).str.strip().str.title()
                df["hotel_name"] = df["hotel_name"].astype(str).str.strip()
                df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")
                df["review_rating"] = pd.to_numeric(df["review_rating"], errors="coerce")
                df["review_count"] = pd.to_numeric(df["review_count"], errors="coerce")
                df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
                df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
                df["hotel_facilities"] = df["hotel_facilities"].fillna("").astype(str)
                df["room_facilities"] = df["room_facilities"].fillna("").astype(str)
                self._hotels_df = df
        return self._hotels_df

    def load_hotels_json(self) -> List[Dict[str, Any]]:
        if self._hotels_json is None:
            if not config.HOTELS_JSON.exists():
                logger.warning(f"Hotels JSON not found at {config.HOTELS_JSON}")
                self._hotels_json = []
            else:
                with open(config.HOTELS_JSON, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                normalized = []
                for item in raw:
                    normalized.append({
                        "hotel_id": item.get("hotel_id", ""),
                        "hotel_name": item.get("name", "Hotel").strip(),
                        "city": config.normalize_city_name(item.get("city", "")),
                        "star_rating": float(item.get("stars", 3)),
                        "review_rating": float(item.get("stars", 3)),
                        "price_per_night": float(item.get("price_per_night", 3500)),
                        "amenities": [a.lower().strip() for a in item.get("amenities", [])],
                        "address": f"Central {item.get('city', '')}",
                        "is_reference": True,
                        "has_verified_price": True,
                    })
                self._hotels_json = normalized
        return self._hotels_json

    # -------------------------------------------------------------------------
    # PLACES / ATTRACTIONS
    # -------------------------------------------------------------------------
    def load_places_csv(self) -> pd.DataFrame:
        if self._places_df is None:
            if not config.PLACES_CLEAN_CSV.exists():
                logger.warning(f"Places CSV not found at {config.PLACES_CLEAN_CSV}")
                self._places_df = pd.DataFrame()
            else:
                df = pd.read_csv(config.PLACES_CLEAN_CSV)
                df["city"] = df["city"].astype(str).str.strip().str.title()
                df["place_name"] = df["place_name"].astype(str).str.strip()
                df["type"] = df["type"].astype(str).str.strip().str.title()
                df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(4.0)
                df["entrance_fee_inr"] = pd.to_numeric(df["entrance_fee_inr"], errors="coerce").fillna(0.0)
                df["visit_duration_hours"] = pd.to_numeric(df["visit_duration_hours"], errors="coerce").fillna(2.0)
                df["weekly_off"] = df["weekly_off"].fillna("None").astype(str)
                df["dslr_allowed"] = df["dslr_allowed"].fillna("Yes").astype(str)
                df["best_time_to_visit"] = df["best_time_to_visit"].fillna("All Year").astype(str)
                self._places_df = df
        return self._places_df

    def load_places_json(self) -> List[Dict[str, Any]]:
        if self._places_json is None:
            if not config.PLACES_JSON.exists():
                logger.warning(f"Places JSON not found at {config.PLACES_JSON}")
                self._places_json = []
            else:
                with open(config.PLACES_JSON, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                normalized = []
                for item in raw:
                    normalized.append({
                        "place_id": item.get("place_id", ""),
                        "place_name": item.get("name", "Attraction").strip(),
                        "city": config.normalize_city_name(item.get("city", "")),
                        "type": item.get("type", "Sightseeing").title(),
                        "rating": float(item.get("rating", 4.0)),
                        "entrance_fee_inr": 0.0,
                        "visit_duration_hours": 2.0,
                        "is_reference": True,
                    })
                self._places_json = normalized
        return self._places_json

    def get_city_hotel_baseline_price(self, city: str) -> float:
        """Returns the median nightly price for a city from reference hotels or fallback."""
        norm_city = config.normalize_city_name(city)
        ref_hotels = self.load_hotels_json()
        city_prices = [h["price_per_night"] for h in ref_hotels if h["city"] == norm_city]
        if city_prices:
            return float(np.median(city_prices))
        all_prices = [h["price_per_night"] for h in ref_hotels if "price_per_night" in h]
        return float(np.median(all_prices)) if all_prices else 3500.0


# Module-level convenience functions
def get_data_loader() -> DataLoader:
    return DataLoader.get_instance()
