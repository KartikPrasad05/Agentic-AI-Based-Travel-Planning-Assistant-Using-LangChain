"""Tourist Attraction and Places Discovery Tool for the Agentic Travel Planning Assistant.

Queries the Indian destinations attraction dataset (323 curated places)
supplemented by reference places, filtering by category, rating, entrance fee,
and visit duration.
"""

import json
import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from langchain_core.tools import tool

import config
from src.data_loader import get_data_loader

logger = logging.getLogger(__name__)


def search_places(
    city: str,
    category: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_fee: Optional[float] = None,
    max_duration_hours: Optional[float] = None,
    limit: int = 8,
) -> Dict[str, Any]:
    """Discover tourist attractions and points of interest for a destination.

    Args:
        city: Destination city (e.g. 'Goa', 'Delhi', 'Jaipur', 'Mumbai').
        category: Optional category filter (e.g. 'Beach', 'Fort', 'Temple', 'Museum', 'Lake', 'Palace').
        min_rating: Optional minimum tourist rating (1.0 to 5.0).
        max_fee: Optional maximum entry fee in INR.
        max_duration_hours: Optional maximum visit duration in hours.
        limit: Maximum number of attractions to return (default 8).

    Returns:
        Dictionary with matched attractions, entry fees, ratings, visit times,
        and recommendations.
    """
    loader = get_data_loader()
    norm_city = config.normalize_city_name(city)

    if not norm_city:
        return {
            "status": "error",
            "message": "Destination city must be provided.",
            "results": [],
        }

    results: List[Dict[str, Any]] = []

    # 1. Query rich places_clean.csv (323 attraction spots across India)
    df_places = loader.load_places_csv()
    if not df_places.empty:
        city_places = df_places[df_places["city"].str.lower() == norm_city.lower()]
        for _, row in city_places.iterrows():
            attraction_name = str(row.get("place_name", "")).strip()
            place_type = str(row.get("type", "Attraction")).strip().title()
            rating = float(row.get("rating", 4.0))
            fee = float(row.get("entrance_fee_inr", 0.0))
            duration = float(row.get("visit_duration_hours", 2.0))
            weekly_off = str(row.get("weekly_off", "None")).strip()
            dslr = str(row.get("dslr_allowed", "Yes")).strip()
            best_time = str(row.get("best_time_to_visit", "All Year")).strip()

            if min_rating and rating < min_rating:
                continue
            if max_fee is not None and fee > max_fee:
                continue
            if max_duration_hours and duration > max_duration_hours:
                continue
            if category and category.lower() not in place_type.lower():
                continue

            results.append({
                "place_name": attraction_name,
                "city": norm_city,
                "category": place_type,
                "rating": rating,
                "entrance_fee_inr": fee,
                "visit_duration_hours": duration,
                "weekly_off": weekly_off if weekly_off.lower() != "nan" else "None",
                "dslr_allowed": dslr,
                "best_time_to_visit": best_time,
                "source_dataset": "Destinations Dataset",
            })

    # 2. Query reference places.json baseline
    ref_places = loader.load_places_json()
    for p in ref_places:
        if p["city"].lower() == norm_city.lower():
            p_name = p.get("place_name", "Attraction")
            p_type = p.get("type", "Sightseeing").title()
            p_rating = float(p.get("rating", 4.0))

            if min_rating and p_rating < min_rating:
                continue
            if category and category.lower() not in p_type.lower():
                continue

            results.append({
                "place_name": p_name,
                "city": norm_city,
                "category": p_type,
                "rating": p_rating,
                "entrance_fee_inr": 0.0,
                "visit_duration_hours": 2.0,
                "weekly_off": "None",
                "dslr_allowed": "Yes",
                "best_time_to_visit": "Daytime",
                "source_dataset": "Reference Places Baseline",
            })

    if not results:
        return {
            "status": "not_found",
            "message": f"No tourist attractions found for {norm_city} matching the requested criteria.",
            "city": norm_city,
            "results": [],
            "suggestion": "Try clearing category filters or lowering minimum rating.",
        }

    # Deduplicate by place name
    seen = set()
    unique_places = []
    for item in results:
        key = item["place_name"].lower().strip()
        if key not in seen:
            seen.add(key)
            unique_places.append(item)

    # Sort by rating descending
    sorted_places = sorted(unique_places, key=lambda x: -x["rating"])[:limit]

    # Assign recommendation reason
    for p in sorted_places:
        reasons = [f"Popular {p['category']} rated {p['rating']}★"]
        if p["entrance_fee_inr"] == 0:
            reasons.append("Free entry")
        else:
            reasons.append(f"Entry fee ₹{int(p['entrance_fee_inr'])}")
        if p["weekly_off"] != "None":
            reasons.append(f"Closed on {p['weekly_off']}")
        p["recommendation_reason"] = " • ".join(reasons)

    return {
        "status": "success",
        "city": norm_city,
        "count_found": len(sorted_places),
        "results": sorted_places,
        "data_notice": "Attractions derived from Indian tourism records. Hours and fees are reference values.",
    }


@tool
def places_discovery_tool(
    city: str,
    category: Optional[str] = None,
    min_rating: Optional[float] = None,
    max_fee: Optional[float] = None,
) -> str:
    """Discover tourist attractions, sightseeing spots, and activities for a destination.

    Args:
        city: Destination city name (e.g. 'Goa', 'Delhi', 'Jaipur', 'Mumbai').
        category: Optional category filter (e.g. 'Beach', 'Fort', 'Temple', 'Museum').
        min_rating: Optional minimum rating (e.g. 4.0).
        max_fee: Optional maximum entrance ticket fee in INR.

    Returns:
        JSON-formatted string with recommended attractions, categories, ratings, and fees.
    """
    res = search_places(
        city=city,
        category=category,
        min_rating=min_rating,
        max_fee=max_fee,
        limit=8,
    )
    return json.dumps(res, indent=2)
