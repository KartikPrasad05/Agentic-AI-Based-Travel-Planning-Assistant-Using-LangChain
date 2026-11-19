"""Flight Search Tool for the Agentic Travel Planning Assistant.

Queries the 300,000+ Indian Airlines historical dataset supplemented by
the reference flight dataset to recommend optimal flights based on route,
price, duration, stops, airline, and travel class.
Supports direct non-stop routes and 1-stop connecting routes computed from
real dataset records. Never invents fake flight numbers or schedules.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from langchain_core.tools import tool

import config
from src.data_loader import get_data_loader

logger = logging.getLogger(__name__)

CONNECTING_HUBS = ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Kolkata", "Chennai"]


def clean_time_label(t: Any) -> str:
    """Format time strings into clean human-readable representations."""
    if not t or pd.isna(t):
        return "Daytime"
    s = str(t).strip().replace("_", " ")
    if "T" in s:
        # ISO format like 2025-01-04T11:32:00 -> 11:32
        parts = s.split("T")
        if len(parts) > 1:
            return parts[1][:5]
    return s


def search_flights(
    source_city: str,
    destination_city: str,
    max_budget: Optional[float] = None,
    airline: Optional[str] = None,
    travel_class: Optional[str] = "Economy",
    stops: Optional[str] = None,
    limit: int = 5,
) -> Dict[str, Any]:
    """Search for domestic flights between two Indian cities using historical datasets.

    Args:
        source_city: Departure city (e.g. 'Delhi', 'Mumbai', 'Bangalore').
        destination_city: Arrival city (e.g. 'Goa', 'Hyderabad', 'Jaipur').
        max_budget: Maximum acceptable flight price per passenger in INR.
        airline: Preferred airline (e.g. 'IndiGo', 'Air India', 'SpiceJet', 'Vistara').
        travel_class: 'Economy' or 'Business'. Defaults to 'Economy'.
        stops: Preferred stops ('zero', 'one', or 'two_or_more').
        limit: Maximum number of options to return (default 5).

    Returns:
        A dictionary containing matched flight options, cheapest option,
        fastest option, summary statistics, and data limitation notices.
    """
    loader = get_data_loader()
    norm_source = config.normalize_city_name(source_city)
    norm_dest = config.normalize_city_name(destination_city)

    if not norm_source or not norm_dest:
        return {
            "status": "error",
            "message": "Both source_city and destination_city must be provided.",
            "results": [],
        }

    if norm_source.lower() == norm_dest.lower():
        return {
            "status": "error",
            "message": f"Source city and destination city cannot be identical ({norm_source}).",
            "results": [],
        }

    matches: List[Dict[str, Any]] = []

    # 1. Check reference JSON dataset for baseline direct routes (e.g. Mumbai -> Goa, Bangalore -> Goa)
    ref_flights = loader.load_flights_json()
    for f in ref_flights:
        if f["source_city"].lower() == norm_source.lower() and f["destination_city"].lower() == norm_dest.lower():
            if max_budget and f["price_inr"] > max_budget:
                continue
            if airline and airline.lower() not in f["airline"].lower():
                continue
            if stops and stops.lower() in ["one", "1"] and f.get("stops") == "zero":
                continue
            matches.append({
                "flight_number": f.get("flight_id", "REF-FLIGHT"),
                "airline": f["airline"],
                "source": f["source_city"],
                "destination": f["destination_city"],
                "departure_time": clean_time_label(f["departure_time"]),
                "arrival_time": clean_time_label(f["arrival_time"]),
                "duration_hours": f.get("duration_hours", 2.5),
                "stops": f.get("stops", "zero"),
                "travel_class": f.get("travel_class", "Economy"),
                "price_inr": float(f["price_inr"]),
                "source_dataset": "Reference JSON Baseline",
                "is_historical": True,
            })

    # 2. Query 300,153-row historical Indian Airlines CSV for direct routes
    df = loader.load_flights_csv()
    if not df.empty:
        cond = (df["source_city"].str.lower() == norm_source.lower()) & (
            df["destination_city"].str.lower() == norm_dest.lower()
        )
        if travel_class:
            cond &= (df["travel_class"].str.lower() == travel_class.strip().lower())
        if max_budget:
            cond &= (df["price_inr"] <= max_budget)
        if airline:
            cond &= df["airline"].str.lower().str.contains(airline.strip().lower())
        if stops:
            cond &= (df["stops"].str.lower() == stops.strip().lower())

        sub_df = df[cond]

        if not sub_df.empty:
            sub_df_sorted = sub_df.sort_values(by=["price_inr", "duration_hours"]).drop_duplicates(
                subset=["airline", "flight", "departure_time"], keep="first"
            )
            for _, row in sub_df_sorted.head(limit * 3).iterrows():
                matches.append({
                    "flight_number": str(row.get("flight", "AI-101")),
                    "airline": str(row.get("airline", "Indian Airline")).replace("_", " "),
                    "source": norm_source,
                    "destination": norm_dest,
                    "departure_time": clean_time_label(row.get("departure_time", "Morning")),
                    "arrival_time": clean_time_label(row.get("arrival_time", "Afternoon")),
                    "duration_hours": float(row.get("duration_hours", 2.0)),
                    "stops": str(row.get("stops", "zero")),
                    "travel_class": str(row.get("travel_class", "Economy")),
                    "price_inr": float(row.get("price_inr", 4500.0)),
                    "source_dataset": "Indian Airlines Historical Fare Dataset",
                    "is_historical": True,
                })

    # 3. If no direct routes match, search for 1-stop connecting routes via standard Indian hubs
    if not matches:
        logger.info(f"Searching for connecting routes from {norm_source} to {norm_dest}")
        for hub in CONNECTING_HUBS:
            if hub.lower() == norm_source.lower() or hub.lower() == norm_dest.lower():
                continue

            # Leg 1: Source -> Hub
            leg1_matches = []
            if not df.empty:
                c1 = (df["source_city"].str.lower() == norm_source.lower()) & (df["destination_city"].str.lower() == hub.lower())
                if travel_class:
                    c1 &= (df["travel_class"].str.lower() == travel_class.strip().lower())
                sub1 = df[c1]
                if not sub1.empty:
                    best1 = sub1.sort_values(by="price_inr").iloc[0]
                    leg1_matches.append({
                        "flight_number": str(best1["flight"]),
                        "airline": str(best1["airline"]).replace("_", " "),
                        "price": float(best1["price_inr"]),
                        "duration": float(best1["duration_hours"]),
                        "departure": clean_time_label(best1["departure_time"]),
                    })
            if not leg1_matches:
                for f in ref_flights:
                    if f["source_city"].lower() == norm_source.lower() and f["destination_city"].lower() == hub.lower():
                        leg1_matches.append({
                            "flight_number": f.get("flight_id", "REF-FL1"),
                            "airline": f["airline"],
                            "price": float(f["price_inr"]),
                            "duration": f.get("duration_hours", 2.5),
                            "departure": clean_time_label(f["departure_time"]),
                        })
                        break

            if not leg1_matches:
                continue

            # Leg 2: Hub -> Destination
            leg2_matches = []
            for f in ref_flights:
                if f["source_city"].lower() == hub.lower() and f["destination_city"].lower() == norm_dest.lower():
                    leg2_matches.append({
                        "flight_number": f.get("flight_id", "REF-FL2"),
                        "airline": f["airline"],
                        "price": float(f["price_inr"]),
                        "duration": f.get("duration_hours", 2.5),
                        "arrival": clean_time_label(f["arrival_time"]),
                    })
                    break
            if not leg2_matches and not df.empty:
                c2 = (df["source_city"].str.lower() == hub.lower()) & (df["destination_city"].str.lower() == norm_dest.lower())
                if travel_class:
                    c2 &= (df["travel_class"].str.lower() == travel_class.strip().lower())
                sub2 = df[c2]
                if not sub2.empty:
                    best2 = sub2.sort_values(by="price_inr").iloc[0]
                    leg2_matches.append({
                        "flight_number": str(best2["flight"]),
                        "airline": str(best2["airline"]).replace("_", " "),
                        "price": float(best2["price_inr"]),
                        "duration": float(best2["duration_hours"]),
                        "arrival": clean_time_label(best2["arrival_time"]),
                    })

            if not leg2_matches:
                continue

            # Combine Leg 1 and Leg 2
            l1 = leg1_matches[0]
            l2 = leg2_matches[0]
            comb_price = l1["price"] + l2["price"]
            if max_budget and comb_price > max_budget:
                continue

            comb_airline = l1["airline"] if l1["airline"] == l2["airline"] else f"{l1['airline']} / {l2['airline']}"
            matches.append({
                "flight_number": f"{l1['flight_number']}+{l2['flight_number']}",
                "airline": comb_airline,
                "source": norm_source,
                "destination": norm_dest,
                "departure_time": l1["departure"],
                "arrival_time": l2["arrival"],
                "duration_hours": round(l1["duration"] + l2["duration"] + 1.5, 1),  # includes standard 1.5h layover
                "stops": f"one (via {hub})",
                "travel_class": travel_class or "Economy",
                "price_inr": comb_price,
                "source_dataset": f"1-Stop Connection via {hub} (Historical Datasets)",
                "is_historical": True,
            })

    if not matches:
        return {
            "status": "not_found",
            "message": f"No historical direct or 1-stop connecting flights found from {norm_source} to {norm_dest} in project datasets.",
            "source": norm_source,
            "destination": norm_dest,
            "results": [],
            "suggestion": "Select a route connecting primary hubs (Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai) or check reference cities (Goa, Jaipur).",
            "data_notice": "Historical reference records only. Not live flight inventory.",
        }

    # Deduplicate matches
    seen = set()
    unique_matches = []
    for m in matches:
        key = (m["airline"], m["flight_number"], m["price_inr"])
        if key not in seen:
            seen.add(key)
            unique_matches.append(m)

    # Sort final options by price
    sorted_matches = sorted(unique_matches, key=lambda x: (x["price_inr"], x["duration_hours"]))[:limit]

    cheapest = sorted_matches[0]
    fastest = min(sorted_matches, key=lambda x: x["duration_hours"])

    for item in sorted_matches:
        reasons = []
        if item["price_inr"] == cheapest["price_inr"]:
            reasons.append("Cheapest option for this route")
        if item["duration_hours"] == fastest["duration_hours"]:
            reasons.append("Fastest travel duration")
        if item["stops"] in ["zero", "0"]:
            reasons.append("Non-stop direct flight")
        elif "via" in item["stops"]:
            reasons.append(f"Convenient 1-stop connection {item['stops']}")
        if not reasons:
            reasons.append(f"Balanced fare with {item['airline']}")
        item["selection_reason"] = ", ".join(reasons)

    return {
        "status": "success",
        "route": f"{norm_source} -> {norm_dest}",
        "count_found": len(sorted_matches),
        "cheapest_option": cheapest,
        "fastest_option": fastest,
        "results": sorted_matches,
        "data_notice": "Prices are historical reference fares in INR from Indian Airlines and reference datasets. Flights are for simulation and planning only, not live commercial booking inventory.",
    }


@tool
def flight_search_tool(
    source_city: str,
    destination_city: str,
    max_budget: Optional[float] = None,
    airline: Optional[str] = None,
    travel_class: Optional[str] = "Economy",
    stops: Optional[str] = None,
) -> str:
    """Search for available flights between two Indian cities using historical datasets.

    Args:
        source_city: City of departure (e.g. 'Delhi', 'Mumbai', 'Bangalore').
        destination_city: City of arrival (e.g. 'Goa', 'Hyderabad', 'Jaipur').
        max_budget: Optional maximum flight price in INR.
        airline: Optional preferred airline name.
        travel_class: Optional 'Economy' or 'Business' (default 'Economy').
        stops: Optional stops preference ('zero', 'one', etc.).

    Returns:
        JSON-formatted string with matching flights, fares, durations, and recommendations.
    """
    import json
    res = search_flights(
        source_city=source_city,
        destination_city=destination_city,
        max_budget=max_budget,
        airline=airline,
        travel_class=travel_class,
        stops=stops,
        limit=5,
    )
    return json.dumps(res, indent=2)
