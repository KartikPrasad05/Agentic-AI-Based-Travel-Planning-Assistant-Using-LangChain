"""Budget Estimation Tool for the Agentic Travel Planning Assistant.

Computes comprehensive trip expenses using dataset-provided flight and hotel costs
combined with configurable daily allowances for food, local transport, entrance fees,
and emergency buffers.
Accurately differentiates 1-day trips (0 hotel nights) from multi-day trips (days - 1 nights).
"""

import json
import logging
from typing import Dict, Any, Optional
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


def calculate_budget(
    flight_price_per_person: float,
    hotel_nightly_price: float,
    trip_days: int,
    num_travelers: int = 1,
    is_round_trip: bool = True,
    daily_food_per_person: float = 800.0,
    daily_transport: float = 500.0,
    entrance_fees_per_person: float = 0.0,
    other_buffer_per_person: float = 1000.0,
) -> Dict[str, Any]:
    """Calculate an itemized travel budget for the entire trip.

    Formula:
      Total Budget = Flight Cost + Hotel Cost + Food Estimate + Local Transport
                     + Attraction Entrance Fees + Other Buffer Expenses

    Args:
        flight_price_per_person: One-way flight cost in INR.
        hotel_nightly_price: Nightly hotel accommodation rate in INR.
        trip_days: Total number of trip days (must be >= 1).
        num_travelers: Number of travelers (default 1).
        is_round_trip: Whether to account for both onward and return flights (default True).
        daily_food_per_person: Daily meal/dining allowance per traveler (default ₹800).
        daily_transport: Daily local cab/auto transit allowance (default ₹500).
        entrance_fees_per_person: Total monument/museum entry fees per person (default ₹0).
        other_buffer_per_person: Contingency/souvenir buffer per person (default ₹1000).

    Returns:
        Dictionary with itemized breakdowns, assumptions, and total estimated cost.
    """
    days = max(int(trip_days), 1)
    # Accurate night calculation: 1-day trip has 0 nights; D-day trip has D - 1 nights
    nights = max(days - 1, 0)
    travelers = max(int(num_travelers), 1)
    rooms_needed = ((travelers + 1) // 2) if nights > 0 else 0  # 1 room per 2 guests

    # Flight expenses
    flight_multiplier = 2 if is_round_trip else 1
    total_flight_cost = float(flight_price_per_person * flight_multiplier * travelers)

    # Hotel expenses (0 if same-day return trip)
    total_hotel_cost = float(hotel_nightly_price * nights * rooms_needed) if nights > 0 else 0.0

    # Food and dining estimate
    total_food_cost = float(daily_food_per_person * days * travelers)

    # Local transport estimate (group transit allowance per day)
    total_transport_cost = float(daily_transport * days)

    # Sightseeing / activities
    total_entrance_cost = float(entrance_fees_per_person * travelers)

    # Buffer / other expenses
    total_other_cost = float(other_buffer_per_person * travelers)

    # Grand total
    grand_total = (
        total_flight_cost
        + total_hotel_cost
        + total_food_cost
        + total_transport_cost
        + total_entrance_cost
        + total_other_cost
    )

    per_person_cost = grand_total / travelers

    hotel_notes = (
        f"₹{hotel_nightly_price:,.0f}/night x {nights} night(s) x {rooms_needed} room(s)"
        if nights > 0
        else "Same-day return trip (0 hotel nights required)"
    )

    return {
        "status": "success",
        "currency": "INR (₹)",
        "summary": {
            "grand_total": round(grand_total, 2),
            "per_person_cost": round(per_person_cost, 2),
            "trip_duration_days": days,
            "nights_count": nights,
            "travelers_count": travelers,
            "rooms_count": rooms_needed,
        },
        "breakdown": {
            "flights": {
                "amount": round(total_flight_cost, 2),
                "is_dataset_value": True,
                "notes": f"₹{flight_price_per_person:,.0f} x {travelers} traveler(s) x {'Round-trip (2 flights)' if is_round_trip else 'One-way'}",
            },
            "accommodation": {
                "amount": round(total_hotel_cost, 2),
                "is_dataset_value": nights > 0,
                "notes": hotel_notes,
            },
            "food_and_dining": {
                "amount": round(total_food_cost, 2),
                "is_dataset_value": False,
                "notes": f"Estimated ₹{daily_food_per_person:,.0f}/day x {days} days x {travelers} traveler(s)",
            },
            "local_transport": {
                "amount": round(total_transport_cost, 2),
                "is_dataset_value": False,
                "notes": f"Estimated ₹{daily_transport:,.0f}/day group cab/auto allowance x {days} days",
            },
            "sightseeing_entrance": {
                "amount": round(total_entrance_cost, 2),
                "is_dataset_value": True,
                "notes": f"Dataset ticket fees (₹{entrance_fees_per_person:,.0f} x {travelers} traveler(s))",
            },
            "other_buffer": {
                "amount": round(total_other_cost, 2),
                "is_dataset_value": False,
                "notes": f"Contingency / miscellaneous allowance (₹{other_buffer_per_person:,.0f} x {travelers} traveler(s))",
            },
        },
        "transparency_notice": (
            "Flight fares and monument ticket prices are referenced from project datasets. "
            "Food and local transport are realistic standard estimates and can be customized."
        ),
    }


@tool
def budget_estimation_tool(
    flight_price: float,
    hotel_nightly_rate: float,
    trip_days: int,
    num_travelers: int = 1,
    is_round_trip: bool = True,
) -> str:
    """Calculate total travel budget with itemized breakdown for flights, hotels, food, and activities.

    Args:
        flight_price: One-way flight price per person in INR.
        hotel_nightly_rate: Nightly hotel rate in INR.
        trip_days: Total number of days for the trip.
        num_travelers: Number of people traveling (default 1).
        is_round_trip: True if booking round-trip flights (default True).

    Returns:
        JSON-formatted string with complete itemized budget, totals, and per-person estimates.
    """
    res = calculate_budget(
        flight_price_per_person=flight_price,
        hotel_nightly_price=hotel_nightly_rate,
        trip_days=trip_days,
        num_travelers=num_travelers,
        is_round_trip=is_round_trip,
    )
    return json.dumps(res, indent=2)
