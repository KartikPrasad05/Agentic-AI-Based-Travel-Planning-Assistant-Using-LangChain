"""Unit tests for end-to-end itinerary generator and schema validation."""

import pytest
from src.itinerary_generator import get_itinerary_generator
from src.prompts import CompleteTravelItinerary


def test_build_itinerary_delhi_to_goa_multi_day():
    """Verify end-to-end synthesis of a 3-day Delhi to Goa trip (2 hotel nights)."""
    gen = get_itinerary_generator()
    itinerary = gen.build_itinerary(
        source_city="Delhi",
        destination_city="Goa",
        duration_days=3,
        num_travelers=2,
        total_budget=50000.0,
    )

    # Schema validation
    assert isinstance(itinerary, CompleteTravelItinerary)
    assert itinerary.source_city == "Delhi"
    assert itinerary.destination_city == "Goa"
    assert itinerary.trip_duration_days == 3
    assert itinerary.num_travelers == 2

    # Flight details (connecting or direct)
    assert itinerary.flight.source == "Delhi"
    assert itinerary.flight.destination == "Goa"
    assert itinerary.flight.price_inr > 0

    # Hotel details
    assert itinerary.hotel.city == "Goa"
    assert itinerary.hotel.price_per_night_inr > 0

    # Schedule: 3 days
    assert len(itinerary.day_wise_schedule) == 3
    # Check Day 3 (return day) has departure activity
    day3_acts = [a.category for a in itinerary.day_wise_schedule[2].activities]
    assert "Departure" in day3_acts

    # Budget: 2 nights for 3 days
    assert itinerary.budget.grand_total_inr > 0
    assert itinerary.budget.hotel_total_inr == itinerary.hotel.price_per_night_inr * 2 * 1  # 2 nights, 1 room for 2 guests


def test_build_itinerary_single_day_trip():
    """Verify end-to-end synthesis of a 1-day express trip (0 hotel nights)."""
    gen = get_itinerary_generator()
    itinerary = gen.build_itinerary(
        source_city="Delhi",
        destination_city="Mumbai",
        duration_days=1,
        num_travelers=1,
    )

    assert itinerary.trip_duration_days == 1
    assert len(itinerary.day_wise_schedule) == 1
    # Check budget has 0 hotel cost
    assert itinerary.budget.hotel_total_inr == 0.0

    # Day 1 should have arrival and departure on the same day
    day1_cats = [a.category for a in itinerary.day_wise_schedule[0].activities]
    assert "Logistics" in day1_cats
    assert "Departure" in day1_cats


def test_save_and_serialize_itinerary(tmp_path):
    """Verify JSON dumping and reloading match schema."""
    gen = get_itinerary_generator()
    itinerary = gen.build_itinerary(
        source_city="Mumbai",
        destination_city="Jaipur",
        duration_days=2,
        num_travelers=1,
    )

    json_str = itinerary.model_dump_json()
    assert "Mumbai" in json_str
    assert "Jaipur" in json_str

    # Reload from JSON
    reloaded = CompleteTravelItinerary.model_validate_json(json_str)
    assert reloaded.source_city == itinerary.source_city
    assert reloaded.budget.grand_total_inr == itinerary.budget.grand_total_inr
