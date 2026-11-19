"""Unit tests for budget calculation tool."""

import pytest
from src.budget_tool import calculate_budget


def test_budget_calculation_single_day_trip():
    """Verify that a 1-day trip requires 0 hotel nights and charges 0 for accommodation."""
    res = calculate_budget(
        flight_price_per_person=3000.0,
        hotel_nightly_price=4000.0,
        trip_days=1,
        num_travelers=1,
        is_round_trip=True,
        daily_food_per_person=800.0,
        daily_transport=500.0,
        entrance_fees_per_person=200.0,
        other_buffer_per_person=1000.0,
    )
    summary = res["summary"]
    bd = res["breakdown"]

    assert summary["nights_count"] == 0
    assert summary["rooms_count"] == 0
    assert bd["accommodation"]["amount"] == 0.0
    assert "0 hotel nights" in bd["accommodation"]["notes"]

    # Flights: 3000 * 2 = 6000
    # Food: 800 * 1 = 800
    # Transport: 500 * 1 = 500
    # Entrance: 200
    # Buffer: 1000
    expected_total = 6000.0 + 0.0 + 800.0 + 500.0 + 200.0 + 1000.0
    assert summary["grand_total"] == expected_total
    assert summary["per_person_cost"] == expected_total


def test_budget_calculation_multi_day_single_traveler():
    """Verify formula for 1 traveler on a 3-day (2 nights) trip."""
    flight_one_way = 3000.0
    hotel_nightly = 4000.0
    trip_days = 3
    daily_food = 800.0
    daily_transport = 500.0
    entrance_fees = 300.0
    buffer = 1000.0

    res = calculate_budget(
        flight_price_per_person=flight_one_way,
        hotel_nightly_price=hotel_nightly,
        trip_days=trip_days,
        num_travelers=1,
        is_round_trip=True,
        daily_food_per_person=daily_food,
        daily_transport=daily_transport,
        entrance_fees_per_person=entrance_fees,
        other_buffer_per_person=buffer,
    )

    summary = res["summary"]
    bd = res["breakdown"]

    # Expected:
    # Flight: 3000 * 2 * 1 = 6000
    # Hotel: 4000 * 2 nights * 1 room = 8000
    # Food: 800 * 3 days * 1 = 2400
    # Transport: 500 * 3 days = 1500
    # Entrance: 300 * 1 = 300
    # Buffer: 1000 * 1 = 1000
    # Total = 6000 + 8000 + 2400 + 1500 + 300 + 1000 = 19200
    expected_total = 19200.0

    assert summary["nights_count"] == 2
    assert summary["rooms_count"] == 1
    assert summary["grand_total"] == expected_total
    assert summary["per_person_cost"] == expected_total
    assert bd["flights"]["amount"] == 6000.0
    assert bd["accommodation"]["amount"] == 8000.0
    assert bd["food_and_dining"]["amount"] == 2400.0
    assert bd["local_transport"]["amount"] == 1500.0
    assert bd["sightseeing_entrance"]["amount"] == 300.0
    assert bd["other_buffer"]["amount"] == 1000.0


def test_budget_calculation_multiple_travelers():
    """Verify scaling for 4 travelers (2 rooms needed, 3 nights for 4-day trip)."""
    res = calculate_budget(
        flight_price_per_person=2500.0,
        hotel_nightly_price=3500.0,
        trip_days=4,  # 3 nights
        num_travelers=4,
        is_round_trip=True,
        daily_food_per_person=1000.0,
        daily_transport=600.0,
        entrance_fees_per_person=200.0,
        other_buffer_per_person=500.0,
    )

    summary = res["summary"]
    bd = res["breakdown"]

    assert summary["travelers_count"] == 4
    assert summary["rooms_count"] == 2
    assert summary["nights_count"] == 3
    # Flights: 2500 * 2 * 4 = 20,000
    assert bd["flights"]["amount"] == 20000.0
    # Hotels: 3500 * 3 nights * 2 rooms = 21,000
    assert bd["accommodation"]["amount"] == 21000.0
    # Food: 1000 * 4 days * 4 people = 16,000
    assert bd["food_and_dining"]["amount"] == 16000.0
    # Transport: 600 * 4 days = 2,400
    assert bd["local_transport"]["amount"] == 2400.0
    # Entrance: 200 * 4 = 800
    assert bd["sightseeing_entrance"]["amount"] == 800.0
    # Buffer: 500 * 4 = 2,000
    assert bd["other_buffer"]["amount"] == 2000.0

    expected_sum = 20000.0 + 21000.0 + 16000.0 + 2400.0 + 800.0 + 2000.0
    assert summary["grand_total"] == expected_sum
    assert summary["per_person_cost"] == expected_sum / 4


def test_budget_one_way_flight():
    """Verify flight multiplier when is_round_trip is False."""
    res = calculate_budget(
        flight_price_per_person=3000.0,
        hotel_nightly_price=2000.0,
        trip_days=2,
        num_travelers=1,
        is_round_trip=False,
    )
    assert res["breakdown"]["flights"]["amount"] == 3000.0
    assert res["summary"]["nights_count"] == 1
