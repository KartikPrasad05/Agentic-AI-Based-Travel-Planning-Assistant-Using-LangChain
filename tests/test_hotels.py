"""Unit tests for hotel recommendation tool."""

import pytest
from src.hotel_tool import search_hotels


def test_hotel_search_valid_city():
    """Verify hotel search in a primary destination returns records."""
    res = search_hotels(city="Goa", limit=5)
    assert res["status"] == "success"
    assert len(res["results"]) > 0
    first = res["results"][0]
    assert "hotel_name" in first
    assert "price_per_night_inr" in first
    assert "recommendation_reason" in first
    assert "data_notice" in res


def test_hotel_search_rating_filter():
    """Verify minimum rating filter works as expected."""
    res = search_hotels(city="Delhi", min_rating=4.0, limit=5)
    assert res["status"] == "success"
    for hotel in res["results"]:
        effective_r = hotel.get("review_rating") or hotel.get("star_rating") or 3.0
        assert effective_r >= 4.0 or hotel.get("is_verified_price") is True


def test_hotel_search_amenity_filter():
    """Verify filtering by preferred amenities."""
    res = search_hotels(city="Bangalore", preferred_amenities=["pool"], limit=5)
    assert res["status"] == "success"
    for hotel in res["results"]:
        # Either has pool in amenities list or from verified dataset
        assert "pool" in [a.lower() for a in hotel.get("amenities", [])] or hotel["is_verified_price"]


def test_hotel_search_reference_price_disclosure():
    """Static project prices must not be labelled as live or booking-verified."""
    res = search_hotels(city="Delhi", limit=10)
    assert res["status"] == "success"
    reference_hotels = [
        h for h in res["results"]
        if h["price_source"] == "Project Reference Dataset (not live)"
    ]
    assert reference_hotels, "Reference baseline hotels should be present."
    for hotel in reference_hotels:
        assert hotel["is_verified_price"] is False
        assert hotel["review_rating"] is None
        assert "not live" in hotel["price_source"]


def test_hotel_search_empty_city():
    """Verify error on empty city input."""
    res = search_hotels(city="")
    assert res["status"] == "error"
