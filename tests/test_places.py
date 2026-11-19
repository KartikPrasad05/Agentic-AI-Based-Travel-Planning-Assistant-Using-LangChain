"""Unit tests for tourist attractions and places discovery tool."""

import pytest
from src.places_tool import search_places


def test_places_search_valid_city():
    """Verify discovery of attractions for a known tourist hub."""
    res = search_places(city="Delhi", limit=8)
    assert res["status"] == "success"
    assert len(res["results"]) > 0
    first = res["results"][0]
    assert "place_name" in first
    assert "category" in first
    assert "rating" in first
    assert "entrance_fee_inr" in first
    assert "recommendation_reason" in first


def test_places_search_category_filter():
    """Verify filtering attractions by category."""
    res = search_places(city="Goa", category="Beach", limit=5)
    assert res["status"] == "success"
    for place in res["results"]:
        assert "beach" in place["category"].lower()


def test_places_search_fee_cap():
    """Verify maximum entrance fee constraint."""
    res = search_places(city="Delhi", max_fee=100.0, limit=5)
    assert res["status"] == "success"
    for place in res["results"]:
        assert place["entrance_fee_inr"] <= 100.0


def test_places_search_deduplication():
    """Verify no duplicate attraction names are returned."""
    res = search_places(city="Delhi", limit=15)
    assert res["status"] == "success"
    names = [p["place_name"].lower() for p in res["results"]]
    assert len(names) == len(set(names)), "Attraction names should be unique in output."


def test_places_search_empty_city():
    """Verify error on empty city."""
    res = search_places(city="")
    assert res["status"] == "error"
