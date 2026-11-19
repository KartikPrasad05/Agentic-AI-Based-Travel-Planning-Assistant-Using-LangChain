"""Unit tests for flight search tool and data processing."""

import pytest
from src.flight_tool import search_flights


def test_flight_search_direct_route():
    """Verify standard direct domestic route search returns matching flights."""
    res = search_flights(source_city="Delhi", destination_city="Mumbai", limit=5)
    assert res["status"] == "success"
    assert len(res["results"]) > 0
    first = res["results"][0]
    assert first["source"].lower() == "delhi"
    assert first["destination"].lower() == "mumbai"
    assert first["price_inr"] > 0
    assert first["is_historical"] is True
    assert "cheapest_option" in res
    assert "data_notice" in res


def test_flight_search_connecting_route():
    """Verify that a route without direct flights (Delhi -> Goa) finds real 1-stop connecting flights."""
    res = search_flights(source_city="Delhi", destination_city="Goa", limit=5)
    assert res["status"] == "success"
    assert len(res["results"]) > 0
    first = res["results"][0]
    assert first["source"].lower() == "delhi"
    assert first["destination"].lower() == "goa"
    assert "one" in first["stops"].lower() or "via" in first["stops"].lower()
    assert first["price_inr"] > 0
    assert "+" in first["flight_number"] or "FL" in first["flight_number"]


def test_flight_search_cheapest_ranking():
    """Verify that results are ordered by price and cheapest is marked."""
    res = search_flights(source_city="Bangalore", destination_city="Delhi", limit=5)
    assert res["status"] == "success"
    prices = [r["price_inr"] for r in res["results"]]
    assert prices == sorted(prices)
    assert res["cheapest_option"]["price_inr"] == min(prices)


def test_flight_search_budget_filter():
    """Verify maximum budget constraint is respected."""
    budget_cap = 4000.0
    res = search_flights(source_city="Delhi", destination_city="Mumbai", max_budget=budget_cap)
    if res["status"] == "success":
        for flight in res["results"]:
            assert flight["price_inr"] <= budget_cap


def test_flight_search_airline_filter():
    """Verify filtering by specific airline."""
    res = search_flights(source_city="Delhi", destination_city="Bangalore", airline="Vistara")
    if res["status"] == "success":
        for flight in res["results"]:
            assert "vistara" in flight["airline"].lower()


def test_flight_search_identical_cities():
    """Verify error when source and destination are identical."""
    res = search_flights(source_city="Delhi", destination_city="Delhi")
    assert res["status"] == "error"
    assert "identical" in res["message"].lower()


def test_flight_search_missing_city():
    """Verify error when source or destination is empty."""
    res = search_flights(source_city="", destination_city="Mumbai")
    assert res["status"] == "error"


def test_flight_search_unconnected_city():
    """Verify honest not_found response for an unknown city without fabricating data."""
    res = search_flights(source_city="Delhi", destination_city="NonExistentCityXYZ")
    assert res["status"] == "not_found"
    assert len(res["results"]) == 0
