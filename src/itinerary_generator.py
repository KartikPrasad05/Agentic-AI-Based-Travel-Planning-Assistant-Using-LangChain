
"""Itinerary Generator module for the Agentic Travel Planning Assistant.

Synthesizes flight, hotel, tourist attraction, weather, and budget data
into a coherent, realistically paced day-wise itinerary validated
with Pydantic schemas.

Avoids late-arrival conflicts, excessive sightseeing, fabricated data,
and misleading booking or pricing claims.
"""

import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional

import config

from src.flight_tool import search_flights
from src.hotel_tool import search_hotels
from src.places_tool import search_places
from src.weather_tool import get_weather_forecast
from src.budget_tool import calculate_budget

from src.prompts import (
    CompleteTravelItinerary,
    FlightRecommendation,
    HotelRecommendation,
    ActivityItem,
    WeatherForecastDay,
    DayItinerary,
    BudgetBreakdown,
    SelectionJustifications,
)

logger = logging.getLogger(__name__)


def _parse_arrival_hour(value: Any) -> Optional[int]:
    """Extract an arrival hour from common time formats."""

    if value is None:
        return None

    text = str(value).strip()

    if not text or text.lower() in {
        "unknown", "not available", "not provided",
        "flexible", "n/a", "none",
    }:
        return None

    formats = (
        "%H:%M",
        "%H:%M:%S",
        "%I:%M %p",
        "%I:%M:%S %p",
    )

    for fmt in formats:
        try:
            return datetime.strptime(text, fmt).hour
        except ValueError:
            continue

    try:
        return datetime.fromisoformat(
            text.replace("Z", "+00:00")
        ).hour
    except ValueError:
        return None


def _safe_float(value: Any, default: float = 0.0) -> float:
    """Convert a value to float safely."""

    try:
        return default if value is None else float(value)
    except (TypeError, ValueError):
        return default


def _format_rating(value: Any) -> str:
    """Format an available rating without inventing a default."""

    if value is None or value == "":
        return "Not available"

    return str(value)


class ItineraryGenerator:
    """Orchestrates tools and synthesizes validated travel itineraries."""

    def __init__(self):
        pass

    def build_itinerary(
        self,
        source_city: str,
        destination_city: str,
        duration_days: int = 3,
        start_date: Optional[str] = None,
        num_travelers: int = 1,
        total_budget: Optional[float] = None,
        travel_class: str = "Economy",
        preferred_category: Optional[str] = None,
        preferred_amenities: Optional[List[str]] = None,
        daily_food_per_person: float = 800.0,
        daily_transport: float = 500.0,
        other_buffer_per_person: float = 1000.0,
        selected_flight_data: Optional[Dict[str, Any]] = None,
        selected_hotel_data: Optional[Dict[str, Any]] = None,
    ) -> CompleteTravelItinerary:
        """Construct an end-to-end validated travel itinerary."""

        days = max(1, min(int(duration_days), 14))
        travelers = max(1, int(num_travelers))

        # 1. Trip dates
        if not start_date:
            base_date = date.today() + timedelta(days=1)
        else:
            try:
                base_date = datetime.strptime(
                    start_date, "%Y-%m-%d"
                ).date()
            except ValueError:
                base_date = date.today() + timedelta(days=1)

        end_date = base_date + timedelta(days=days - 1)

        start_date_str = base_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # 2. Flight selection
        best_flight_data = selected_flight_data

        if not best_flight_data:
            flight_res = search_flights(
                source_city=source_city,
                destination_city=destination_city,
                travel_class=travel_class,
                limit=5,
            )

            if (
                flight_res.get("status") == "success"
                and flight_res.get("results")
            ):
                best_flight_data = flight_res.get("cheapest_option")

        if best_flight_data:
            flight_model = FlightRecommendation(
                airline=best_flight_data.get(
                    "airline", "Indian Domestic Airline"
                ),
                flight_number=best_flight_data.get(
                    "flight_number", "FLIGHT"
                ),
                source=best_flight_data.get(
                    "source", config.normalize_city_name(source_city)
                ),
                destination=best_flight_data.get(
                    "destination",
                    config.normalize_city_name(destination_city),
                ),
                departure_time=str(
                    best_flight_data.get("departure_time")
                    or "Not provided"
                ),
                arrival_time=str(
                    best_flight_data.get("arrival_time")
                    or "Not provided"
                ),
                duration_hours=_safe_float(
                    best_flight_data.get("duration_hours")
                ),
                stops=str(
                    best_flight_data.get("stops") or "Not provided"
                ),
                travel_class=best_flight_data.get(
                    "travel_class", travel_class
                ),
                price_inr=_safe_float(
                    best_flight_data.get("price_inr")
                ),
                selection_reason=best_flight_data.get(
                    "selection_reason",
                    "Historical route option from dataset",
                ),
                is_historical=True,
            )
        else:
            flight_model = FlightRecommendation(
                airline="No direct dataset route",
                flight_number="N/A",
                source=config.normalize_city_name(source_city),
                destination=config.normalize_city_name(destination_city),
                departure_time="Flexible",
                arrival_time="Flexible",
                duration_hours=0.0,
                stops="N/A",
                travel_class=travel_class,
                price_inr=0.0,
                selection_reason=(
                    "No flight records found in project historical "
                    "datasets for this route; fare excluded "
                    "from calculation."
                ),
                is_historical=True,
            )

        arrival_hour = _parse_arrival_hour(
            flight_model.arrival_time
        )

        # 3. Hotel selection
        best_hotel_data = selected_hotel_data

        if not best_hotel_data:
            hotel_res = search_hotels(
                city=destination_city,
                preferred_amenities=preferred_amenities,
                limit=5,
            )

            if (
                hotel_res.get("status") == "success"
                and hotel_res.get("results")
            ):
                best_hotel_data = hotel_res["results"][0]

        if best_hotel_data:
            hotel_model = HotelRecommendation(
                hotel_name=best_hotel_data["hotel_name"],
                city=best_hotel_data["city"],
                address=best_hotel_data.get(
                    "address", destination_city
                ),
                star_rating=best_hotel_data.get("star_rating"),
                review_rating=best_hotel_data.get("review_rating"),
                price_per_night_inr=_safe_float(
                    best_hotel_data.get("price_per_night_inr")
                ),
                amenities=best_hotel_data.get("amenities") or [],
                recommendation_reason=best_hotel_data.get(
                    "recommendation_reason",
                    "Property selected from destination catalogue.",
                ),
                is_verified_price=best_hotel_data.get(
                    "is_verified_price", False
                ),
            )
        else:
            hotel_model = HotelRecommendation(
                hotel_name="No catalogue property found",
                city=config.normalize_city_name(destination_city),
                address=(
                    f"Central "
                    f"{config.normalize_city_name(destination_city)}"
                ),
                star_rating=None,
                review_rating=None,
                price_per_night_inr=0.0,
                amenities=[],
                recommendation_reason=(
                    "No hotel records exist in dataset for this city."
                ),
                is_verified_price=False,
            )

        # 4. Attraction selection
        places_res = search_places(
            city=destination_city,
            category=preferred_category,
            limit=max(days * 3, 6),
        )

        raw_places = places_res.get("results", [])

        if not isinstance(raw_places, list):
            raw_places = []

        raw_places = [
            place for place in raw_places
            if isinstance(place, dict) and place.get("place_name")
        ]

        # 5. Weather forecast
        weather_res = get_weather_forecast(
            city=destination_city,
            days=days,
            start_date=start_date_str,
            end_date=end_date_str,
        )

        weather_by_date: Dict[str, Dict[str, Any]] = {}

        for w in weather_res.get("daily_forecast", []):
            if isinstance(w, dict) and w.get("date"):
                weather_by_date[w["date"]] = w

        weather_days: List[WeatherForecastDay] = []

        for i in range(days):
            target_date_str = (
                base_date + timedelta(days=i)
            ).strftime("%Y-%m-%d")

            w_item = weather_by_date.get(target_date_str)

            if w_item:
                weather_days.append(
                    WeatherForecastDay(
                        date=target_date_str,
                        condition=w_item.get(
                            "condition", "Forecast unavailable"
                        ),
                        temp_max_c=w_item.get("temp_max_c"),
                        temp_min_c=w_item.get("temp_min_c"),
                        precipitation_probability_pct=w_item.get(
                            "precipitation_probability_pct"
                        ),
                        travel_advice=w_item.get(
                            "travel_advice",
                            "Check local weather conditions.",
                        ),
                    )
                )
            else:
                weather_days.append(
                    WeatherForecastDay(
                        date=target_date_str,
                        condition="Forecast unavailable",
                        temp_max_c=None,
                        temp_min_c=None,
                        precipitation_probability_pct=None,
                        travel_advice=weather_res.get(
                            "data_notice",
                            weather_res.get(
                                "message",
                                "No forecast data returned for this date.",
                            ),
                        ),
                    )
                )

        # 6. Day-wise itinerary generation
        schedule: List[DayItinerary] = []
        total_entrance_fees_per_person = 0.0
        place_idx = 0
        num_places = len(raw_places)

        for d in range(1, days + 1):
            curr_date = base_date + timedelta(days=d - 1)
            curr_date_str = curr_date.strftime("%Y-%m-%d")

            w_summary = weather_days[d - 1].condition

            activities: List[ActivityItem] = []

            day_expense = (
                daily_food_per_person * travelers
            ) + daily_transport

            # CASE A: Single-day trip
            if days == 1:
                theme = (
                    f"Same-Day Express Tour in "
                    f"{config.normalize_city_name(destination_city)}"
                )

                activities.append(
                    ActivityItem(
                        time_slot="Arrival",
                        activity_name=(
                            f"Flight Arrival ({flight_model.airline} "
                            f"{flight_model.flight_number}) & "
                            "Transit into City"
                        ),
                        category="Logistics",
                        estimated_duration_hours=1.5,
                        entrance_fee_inr=0.0,
                        highlights_or_tips=(
                            "Allow time for arrival formalities "
                            "and airport transfers."
                        ),
                    )
                )

                if arrival_hour is not None and arrival_hour <= 8:
                    if place_idx < num_places:
                        p1 = raw_places[place_idx]
                        place_idx += 1

                        fee = _safe_float(
                            p1.get("entrance_fee_inr")
                        )

                        activities.append(
                            ActivityItem(
                                time_slot="Afternoon (Sightseeing)",
                                activity_name=p1["place_name"],
                                category=p1.get(
                                    "category", "Attraction"
                                ),
                                estimated_duration_hours=_safe_float(
                                    p1.get("visit_duration_hours", 2.5),
                                    2.5,
                                ),
                                entrance_fee_inr=fee,
                                highlights_or_tips=(
                                    f"Listed rating: "
                                    f"{_format_rating(p1.get('rating'))}. "
                                    "Check local visiting hours "
                                    "before departure."
                                ),
                            )
                        )

                        total_entrance_fees_per_person += fee

                activities.append(
                    ActivityItem(
                        time_slot="Departure",
                        activity_name=(
                            f"Return travel to "
                            f"{config.normalize_city_name(source_city)} "
                            "(booking required)"
                        ),
                        category="Departure",
                        estimated_duration_hours=0.0,
                        entrance_fee_inr=0.0,
                        highlights_or_tips=(
                            "A return flight has not been searched "
                            "or confirmed. Arrange transport and "
                            "book a separate return journey."
                        ),
                    )
                )

            # CASE B: First day of a multi-day trip
            elif d == 1:
                theme = (
                    f"Arrival & Discovery in "
                    f"{config.normalize_city_name(destination_city)}"
                )

                activities.append(
                    ActivityItem(
                        time_slot="Arrival",
                        activity_name=(
                            f"Flight Arrival ({flight_model.airline} "
                            f"{flight_model.flight_number}) & "
                            f"Check-in at {hotel_model.hotel_name}"
                        ),
                        category="Logistics",
                        estimated_duration_hours=2.0,
                        entrance_fee_inr=0.0,
                        highlights_or_tips=(
                            "Allow time for airport transfer and "
                            f"hotel check-in. Address: {hotel_model.address}"
                        ),
                    )
                )

                if arrival_hour is not None and arrival_hour <= 10:
                    if place_idx < num_places:
                        p1 = raw_places[place_idx]
                        place_idx += 1

                        fee = _safe_float(
                            p1.get("entrance_fee_inr")
                        )

                        activities.append(
                            ActivityItem(
                                time_slot="Afternoon",
                                activity_name=p1["place_name"],
                                category=p1.get(
                                    "category", "Attraction"
                                ),
                                estimated_duration_hours=_safe_float(
                                    p1.get("visit_duration_hours", 2.0),
                                    2.0,
                                ),
                                entrance_fee_inr=fee,
                                highlights_or_tips=(
                                    f"Listed rating: "
                                    f"{_format_rating(p1.get('rating'))}. "
                                    "Visit only if arrival, check-in "
                                    "and transport timings permit."
                                ),
                            )
                        )

                        total_entrance_fees_per_person += fee
                    else:
                        activities.append(
                            ActivityItem(
                                time_slot="Afternoon",
                                activity_name=(
                                    "Free time and nearby exploration"
                                ),
                                category="Leisure",
                                estimated_duration_hours=2.0,
                                entrance_fee_inr=0.0,
                                highlights_or_tips=(
                                    "Keep the first day flexible "
                                    "after travelling."
                                ),
                            )
                        )
                else:
                    activities.append(
                        ActivityItem(
                            time_slot="Evening",
                            activity_name=(
                                "Hotel check-in, dinner and rest"
                            ),
                            category="Dining & Leisure",
                            estimated_duration_hours=2.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips=(
                                "No sightseeing is scheduled because "
                                "the arrival time is late or unavailable."
                            ),
                        )
                    )

            # CASE C: Final return day
            elif d == days:
                theme = (
                    f"Hotel Check-out & Return to "
                    f"{config.normalize_city_name(source_city)}"
                )

                activities.append(
                    ActivityItem(
                        time_slot="Morning (Check-out)",
                        activity_name=(
                            f"Hotel Check-out at {hotel_model.hotel_name}"
                        ),
                        category="Logistics",
                        estimated_duration_hours=1.0,
                        entrance_fee_inr=0.0,
                        highlights_or_tips=(
                            "Confirm the property's checkout time "
                            "and settle outstanding charges."
                        ),
                    )
                )

                activities.append(
                    ActivityItem(
                        time_slot="After Check-out",
                        activity_name=(
                            f"Return travel to "
                            f"{config.normalize_city_name(source_city)} "
                            "(booking required)"
                        ),
                        category="Departure",
                        estimated_duration_hours=0.0,
                        entrance_fee_inr=0.0,
                        highlights_or_tips=(
                            "A return flight has not been searched "
                            "or confirmed. Arrange airport transfer "
                            "and book a separate return flight."
                        ),
                    )
                )

            # CASE D: Middle full days
            else:
                theme = (
                    f"Sightseeing & Flexible Leisure (Day {d})"
                )

                # Morning: One major attraction
                if place_idx < num_places:
                    p_morn = raw_places[place_idx]
                    place_idx += 1

                    fee = _safe_float(
                        p_morn.get("entrance_fee_inr")
                    )

                    activities.append(
                        ActivityItem(
                            time_slot="Morning (09:30 - 12:30)",
                            activity_name=p_morn["place_name"],
                            category=p_morn.get(
                                "category", "Sightseeing"
                            ),
                            estimated_duration_hours=_safe_float(
                                p_morn.get("visit_duration_hours", 2.5),
                                2.5,
                            ),
                            entrance_fee_inr=fee,
                            highlights_or_tips=(
                                f"Listed rating: "
                                f"{_format_rating(p_morn.get('rating'))}. "
                                f"DSLR permission: "
                                f"{p_morn.get('dslr_allowed', 'Not available')}. "
                                f"Weekly off: "
                                f"{p_morn.get('weekly_off', 'Not available')}. "
                                "Confirm visiting hours and transport "
                                "before departure."
                            ),
                        )
                    )

                    total_entrance_fees_per_person += fee
                else:
                    activities.append(
                        ActivityItem(
                            time_slot="Morning (09:30 - 12:30)",
                            activity_name=(
                                "Free time for local exploration"
                            ),
                            category="Leisure",
                            estimated_duration_hours=2.5,
                            entrance_fee_inr=0.0,
                            highlights_or_tips=(
                                "Explore nearby areas at your own pace."
                            ),
                        )
                    )

                # Afternoon: Flexible leisure
                activities.append(
                    ActivityItem(
                        time_slot="Afternoon (14:00 - 16:30)",
                        activity_name=(
                            "Flexible leisure and relaxation"
                        ),
                        category="Leisure",
                        estimated_duration_hours=2.5,
                        entrance_fee_inr=0.0,
                        highlights_or_tips=(
                            "Use this time for rest or optional "
                            "activities. Check distances and "
                            "transport availability before "
                            "visiting another attraction."
                        ),
                    )
                )

                # Evening: Dinner and free time
                activities.append(
                    ActivityItem(
                        time_slot="Evening (17:30 - 20:00)",
                        activity_name="Dinner and local leisure",
                        category="Dining & Leisure",
                        estimated_duration_hours=2.0,
                        entrance_fee_inr=0.0,
                        highlights_or_tips=(
                            "Enjoy local food and leisure activities "
                            "according to your location and preferences."
                        ),
                    )
                )

            # Calculate daily expenses
            day_entrance_sum = (
                sum(
                    _safe_float(a.entrance_fee_inr)
                    for a in activities
                ) * travelers
            )

            day_total_cost = day_expense + day_entrance_sum

            schedule.append(
                DayItinerary(
                    day_number=d,
                    date_label=f"Day {d} ({curr_date_str})",
                    theme=theme,
                    weather_summary=w_summary,
                    activities=activities,
                    estimated_daily_expense_inr=round(
                        day_total_cost, 2
                    ),
                )
            )

        # 7. Comprehensive budget
        budget_res = calculate_budget(
            flight_price_per_person=flight_model.price_inr,
            hotel_nightly_price=hotel_model.price_per_night_inr,
            trip_days=days,
            num_travelers=travelers,
            is_round_trip=False,
            daily_food_per_person=daily_food_per_person,
            daily_transport=daily_transport,
            entrance_fees_per_person=total_entrance_fees_per_person,
            other_buffer_per_person=other_buffer_per_person,
        )

        bd = budget_res["breakdown"]

        budget_model = BudgetBreakdown(
            grand_total_inr=budget_res["summary"]["grand_total"],
            per_person_inr=budget_res["summary"]["per_person_cost"],
            flights_total_inr=bd["flights"]["amount"],
            hotel_total_inr=bd["accommodation"]["amount"],
            food_total_inr=bd["food_and_dining"]["amount"],
            local_transport_total_inr=bd["local_transport"]["amount"],
            sightseeing_total_inr=bd["sightseeing_entrance"]["amount"],
            other_buffer_inr=bd["other_buffer"]["amount"],
        )

        # 8. Selection justifications
        if flight_model.price_inr > 0:
            flight_just = (
                f"Selected {flight_model.airline} "
                f"({flight_model.flight_number}) using the available "
                f"historical reference fare of "
                f"₹{int(flight_model.price_inr):,} with "
                f"{flight_model.stops} stops. "
                "This is not a confirmed live booking price."
            )
        else:
            flight_just = (
                f"Flight route between {source_city} and "
                f"{destination_city} requires external booking "
                "because no matching flight record was found."
            )

        if hotel_model.price_per_night_inr > 0:
            hotel_just = (
                f"Recommended {hotel_model.hotel_name} from the "
                "project hotel catalogue with a reference budget "
                f"rate of ₹{int(hotel_model.price_per_night_inr):,}/night. "
            )

            if hotel_model.star_rating is not None:
                hotel_just += (
                    f"Property star classification: "
                    f"{hotel_model.star_rating}. "
                )

            if hotel_model.review_rating is not None:
                hotel_just += (
                    f"Guest review rating: "
                    f"{hotel_model.review_rating}. "
                )

            hotel_just += (
                "Confirm current rates and availability "
                "with the property before booking."
            )
        else:
            hotel_just = (
                f"Accommodation in {destination_city} requires "
                "external booking because no priced catalogue "
                "property was found."
            )

        selected_count = min(num_places, place_idx)

        justifications = SelectionJustifications(
            flight_justification=flight_just,
            hotel_justification=hotel_just,
            attractions_justification=(
                f"Selected {selected_count} attractions from "
                "the available destination dataset. The itinerary "
                "schedules one major attraction per full sightseeing "
                "day and keeps remaining time flexible. Geographic "
                "route optimization is not supported because "
                "verified coordinates and travel-time data "
                "are unavailable."
            ),
            pacing_justification=(
                "Late or unavailable arrival times do not receive "
                "assumed sightseeing schedules. Full days contain "
                "one major attraction with flexible afternoon "
                "and evening periods. The final day is reserved "
                "for checkout and onward travel."
            ),
        )

        # 9. Data limitations and budget status
        data_limitations = (
            "Outbound flight fare and schedule are historical/"
            "reference dataset values, not confirmed live bookings. "
            "No return flight was searched or included in the budget. "
            "Hotel rates are catalogue reference values, not verified "
            "live prices or availability. Weather dates without "
            "available API results are marked unavailable or "
            "identified according to the weather tool's data notice. "
            "Attraction sequencing uses available dataset metadata; "
            "geographic route optimization is not supported because "
            "verified place coordinates and travel-time data are "
            "unavailable. The requested budget is not enforced "
            "during flight or hotel selection."
        )

        if total_budget is not None:
            budget_status = (
                "The reference estimate is within the requested budget."
                if budget_model.grand_total_inr <= total_budget
                else "The reference estimate exceeds the requested budget."
            )

            data_limitations += (
                f" Requested budget: ₹{total_budget:,.0f}. "
                f"Calculated reference estimate: "
                f"₹{budget_model.grand_total_inr:,.0f}. "
                f"{budget_status} "
                "The budget is not enforced during flight or "
                "hotel selection."
            )

        # 10. Final structured response
        title = (
            f"{days}-Day Personalized Tour: "
            f"{config.normalize_city_name(source_city)} to "
            f"{config.normalize_city_name(destination_city)}"
        )

        return CompleteTravelItinerary(
            trip_title=title,
            source_city=config.normalize_city_name(source_city),
            destination_city=config.normalize_city_name(destination_city),
            trip_duration_days=days,
            num_travelers=travelers,
            flight=flight_model,
            hotel=hotel_model,
            weather=weather_days,
            day_wise_schedule=schedule,
            budget=budget_model,
            justifications=justifications,
            data_limitations=data_limitations,
        )

    def save_itinerary_json(
        self,
        itinerary: CompleteTravelItinerary,
        filename: Optional[str] = None,
    ) -> Path:
        """Save the validated itinerary as formatted JSON."""

        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            filename = (
                f"itinerary_{itinerary.destination_city.lower()}_"
                f"{timestamp}.json"
            )

        out_path = config.OUTPUTS_DIR / filename
        out_path.parent.mkdir(parents=True, exist_ok=True)

        out_path.write_text(
            itinerary.model_dump_json(indent=2),
            encoding="utf-8",
        )

        logger.info("Itinerary saved to %s", out_path)

        return out_path


def get_itinerary_generator() -> ItineraryGenerator:
    """Return an itinerary generator instance."""

    return ItineraryGenerator()
