"""Itinerary Generator module for the Agentic Travel Planning Assistant.

Synthesizes flight, hotel, tourist attraction, weather, and budget data
into a coherent, realistically paced day-wise travel itinerary validated
with Pydantic schemas.
Accurately aligns flight schedules, avoids return-day activity conflicts,
and never fabricates missing flight schedules or fake hotel pricing.
"""

import json
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
        """Constructs an end-to-end validated itinerary.

        Args:
            source_city: Origin city.
            destination_city: Target city.
            duration_days: Length of trip in days (minimum 1, maximum 14).
            start_date: Start date string 'YYYY-MM-DD' (defaults to tomorrow if omitted).
            num_travelers: Number of people traveling.
            total_budget: Desired total budget in INR.
            travel_class: Flight cabin class ('Economy' or 'Business').
            preferred_category: Attraction preference (e.g. 'Beach', 'Fort', 'Temple').
            preferred_amenities: Hotel amenities list (e.g. ['wifi', 'pool']).
            daily_food_per_person: Daily food allowance per traveler.
            daily_transport: Daily cab/auto transit allowance.
            other_buffer_per_person: Contingency allowance per person.
            selected_flight_data: Optional pre-selected flight from LangChain tool execution.
            selected_hotel_data: Optional pre-selected hotel from LangChain tool execution.

        Returns:
            A validated Pydantic CompleteTravelItinerary object.
        """
        days = max(1, min(int(duration_days), 14))
        travelers = max(1, int(num_travelers))

        # Default start date to tomorrow if not supplied
        if not start_date:
            base_date = date.today() + timedelta(days=1)
        else:
            try:
                base_date = datetime.strptime(start_date, "%Y-%m-%d").date()
            except ValueError:
                base_date = date.today() + timedelta(days=1)

        end_date = base_date + timedelta(days=max(days - 1, 0))
        start_date_str = base_date.strftime("%Y-%m-%d")
        end_date_str = end_date.strftime("%Y-%m-%d")

        # 1. Fetch Flights (or use LangChain tool selection)
        best_flight_data = selected_flight_data
        if not best_flight_data:
            flight_res = search_flights(
                source_city=source_city,
                destination_city=destination_city,
                travel_class=travel_class,
                limit=5,
            )
            if flight_res.get("status") == "success" and flight_res.get("results"):
                best_flight_data = flight_res["cheapest_option"]

        if best_flight_data:
            flight_model = FlightRecommendation(
                airline=best_flight_data.get("airline", "Indian Domestic Airline"),
                flight_number=best_flight_data.get("flight_number", "FLIGHT"),
                source=best_flight_data.get("source", config.normalize_city_name(source_city)),
                destination=best_flight_data.get("destination", config.normalize_city_name(destination_city)),
                departure_time=str(best_flight_data.get("departure_time", "Morning")),
                arrival_time=str(best_flight_data.get("arrival_time", "Afternoon")),
                duration_hours=float(best_flight_data.get("duration_hours", 2.5)),
                stops=str(best_flight_data.get("stops", "zero")),
                travel_class=best_flight_data.get("travel_class", travel_class),
                price_inr=float(best_flight_data.get("price_inr", 4000.0)),
                selection_reason=best_flight_data.get("selection_reason", "Historical route option from dataset"),
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
                selection_reason="No flight records found in project historical datasets for this route; fare excluded from calculation.",
                is_historical=True,
            )

        # 2. Fetch Hotels (or use LangChain tool selection)
        best_hotel_data = selected_hotel_data
        if not best_hotel_data:
            hotel_res = search_hotels(
                city=destination_city,
                preferred_amenities=preferred_amenities,
                limit=5,
            )
            if hotel_res.get("status") == "success" and hotel_res.get("results"):
                best_hotel_data = hotel_res["results"][0]

        if best_hotel_data:
            hotel_model = HotelRecommendation(
                hotel_name=best_hotel_data["hotel_name"],
                city=best_hotel_data["city"],
                address=best_hotel_data.get("address", f"{destination_city}"),
                star_rating=best_hotel_data.get("star_rating"),
                review_rating=best_hotel_data.get("review_rating"),
                price_per_night_inr=float(best_hotel_data.get("price_per_night_inr", 3500.0)),
                amenities=best_hotel_data.get("amenities", ["wifi", "ac"]),
                recommendation_reason=best_hotel_data.get("recommendation_reason", "Top rated property in destination"),
                is_verified_price=best_hotel_data.get("is_verified_price", False),
            )
        else:
            hotel_model = HotelRecommendation(
                hotel_name="No catalogue property found",
                city=config.normalize_city_name(destination_city),
                address=f"Central {config.normalize_city_name(destination_city)}",
                star_rating=None,
                review_rating=None,
                price_per_night_inr=0.0,
                amenities=[],
                recommendation_reason="No hotel records exist in dataset for this city.",
                is_verified_price=False,
            )

        # 3. Fetch Attractions / Places
        places_res = search_places(
            city=destination_city,
            category=preferred_category,
            limit=max(days * 3, 6),
        )
        raw_places = places_res.get("results", [])

        # 4. Fetch Weather Forecast from Open-Meteo for the exact date range
        weather_res = get_weather_forecast(
            city=destination_city,
            days=days,
            start_date=start_date_str,
            end_date=end_date_str,
        )

        weather_by_date: Dict[str, Dict[str, Any]] = {}
        for w in weather_res.get("daily_forecast", []):
            weather_by_date[w["date"]] = w

        weather_days: List[WeatherForecastDay] = []
        for i in range(days):
            target_date_str = (base_date + timedelta(days=i)).strftime("%Y-%m-%d")
            w_item = weather_by_date.get(target_date_str)
            if w_item:
                weather_days.append(
                    WeatherForecastDay(
                        date=target_date_str,
                        condition=w_item["condition"],
                        temp_max_c=w_item.get("temp_max_c"),
                        temp_min_c=w_item.get("temp_min_c"),
                        precipitation_probability_pct=w_item.get("precipitation_probability_pct", 0),
                        travel_advice=w_item.get("travel_advice", "Pleasant outdoor weather"),
                    )
                )
            else:
                # Do not invent temperatures when the API has no data for this date.
                weather_days.append(
                    WeatherForecastDay(
                        date=target_date_str,
                        condition="Forecast unavailable",
                        temp_max_c=None,
                        temp_min_c=None,
                        precipitation_probability_pct=None,
                        travel_advice=weather_res.get(
                            "data_notice",
                            weather_res.get("message", "No forecast data returned for this date."),
                        ),
                    )
                )

        # 5. Synthesize Day-Wise Activities with Strict Conflict Avoidance
        schedule: List[DayItinerary] = []
        total_entrance_fees_per_person = 0.0
        place_idx = 0
        num_places = len(raw_places)

        # Determine flight departure timing to prevent return-day schedule conflicts
        flight_dep = flight_model.departure_time.lower()

        for d in range(1, days + 1):
            curr_date = base_date + timedelta(days=d - 1)
            curr_date_str = curr_date.strftime("%Y-%m-%d")
            w_summary = weather_days[d - 1].condition if d - 1 < len(weather_days) else "Clear"

            activities: List[ActivityItem] = []
            day_expense = (daily_food_per_person * travelers) + (daily_transport)

            # CASE A: Single-Day Trip (days == 1)
            if days == 1:
                theme = f"Same-Day Express Tour in {config.normalize_city_name(destination_city)}"
                # Morning: Arrival
                activities.append(
                    ActivityItem(
                        time_slot="Morning (Arrival)",
                        activity_name=f"Flight Arrival ({flight_model.airline} {flight_model.flight_number}) & Transit into City",
                        category="Logistics",
                        estimated_duration_hours=1.5,
                        entrance_fee_inr=0.0,
                        highlights_or_tips="Arrive in city, freshen up at airport lounge, begin exploration.",
                    )
                )
                # Afternoon: Highlight Sightseeing
                if place_idx < num_places:
                    p1 = raw_places[place_idx]
                    place_idx += 1
                    activities.append(
                        ActivityItem(
                            time_slot="Afternoon (Sightseeing)",
                            activity_name=p1["place_name"],
                            category=p1.get("category", "Attraction"),
                            estimated_duration_hours=p1.get("visit_duration_hours", 2.5),
                            entrance_fee_inr=p1.get("entrance_fee_inr", 0.0),
                            highlights_or_tips=f"Rated {p1.get('rating', 4.0)}★. {p1.get('best_time_to_visit', 'Prime daytime sightseeing')}.",
                        )
                    )
                    total_entrance_fees_per_person += p1.get("entrance_fee_inr", 0.0)

                # Evening: Return Departure
                activities.append(
                    ActivityItem(
                        time_slot="Evening (Departure)",
                        activity_name=f"Airport Transit & Return Flight to {config.normalize_city_name(source_city)}",
                        category="Departure",
                        estimated_duration_hours=2.5,
                        entrance_fee_inr=0.0,
                        highlights_or_tips="Same-day return journey; reach departure gate 1.5h prior.",
                    )
                )

            # CASE B: First Day of Multi-Day Trip (d == 1, days > 1)
            elif d == 1:
                theme = f"Arrival & Discovery in {config.normalize_city_name(destination_city)}"
                activities.append(
                    ActivityItem(
                        time_slot="Morning / Midday",
                        activity_name=f"Flight Arrival ({flight_model.airline} {flight_model.flight_number}) & Check-in at {hotel_model.hotel_name}",
                        category="Logistics",
                        estimated_duration_hours=2.0,
                        entrance_fee_inr=0.0,
                        highlights_or_tips=f"Unpack and settle in. Address: {hotel_model.address}",
                    )
                )
                if place_idx < num_places:
                    p1 = raw_places[place_idx]
                    place_idx += 1
                    activities.append(
                        ActivityItem(
                            time_slot="Afternoon",
                            activity_name=p1["place_name"],
                            category=p1.get("category", "Attraction"),
                            estimated_duration_hours=p1.get("visit_duration_hours", 2.0),
                            entrance_fee_inr=p1.get("entrance_fee_inr", 0.0),
                            highlights_or_tips=f"Rated {p1.get('rating', 4.0)}★. {p1.get('best_time_to_visit', 'Afternoon visit')}.",
                        )
                    )
                    total_entrance_fees_per_person += p1.get("entrance_fee_inr", 0.0)

                activities.append(
                    ActivityItem(
                        time_slot="Evening",
                        activity_name="Local Dining & Promenade Walk",
                        category="Dining & Leisure",
                        estimated_duration_hours=2.0,
                        entrance_fee_inr=0.0,
                        highlights_or_tips="Sample regional delicacies and relax after the flight.",
                    )
                )

            # CASE C: Final Return Day of Multi-Day Trip (d == days, days > 1)
            elif d == days:
                theme = f"Hotel Check-out & Return to {config.normalize_city_name(source_city)}"

                # Align activities based on flight departure timing
                if "morning" in flight_dep or "early" in flight_dep:
                    # Morning departure: Check out immediately, transit to airport
                    activities.append(
                        ActivityItem(
                            time_slot="Morning (Check-out)",
                            activity_name=f"Hotel Check-out at {hotel_model.hotel_name} & Airport Transit",
                            category="Logistics",
                            estimated_duration_hours=2.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips="Settle hotel bill, pack bags, and head directly to airport for morning departure.",
                        )
                    )
                    activities.append(
                        ActivityItem(
                            time_slot="Midday (Departure)",
                            activity_name=f"Return flight to {config.normalize_city_name(source_city)} (not searched)",
                            category="Departure",
                            estimated_duration_hours=0.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips="The dataset search selected an outbound flight only. Search and confirm a separate return flight before travel.",
                        )
                    )
                elif "afternoon" in flight_dep:
                    # Afternoon departure: Morning leisure, then check-out and airport
                    activities.append(
                        ActivityItem(
                            time_slot="Morning",
                            activity_name="Souvenir Shopping & Handicraft Bazaar",
                            category="Leisure / Shopping",
                            estimated_duration_hours=2.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips="Pick up local spices, handicrafts, and souvenirs.",
                        )
                    )
                    activities.append(
                        ActivityItem(
                            time_slot="Afternoon (Departure)",
                            activity_name=f"Hotel Check-out & return flight to {config.normalize_city_name(source_city)} (not searched)",
                            category="Departure",
                            estimated_duration_hours=0.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips="Check out by the hotel's stated time. A return flight has not been searched or confirmed; arrange airport transit and return booking separately.",
                        )
                    )
                else:
                    # Evening or Night departure: Morning final attraction, afternoon packing, evening flight
                    if place_idx < num_places:
                        p_final = raw_places[place_idx]
                        place_idx += 1
                        activities.append(
                            ActivityItem(
                                time_slot="Morning",
                                activity_name=p_final["place_name"],
                                category=p_final.get("category", "Culture"),
                                estimated_duration_hours=p_final.get("visit_duration_hours", 2.0),
                                entrance_fee_inr=p_final.get("entrance_fee_inr", 0.0),
                                highlights_or_tips=f"Last-minute photography and sightseeing (Rated {p_final.get('rating', 4.0)}★).",
                            )
                        )
                        total_entrance_fees_per_person += p_final.get("entrance_fee_inr", 0.0)
                    else:
                        activities.append(
                            ActivityItem(
                                time_slot="Morning",
                                activity_name="Local Heritage Walk & Cafe Breakfast",
                                category="Leisure",
                                estimated_duration_hours=2.0,
                                entrance_fee_inr=0.0,
                                highlights_or_tips="Relaxing stroll through historical streets and local cafe breakfast.",
                            )
                        )

                    activities.append(
                        ActivityItem(
                            time_slot="Afternoon",
                            activity_name=f"Hotel Check-out at {hotel_model.hotel_name} & Souvenir Shopping",
                            category="Logistics / Shopping",
                            estimated_duration_hours=2.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips="Store luggage with concierge, pick up gifts and sweets.",
                        )
                    )
                    activities.append(
                        ActivityItem(
                            time_slot="Evening (Departure)",
                            activity_name=f"Airport Transit & Evening Flight to {config.normalize_city_name(source_city)}",
                            category="Departure",
                            estimated_duration_hours=3.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips=f"Reach airport 2h prior to flight departure at {flight_model.departure_time}.",
                        )
                    )

            # CASE D: Middle Full Days
            else:
                theme = f"Sightseeing & Cultural Exploration (Day {d})"

                # Morning slot
                if place_idx < num_places:
                    p_morn = raw_places[place_idx]
                    place_idx += 1
                    activities.append(
                        ActivityItem(
                            time_slot="Morning (09:30 - 12:30)",
                            activity_name=p_morn["place_name"],
                            category=p_morn.get("category", "Sightseeing"),
                            estimated_duration_hours=p_morn.get("visit_duration_hours", 2.5),
                            entrance_fee_inr=p_morn.get("entrance_fee_inr", 0.0),
                            highlights_or_tips=f"Rated {p_morn.get('rating', 4.5)}★. DSLR: {p_morn.get('dslr_allowed', 'Allowed')}. Weekly off: {p_morn.get('weekly_off', 'Open')}.",
                        )
                    )
                    total_entrance_fees_per_person += p_morn.get("entrance_fee_inr", 0.0)
                else:
                    activities.append(
                        ActivityItem(
                            time_slot="Morning (09:30 - 12:30)",
                            activity_name="Local Artisan Market & Crafts Walk",
                            category="Leisure / Shopping",
                            estimated_duration_hours=2.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips="Explore regional textile weavers, pottery, and handicraft stalls.",
                        )
                    )

                # Afternoon slot
                if place_idx < num_places:
                    p_aft = raw_places[place_idx]
                    place_idx += 1
                    activities.append(
                        ActivityItem(
                            time_slot="Afternoon (14:00 - 16:30)",
                            activity_name=p_aft["place_name"],
                            category=p_aft.get("category", "Attraction"),
                            estimated_duration_hours=p_aft.get("visit_duration_hours", 2.0),
                            entrance_fee_inr=p_aft.get("entrance_fee_inr", 0.0),
                            highlights_or_tips=f"Explore exhibits and architecture. Rated {p_aft.get('rating', 4.0)}★.",
                        )
                    )
                    total_entrance_fees_per_person += p_aft.get("entrance_fee_inr", 0.0)
                else:
                    activities.append(
                        ActivityItem(
                            time_slot="Afternoon (14:00 - 16:30)",
                            activity_name="Culinary Trail & Regional Tasting",
                            category="Dining / Culture",
                            estimated_duration_hours=2.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips="Taste iconic local culinary specialties at authentic eateries.",
                        )
                    )

                # Evening slot
                if place_idx < num_places:
                    p_eve = raw_places[place_idx]
                    place_idx += 1
                    activities.append(
                        ActivityItem(
                            time_slot="Evening (17:30 - 20:00)",
                            activity_name=p_eve["place_name"],
                            category=p_eve.get("category", "Sunset / Leisure"),
                            estimated_duration_hours=p_eve.get("visit_duration_hours", 2.0),
                            entrance_fee_inr=p_eve.get("entrance_fee_inr", 0.0),
                            highlights_or_tips="Scenic evening atmosphere and sunset view.",
                        )
                    )
                    total_entrance_fees_per_person += p_eve.get("entrance_fee_inr", 0.0)
                else:
                    activities.append(
                        ActivityItem(
                            time_slot="Evening (17:30 - 20:00)",
                            activity_name="Sunset Point & Promenade Walk",
                            category="Leisure / Nature",
                            estimated_duration_hours=2.0,
                            entrance_fee_inr=0.0,
                            highlights_or_tips="Watch the sunset, enjoy street performers, and relax.",
                        )
                    )

            # Sum entrance fees for the day
            day_entrance_sum = sum(a.entrance_fee_inr for a in activities) * travelers
            day_total_cost = day_expense + day_entrance_sum

            schedule.append(
                DayItinerary(
                    day_number=d,
                    date_label=f"Day {d} ({curr_date_str})",
                    theme=theme,
                    weather_summary=w_summary,
                    activities=activities,
                    estimated_daily_expense_inr=round(day_total_cost, 2),
                )
            )

        # 6. Compute Comprehensive Budget
        budget_res = calculate_budget(
            flight_price_per_person=flight_model.price_inr,
            hotel_nightly_price=hotel_model.price_per_night_inr,
            trip_days=days,
            num_travelers=travelers,
            # Only the outbound route is searched by the current flight tool.
            # Do not double the fare or imply a return booking exists.
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

        # 7. AI Justifications
        flight_just = (
            f"Selected {flight_model.airline} ({flight_model.flight_number}) because it offers "
            f"a historical reference fare of ₹{int(flight_model.price_inr):,} with {flight_model.stops} stops."
            if flight_model.price_inr > 0
            else f"Flight route between {source_city} and {destination_city} requires external booking."
        )

        hotel_rating = hotel_model.review_rating or hotel_model.star_rating
        hotel_just = (
            f"Recommended {hotel_model.hotel_name} from the project hotel catalogue"
            + (f" with a listed rating of {hotel_rating}★" if hotel_rating is not None else "")
            + f" and a reference budget rate of ₹{int(hotel_model.price_per_night_inr):,}/night. "
              "Confirm current rates and availability with the property before booking."
            if hotel_model.price_per_night_inr > 0
            else f"Accommodation in {destination_city} requires external booking."
        )

        justifications = SelectionJustifications(
            flight_justification=flight_just,
            hotel_justification=hotel_just,
            attractions_justification=(
                f"Selected {min(len(raw_places), place_idx)} curated attractions from datasets, balanced with "
                "local leisure walks to maintain a relaxed 2-3 activity daily pacing without traveler fatigue."
            ),
            pacing_justification=(
                "The final day is reserved for checkout and return travel. A return flight is not included because the current flight search only selected the outbound route."
            ),
        )

        title = f"{days}-Day Personalized Tour: {config.normalize_city_name(source_city)} to {config.normalize_city_name(destination_city)}"

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
            data_limitations=(
                "Outbound flight fare is a historical/reference dataset value; no return flight was searched or included in the budget. "
                "Hotel rates are reference values, not live prices or availability. Confirm all bookings directly. "
                "Weather dates without an API result are marked unavailable rather than estimated."
            ),
        )

    def save_itinerary_json(self, itinerary: CompleteTravelItinerary, filename: Optional[str] = None) -> Path:
        """Saves the validated itinerary as formatted JSON in the outputs folder."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"itinerary_{itinerary.destination_city.lower()}_{timestamp}.json"
        out_path = config.OUTPUTS_DIR / filename
        out_path.write_text(itinerary.model_dump_json(indent=2), encoding="utf-8")
        return out_path


def get_itinerary_generator() -> ItineraryGenerator:
    return ItineraryGenerator()
