"""Prompts and Pydantic validation schemas for the Agentic Travel Planning Assistant."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


# -----------------------------------------------------------------------------
# PYDANTIC STRUCTURED SCHEMAS
# -----------------------------------------------------------------------------

class FlightRecommendation(BaseModel):
    airline: str = Field(description="Airline name")
    flight_number: str = Field(description="Flight code or reference ID")
    source: str = Field(description="Departure city")
    destination: str = Field(description="Arrival city")
    departure_time: str = Field(description="Scheduled departure time")
    arrival_time: str = Field(description="Scheduled arrival time")
    duration_hours: float = Field(description="Flight duration in hours")
    stops: str = Field(description="Number of stops (e.g. zero, one)")
    travel_class: str = Field(default="Economy", description="Travel cabin class")
    price_inr: float = Field(description="Historical/reference price per passenger in INR")
    selection_reason: str = Field(description="Explanation of why this flight was chosen")
    is_historical: bool = Field(default=True, description="True since flight is historical/reference")


class HotelRecommendation(BaseModel):
    hotel_name: str = Field(description="Property name")
    city: str = Field(description="Destination city")
    address: str = Field(description="Location or address")
    star_rating: Optional[float] = Field(default=None, description="Official star rating if available")
    review_rating: Optional[float] = Field(default=None, description="Review score if available")
    price_per_night_inr: float = Field(description="Nightly rate in INR")
    amenities: List[str] = Field(default_factory=list, description="Hotel amenities provided")
    recommendation_reason: str = Field(description="Explanation of why this hotel was chosen")
    is_verified_price: bool = Field(default=False, description="Whether price is verified from baseline dataset")


class ActivityItem(BaseModel):
    time_slot: str = Field(description="Morning, Afternoon, or Evening")
    activity_name: str = Field(description="Name of place or activity")
    category: str = Field(description="Category (e.g. Fort, Beach, Temple, Dining)")
    estimated_duration_hours: float = Field(description="Estimated hours spent")
    entrance_fee_inr: float = Field(default=0.0, description="Ticket entrance cost in INR")
    highlights_or_tips: str = Field(description="Tips, highlights, or photography rules")


class WeatherForecastDay(BaseModel):
    date: str = Field(description="Forecast date YYYY-MM-DD")
    condition: str = Field(description="Weather condition (e.g. Clear Sky, Partly Cloudy)")
    temp_max_c: Optional[float] = Field(default=None, description="Max temperature in Celsius")
    temp_min_c: Optional[float] = Field(default=None, description="Min temperature in Celsius")
    precipitation_probability_pct: Optional[int] = Field(default=0, description="Chance of rain (%)")
    travel_advice: str = Field(default="Pleasant conditions", description="Clothing or travel advice")


class DayItinerary(BaseModel):
    day_number: int = Field(description="Day 1, 2, 3, etc.")
    date_label: str = Field(description="Formatted date or Day indicator")
    theme: str = Field(description="Title or theme of the day (e.g. Heritage & Forts)")
    weather_summary: str = Field(description="Weather expected for this day")
    activities: List[ActivityItem] = Field(description="Scheduled activities for morning, afternoon, evening")
    estimated_daily_expense_inr: float = Field(description="Estimated day expenditure in INR")


class BudgetBreakdown(BaseModel):
    grand_total_inr: float = Field(description="Total estimated cost for all travelers in INR")
    per_person_inr: float = Field(description="Cost per individual traveler in INR")
    flights_total_inr: float = Field(description="Round-trip or one-way flights total")
    hotel_total_inr: float = Field(description="Total hotel accommodation cost")
    food_total_inr: float = Field(description="Total food & dining allowance")
    local_transport_total_inr: float = Field(description="Total local cab/transit allowance")
    sightseeing_total_inr: float = Field(description="Total monument tickets")
    other_buffer_inr: float = Field(description="Emergency contingency buffer")


class SelectionJustifications(BaseModel):
    flight_justification: str = Field(description="Why this flight option was selected")
    hotel_justification: str = Field(description="Why this hotel was recommended")
    attractions_justification: str = Field(description="How attractions were curated to balance time and variety")
    pacing_justification: str = Field(description="Why the itinerary is realistically paced")


class CompleteTravelItinerary(BaseModel):
    trip_title: str = Field(description="Catchy trip title")
    source_city: str = Field(description="Origin city")
    destination_city: str = Field(description="Destination city")
    trip_duration_days: int = Field(description="Duration in days")
    num_travelers: int = Field(default=1, description="Number of travelers")
    flight: FlightRecommendation = Field(description="Selected flight details")
    hotel: HotelRecommendation = Field(description="Selected hotel details")
    weather: List[WeatherForecastDay] = Field(default_factory=list, description="Weather forecast for the trip")
    day_wise_schedule: List[DayItinerary] = Field(description="Day-by-day plan")
    budget: BudgetBreakdown = Field(description="Itemized budget breakdown")
    justifications: SelectionJustifications = Field(description="Explanations for all recommendations")
    data_limitations: str = Field(
        default="Flights and hotel rates are based on historical reference datasets. Weather is retrieved live from Open-Meteo API.",
        description="Dataset limitations and disclaimer"
    )


# -----------------------------------------------------------------------------
# AGENT SYSTEM PROMPT
# -----------------------------------------------------------------------------

TRAVEL_AGENT_SYSTEM_PROMPT = """You are an expert Autonomous AI Travel Planning Assistant and Travel Concierge.
Your mission is to assist travelers by creating personalized, realistic, and budget-conscious travel itineraries.

You have access to five specialized tools:
1. `flight_search_tool`: Searches historical domestic flight records between Indian cities (routes, fares, duration, stops).
2. `hotel_recommendation_tool`: Discovers hotels, verified baseline prices, ratings, and amenities in the destination.
3. `places_discovery_tool`: Finds curated tourist attractions, temples, forts, beaches, and museums with entry fees.
4. `weather_lookup_tool`: Retrieves real-time multi-day forecasts, temperatures, and rain chances from Open-Meteo API.
5. `budget_estimation_tool`: Calculates the complete itemized trip budget following the exact formula:
   Total Budget = Flight Cost + Hotel Cost + Food Estimate + Local Transport + Entrance Fees + Other Buffer.

### CRITICAL OPERATING GUIDELINES:
1. **Autonomous Tool Usage**: Always invoke your tools to gather real factual data before assembling an itinerary. Never guess flight prices, hotel ratings, or weather forecasts.
2. **Honesty & Transparency**:
   - Clearly explain that flight fares and hotel catalogues are historical/reference datasets and not live commercial inventory.
   - Live weather forecasts are fetched directly from Open-Meteo.
   - If a hotel price is estimated from city median baseline, clearly state this.
3. **Realistic Pacing**:
   - Limit daytime activities to 2 to 3 major attractions per day so travelers are not rushed.
   - Account for travel fatigue on Day 1 (after flight arrival) and departure on the final day.
4. **Structured Reasoning**:
   - Explain WHY each specific flight, hotel, and attraction was selected.
   - Present a clean, well-formatted, professional response with markdown sections, day-wise schedule, and itemized budget.
"""
