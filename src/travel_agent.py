"""LangChain Agent module for the Agentic Travel Planning Assistant.

Orchestrates tool-calling reasoning across flights, hotels, attractions,
weather, and budget tools using LangChain and ChatOpenAI.
Extracts real tool outputs and directly feeds them into the structured
itinerary generator.
Provides an honest, fully functional demo mode when an API key is unavailable.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

import config
from src.flight_tool import flight_search_tool
from src.hotel_tool import hotel_recommendation_tool
from src.places_tool import places_discovery_tool
from src.weather_tool import weather_lookup_tool
from src.budget_tool import budget_estimation_tool
from src.prompts import TRAVEL_AGENT_SYSTEM_PROMPT, CompleteTravelItinerary
from src.itinerary_generator import get_itinerary_generator

logger = logging.getLogger(__name__)

# Register all tools
TRAVEL_TOOLS = [
    flight_search_tool,
    hotel_recommendation_tool,
    places_discovery_tool,
    weather_lookup_tool,
    budget_estimation_tool,
]


class TravelAgentRunner:
    """Manages LangChain agent execution and deterministic fallback planning."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        force_demo_mode: bool = False,
    ):
        self.api_key = api_key or config.OPENAI_API_KEY
        self.model_name = model_name or config.OPENAI_MODEL
        self.force_demo_mode = force_demo_mode
        self.generator = get_itinerary_generator()

    @property
    def has_llm_key(self) -> bool:
        """Checks if a valid-format OpenAI API key is configured."""
        if self.force_demo_mode:
            return False
        k = (self.api_key or "").strip()
        return bool(k and k != "your_openai_api_key_here" and len(k) > 10)

    def plan_trip(
        self,
        user_query: str,
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
        force_demo_mode: bool = False,
    ) -> Dict[str, Any]:
        """Plans a travel itinerary using LangChain autonomous agent or deterministic engine.

        Returns:
            Dict containing:
              - 'is_llm_powered': bool
              - 'mode': 'Autonomous LangChain Agent' or 'Deterministic Planner (Demo Mode)'
              - 'tool_trace': list of tool invocations performed
              - 'llm_response': str (if LLM was used)
              - 'structured_itinerary': CompleteTravelItinerary
        """
        tool_trace = []

        # If LLM key is missing or user requested demo mode:
        if not self.has_llm_key or force_demo_mode:
            logger.info("Executing via Deterministic Travel Planner (No API key / Demo Mode)")
            itinerary = self.generator.build_itinerary(
                source_city=source_city,
                destination_city=destination_city,
                duration_days=duration_days,
                start_date=start_date,
                num_travelers=num_travelers,
                total_budget=total_budget,
                travel_class=travel_class,
                preferred_category=preferred_category,
                preferred_amenities=preferred_amenities,
                daily_food_per_person=daily_food_per_person,
                daily_transport=daily_transport,
                other_buffer_per_person=other_buffer_per_person,
            )

            tool_trace = [
                {"tool": "flight_search_tool", "status": "executed", "result": f"Found {itinerary.flight.airline} flight for ₹{itinerary.flight.price_inr:,.0f}"},
                {"tool": "hotel_recommendation_tool", "status": "executed", "result": f"Selected {itinerary.hotel.hotel_name} (₹{itinerary.hotel.price_per_night_inr:,.0f}/night)"},
                {"tool": "places_discovery_tool", "status": "executed", "result": f"Scheduled attractions across {itinerary.trip_duration_days} days"},
                {"tool": "weather_lookup_tool", "status": "executed", "result": f"Retrieved live forecast for {itinerary.destination_city} via Open-Meteo"},
                {"tool": "budget_estimation_tool", "status": "executed", "result": f"Calculated total budget of ₹{itinerary.budget.grand_total_inr:,.0f}"},
            ]

            return {
                "is_llm_powered": False,
                "mode": "Deterministic Planner (Demo Mode)",
                "tool_trace": tool_trace,
                "llm_response": (
                    "**Demo Mode Active**: No OpenAI API key was detected. "
                    "The system executed real data filtering on the 300,000+ flight records, 4,000 hotel listings, "
                    "323 attraction spots, live Open-Meteo weather forecast API, and deterministic budget formulas. "
                    "To enable autonomous LLM reasoning, provide an `OPENAI_API_KEY`."
                ),
                "structured_itinerary": itinerary,
            }

        # Otherwise, run the LangChain Tool-Calling Agent
        try:
            logger.info(f"Initializing LangChain agent with model {self.model_name}")
            llm = ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=0.3,
            )

            # Compile LangChain agent with tools
            agent_graph = create_agent(
                model=llm,
                tools=TRAVEL_TOOLS,
                system_prompt=TRAVEL_AGENT_SYSTEM_PROMPT,
            )

            prompt = (
                f"User Request: {user_query}\n"
                f"Source: {source_city}\n"
                f"Destination: {destination_city}\n"
                f"Trip Duration: {duration_days} days\n"
                f"Start Date: {start_date or 'Tomorrow'}\n"
                f"Travelers: {num_travelers}\n"
                f"Cabin Class: {travel_class}\n"
                f"Preferred Attraction Style: {preferred_category or 'Any'}\n"
                f"Total Budget Cap: ₹{total_budget if total_budget else 'Flexible'}\n\n"
                "Please autonomously call your tools to search flights, find hotels, discover attractions, "
                "retrieve the real-time weather from Open-Meteo, and compute the total budget. "
                "Provide a comprehensive, professional itinerary justification."
            )

            response = agent_graph.invoke({"messages": [HumanMessage(content=prompt)]})
            messages = response.get("messages", [])

            # Extract tool calls, reasoning trace, and tool outputs
            extracted_text = ""
            extracted_flight = None
            extracted_hotel = None

            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.content:
                        extracted_text += str(msg.content) + "\n"
                    if getattr(msg, "tool_calls", None):
                        for tc in msg.tool_calls:
                            tool_trace.append({
                                "tool": tc.get("name"),
                                "args": tc.get("args"),
                                "id": tc.get("id"),
                            })
                elif isinstance(msg, ToolMessage):
                    tool_trace.append({
                        "tool": msg.name or "tool",
                        "status": "completed",
                        "preview": str(msg.content)[:120],
                    })
                    try:
                        content_json = json.loads(msg.content)
                        if msg.name == "flight_search_tool" and content_json.get("status") == "success":
                            extracted_flight = content_json.get("cheapest_option") or (content_json.get("results", [])[0] if content_json.get("results") else None)
                        elif msg.name == "hotel_recommendation_tool" and content_json.get("status") == "success":
                            extracted_hotel = content_json.get("results", [])[0] if content_json.get("results") else None
                    except Exception:
                        pass

            # Directly feed tool outputs into the structured itinerary generator
            itinerary = self.generator.build_itinerary(
                source_city=source_city,
                destination_city=destination_city,
                duration_days=duration_days,
                start_date=start_date,
                num_travelers=num_travelers,
                total_budget=total_budget,
                travel_class=travel_class,
                preferred_category=preferred_category,
                preferred_amenities=preferred_amenities,
                daily_food_per_person=daily_food_per_person,
                daily_transport=daily_transport,
                other_buffer_per_person=other_buffer_per_person,
                selected_flight_data=extracted_flight,
                selected_hotel_data=extracted_hotel,
            )

            return {
                "is_llm_powered": True,
                "mode": f"Autonomous LangChain Agent ({self.model_name})",
                "tool_trace": tool_trace,
                "llm_response": extracted_text.strip() or "Autonomous tool execution complete.",
                "structured_itinerary": itinerary,
            }

        except Exception as e:
            logger.error(f"Error during LangChain LLM execution: {e}. Falling back to deterministic planner.", exc_info=True)
            itinerary = self.generator.build_itinerary(
                source_city=source_city,
                destination_city=destination_city,
                duration_days=duration_days,
                start_date=start_date,
                num_travelers=num_travelers,
                total_budget=total_budget,
                travel_class=travel_class,
                preferred_category=preferred_category,
                preferred_amenities=preferred_amenities,
                daily_food_per_person=daily_food_per_person,
                daily_transport=daily_transport,
                other_buffer_per_person=other_buffer_per_person,
            )
            return {
                "is_llm_powered": False,
                "mode": "Deterministic Fallback (LLM API Error)",
                "tool_trace": [{"tool": "error_handler", "status": "fallback", "error": str(e)}],
                "llm_response": f"Encountered LLM invocation issue: {e}. Automatically fell back to deterministic dataset-backed planner.",
                "structured_itinerary": itinerary,
            }
