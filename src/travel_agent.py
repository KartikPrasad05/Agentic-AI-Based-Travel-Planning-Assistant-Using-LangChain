"""LangChain agent for the Agentic AI Travel Planning Assistant.

Uses Gemini when configured, falls back to OpenAI when available, and
otherwise runs the deterministic planner without claiming LangChain tool calls.
"""

import json
import logging
from typing import Dict, List, Any, Optional

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain.agents import create_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI

import config
from src.flight_tool import flight_search_tool
from src.hotel_tool import hotel_recommendation_tool
from src.places_tool import places_discovery_tool
from src.weather_tool import weather_lookup_tool
from src.budget_tool import budget_estimation_tool
from src.prompts import TRAVEL_AGENT_SYSTEM_PROMPT
from src.itinerary_generator import get_itinerary_generator

logger = logging.getLogger(__name__)

TRAVEL_TOOLS = [
    flight_search_tool,
    hotel_recommendation_tool,
    places_discovery_tool,
    weather_lookup_tool,
    budget_estimation_tool,
]


class TravelAgentRunner:
    """Run a provider-backed LangChain agent or deterministic fallback."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        force_demo_mode: bool = False,
        provider: Optional[str] = None,
    ):
        # An explicitly supplied legacy api_key is treated as an OpenAI key.
        if api_key:
            self.provider = (provider or "openai").lower()
            self.api_key = api_key
            self.model_name = model_name or (
                config.GEMINI_MODEL if self.provider == "gemini" else config.OPENAI_MODEL
            )
        else:
            self.provider = (provider or (
                "gemini" if config.GEMINI_API_KEY else
                "openai" if config.OPENAI_API_KEY else
                "none"
            )).lower()
            self.api_key = (
                config.GEMINI_API_KEY if self.provider == "gemini"
                else config.OPENAI_API_KEY if self.provider == "openai"
                else ""
            )
            self.model_name = model_name or (
                config.GEMINI_MODEL if self.provider == "gemini"
                else config.OPENAI_MODEL if self.provider == "openai"
                else ""
            )

        self.force_demo_mode = force_demo_mode
        self.generator = get_itinerary_generator()

    @property
    def has_llm_key(self) -> bool:
        """Return whether a non-placeholder key is configured for the provider."""
        if self.force_demo_mode or self.provider not in {"gemini", "openai"}:
            return False
        key = (self.api_key or "").strip()
        placeholders = {
            "your_openai_api_key_here",
            "your_gemini_api_key_here",
            "paste_your_actual_gemini_key_here",
        }
        return bool(key and key not in placeholders and len(key) > 20)

    def _build_llm(self):
        """Create the configured chat model."""
        if self.provider == "gemini":
            return ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=self.api_key,
                temperature=0.3,
            )
        if self.provider == "openai":
            return ChatOpenAI(
                model=self.model_name,
                api_key=self.api_key,
                temperature=0.3,
            )
        raise ValueError("No supported LLM provider is configured.")

    def _build_itinerary(self, **kwargs):
        """Delegate itinerary construction to the deterministic structured generator."""
        return self.generator.build_itinerary(**kwargs)

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
        """Plan a trip and return mode, actual tool trace, response and itinerary."""
        itinerary_args = dict(
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

        if not self.has_llm_key or force_demo_mode:
            logger.info("Running deterministic planner; no LLM tool calls are made.")
            itinerary = self._build_itinerary(**itinerary_args)
            return {
                "is_llm_powered": False,
                "mode": "Deterministic Planner (Demo Mode)",
                "tool_trace": [],
                "llm_response": (
                    "Demo Mode: the deterministic itinerary generator created this plan "
                    "using the configured datasets, weather service and budget calculations. "
                    "No LangChain agent or LLM tool calls were executed."
                ),
                "structured_itinerary": itinerary,
            }

        try:
            logger.info("Initializing %s model %s", self.provider, self.model_name)
            agent_graph = create_agent(
                model=self._build_llm(),
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
                "Use the available tools to search flights, hotels, attractions, weather "
                "and estimate the budget. Explain important selections and limitations."
            )
            response = agent_graph.invoke({"messages": [HumanMessage(content=prompt)]})
            messages = response.get("messages", [])

            extracted_text = ""
            extracted_flight = None
            extracted_hotel = None
            tool_trace = []

            for msg in messages:
                if isinstance(msg, AIMessage):
                    if msg.content:
                        extracted_text += str(msg.content) + "\n"
                    for call in getattr(msg, "tool_calls", []) or []:
                        tool_trace.append({
                            "tool": call.get("name"),
                            "status": "requested",
                            "args": call.get("args"),
                            "id": call.get("id"),
                        })
                elif isinstance(msg, ToolMessage):
                    tool_trace.append({
                        "tool": msg.name or "tool",
                        "status": "completed",
                        "preview": str(msg.content)[:300],
                    })
                    try:
                        raw = msg.content
                        content_json = json.loads(raw) if isinstance(raw, str) else raw
                        if not isinstance(content_json, dict):
                            continue
                        if msg.name == "flight_search_tool" and content_json.get("status") == "success":
                            results = content_json.get("results") or []
                            extracted_flight = content_json.get("cheapest_option") or (results[0] if results else None)
                        elif msg.name == "hotel_recommendation_tool" and content_json.get("status") == "success":
                            results = content_json.get("results") or []
                            extracted_hotel = results[0] if results else None
                    except (TypeError, ValueError, AttributeError):
                        logger.warning("Could not parse structured output from tool %s", msg.name)

            itinerary = self._build_itinerary(
                **itinerary_args,
                selected_flight_data=extracted_flight,
                selected_hotel_data=extracted_hotel,
            )
            return {
                "is_llm_powered": True,
                "mode": f"Autonomous LangChain Agent ({self.provider}: {self.model_name})",
                "tool_trace": tool_trace,
                "llm_response": extracted_text.strip() or "Agent execution completed.",
                "structured_itinerary": itinerary,
            }

        except Exception as exc:
            logger.exception("LLM agent failed; using deterministic fallback.")
            itinerary = self._build_itinerary(**itinerary_args)
            return {
                "is_llm_powered": False,
                "mode": "Deterministic Fallback (LLM API Error)",
                "tool_trace": [{
                    "tool": "error_handler",
                    "status": "fallback",
                    "error": str(exc),
                }],
                "llm_response": (
                    f"The {self.provider} agent could not complete the request: {exc}. "
                    "A deterministic dataset-backed itinerary was generated instead."
                ),
                "structured_itinerary": itinerary,
            }
