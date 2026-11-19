# System Architecture: Agentic AI-Based Travel Planning Assistant

## 1. Architectural Overview

The **Agentic AI Travel Planning Assistant** is designed as a modular, tool-calling agentic system developed with **LangChain**, **Streamlit**, and Python 3.10.

The architecture decouples user interface interaction, agent reasoning, deterministic tool execution, and dataset access layers:

```
+--------------------------------------------------------------------------------+
|                             Streamlit Web Interface                            |
|          (Natural Language Query, Route Selection, Preferences, Budget)        |
+---------------------------------------+----------------------------------------+
                                        |
                                        v
+--------------------------------------------------------------------------------+
|                        TravelAgentRunner Orchestration                         |
|   +---------------------------------------+--------------------------------+   |
|   |       Autonomous LangChain Agent      |     Deterministic Planner      |   |
|   |    (ChatOpenAI + create_agent graph)  |          (Demo Mode)           |   |
|   +-------------------+-------------------+----------------+---------------+   |
|                       |                                    |                   |
+-----------------------|------------------------------------|-------------------+
                        v                                    v
+--------------------------------------------------------------------------------+
|                           Agent Tool Execution Layer                           |
|                                                                                |
|  +------------------------+  +------------------------+  +------------------+  |
|  |   flight_search_tool   |  | hotel_recommendation   |  | places_discovery |  |
|  +-----------+------------+  +-----------+------------+  +--------+---------+  |
|              |                           |                        |            |
|  +-----------+------------+  +-----------+------------+           |            |
|  |  weather_lookup_tool   |  | budget_estimation_tool |           |            |
|  +-----------+------------+  +-----------+------------+           |            |
+--------------|---------------------------|------------------------|------------+
               |                           |                        |
               v                           v                        v
+-------------------------------+  +---------------------------------------------+
|    External Live Services     |  |         Local Data Persistence Layer        |
|                               |  |                                             |
|  Open-Meteo Weather API       |  |  - Indian Airlines CSV (300,153 rows)       |
|  (Free, No Key Required,      |  |  - Goibibo Hotel Catalogue (4,000 rows)     |
|   WMO Condition Codes)        |  |  - Destinations Places (323 curated spots)  |
|                               |  |  - Reference JSON Baselines (Flights,       |
|                               |  |    Hotels, Places)                          |
+-------------------------------+  +---------------------------------------------+
                                        |
                                        v
+--------------------------------------------------------------------------------+
|                          Pydantic Validation Layer                             |
|          (CompleteTravelItinerary Schema -> JSON Dump & UI Visualization)      |
+--------------------------------------------------------------------------------+
```

---

## 2. Component Descriptions

### A. Presentation Layer (`app.py`)
- **Technology:** Streamlit 1.65+
- **Responsibilities:**
  - Captures natural language travel queries, departure/return dates, traveler counts, and style preferences.
  - Exposes an AI Agent Settings panel supporting secure OpenAI API key entry, model selection, or one-click Deterministic Demo Mode.
  - Visualizes day-wise itineraries, flight routes, hotel amenities, 5-day weather forecasts, and itemized cost breakdowns with charts.
  - Provides a one-click download of the complete validated JSON itinerary.

### B. Agentic Reasoning Layer (`src/travel_agent.py`)
- **Technology:** LangChain 1.4+, LangGraph compiled state graphs, `langchain_openai.ChatOpenAI`.
- **Responsibilities:**
  - Evaluates user intent and decomposes complex trip planning into discrete tool invocations.
  - Dynamically decides which tool to call, processes tool returns, and maintains reasoning trace.
  - Implements an honest fallback to the deterministic planner if no API key is supplied or if network quotas are exceeded.

### C. Tool Execution Layer
1. **Flight Search Tool (`src/flight_tool.py`):**
   - Vectorized filtering on the 300,153-record Indian Airlines dataset and baseline JSON records.
   - Evaluates routes, flight durations, stops, and fares; identifies cheapest and fastest options.
2. **Hotel Recommendation Tool (`src/hotel_tool.py`):**
   - Filters the 4,000-property Goibibo catalogue by location, rating, and amenities.
   - Cross-references baseline JSON records for verified nightly rates; applies transparent city median baselines where prices were omitted in the crawl.
3. **Places Discovery Tool (`src/places_tool.py`):**
   - Discovers tourist spots from 323 curated destinations across India.
   - Filters by category (Beach, Fort, Temple, Museum, Nature), entrance fee, and visit duration.
4. **Weather Tool (`src/weather_tool.py`):**
   - Connects live to the Open-Meteo Forecast API using destination latitude/longitude.
   - Parses high/low temperatures, precipitation probabilities, and converts WMO codes to travel recommendations.
5. **Budget Estimation Tool (`src/budget_tool.py`):**
   - Computes total cost strictly according to:
     $$\text{Total} = \text{Flights} + \text{Hotels} + \text{Food} + \text{Transport} + \text{Sightseeing} + \text{Buffer}$$
   - Separates dataset-referenced values from configurable daily allowances.

### D. Data Persistence & Caching (`src/data_loader.py`)
- Implements a singleton in-memory caching pattern to eliminate repeated CSV parsing latency on large 22MB datasets.
- Sanitizes and normalizes city aliases (e.g., "Bengaluru" -> "Bangalore", "Bombay" -> "Mumbai").

### E. Output Validation (`src/prompts.py` & `src/itinerary_generator.py`)
- Uses strict **Pydantic v2** models to enforce type safety, non-null guarantees, and consistent JSON output schemas.
