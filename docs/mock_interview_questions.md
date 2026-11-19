# Academic Evaluation: Mock Interview Questions & Presentation Guide

## Part 1: Top 5 Evaluation Questions & Answers

### Question 1: What makes this system an "Agentic AI" rather than a traditional hardcoded script?
**Answer:**
> A traditional script follows a rigid, linear sequence where functions are executed regardless of user intent. In this project, we implement an **autonomous tool-calling agent using LangChain**. 
> The Language Model (e.g. `ChatOpenAI`) acts as a reasoning core. Given an unstructured, natural language travel query, the model inspects the schemas and docstrings of available tools (`flight_search_tool`, `hotel_recommendation_tool`, `places_discovery_tool`, `weather_lookup_tool`, `budget_estimation_tool`), autonomously determines which tools are needed, extracts parameters from context, and evaluates tool outputs dynamically before deciding its next step. Furthermore, when an API key is not configured, the system provides a transparent, deterministic fallback mode that clearly explains its operating state rather than fabricating agent intelligence.

---

### Question 2: How does the system handle historical datasets versus real-time data, and what are their limitations?
**Answer:**
> The system operates under strict transparency guidelines regarding data freshness:
> 1. **Historical Datasets:** Domestic flights (300,153 records from Indian Airlines) and hotel records (4,000 Goibibo crawl records) provide realistic route patterns, flight durations, property facilities, and reference pricing. However, they do not represent live reservation inventory. Missing nightly room rates in the Goibibo crawl are never fabricated; instead, the system references baseline rates from `hotels.json` or city median values, explicitly labeling them as historical estimates.
> 2. **Real-Time Data:** Live multi-day weather forecasts are retrieved on-demand via the free **Open-Meteo API** using geographic coordinates. If the API is unreachable, the system catches the timeout gracefully and provides seasonal climate indicators without inventing false temperatures.

---

### Question 3: How does the budget estimation module work, and how does it prevent inaccurate total calculations?
**Answer:**
> The budget estimation tool adheres to a clear mathematical formula:
> $$\text{Total Budget} = \text{Flight Cost} + \text{Hotel Cost} + \text{Food Estimate} + \text{Local Transport} + \text{Entrance Fees} + \text{Contingency Buffer}$$
> To prevent inaccurate additions:
> - **Dataset-backed values:** Flight fares (scaled by round-trip multiplier and number of passengers), hotel room rates (scaled by nights and room counts), and monument entrance fees (from the curated places dataset) are isolated.
> - **Configurable daily allowances:** Meal expenses and local transit costs are parameterized so the user or agent can adjust them according to travel style.
> - The system itemizes these categories separately in a structured table and chart so the traveler understands exactly which numbers are data-derived and which are estimates.

---

### Question 4: How is modularity and software engineering best practice maintained across the codebase?
**Answer:**
> The project follows **PEP 8** standards, strict modular separation, and defensive programming:
> - `config.py`: Centralized path resolution, environment variables, and city coordinates.
> - `src/data_loader.py`: Implements a singleton in-memory caching pattern to avoid re-reading 22MB CSVs on every tool execution.
> - `src/*_tool.py`: Each tool is decoupled into an independent Python function (easily unit-tested in isolation) and a `@tool` decorator for LangChain integration.
> - `src/prompts.py`: Uses **Pydantic v2** models to guarantee structured, validated schema outputs.
> - `tests/`: 32 automated unit tests with `pytest` covering all tool branches, edge cases, and mocked network responses without requiring a paid LLM key.

---

### Question 5: Why is model accuracy or precision/recall not reported for this project?
**Answer:**
> This system is an **Agentic AI orchestration and retrieval-augmented synthesis application**, not a supervised machine learning classification or regression model trained on labeled ground truth.
> Therefore, standard ML metrics like accuracy, ROC-AUC, or F1-score are technically inapplicable. Instead, the quality of this system is evaluated through:
> 1. **Deterministic tool correctness:** Verified via 32 automated unit tests.
> 2. **Schema validation:** 100% adherence to Pydantic data schemas for JSON outputs.
> 3. **Pacing and feasibility:** Ensuring daylight hours, visit durations, and opening days (weekly off) are logically coordinated without overloading the traveler.

---

## Part 2: Concise Project Presentation Outline (PPT)

### Slide 1: Title & Introduction
- **Project Title:** Agentic AI-Based Travel Planning Assistant Using LangChain
- **Domain:** Artificial Intelligence / Autonomous Agents / Tourism Tech
- **Technologies:** Python 3.10, LangChain, Streamlit, Open-Meteo API, Pydantic, Pandas.

### Slide 2: Problem Statement & Motivation
- Travel planning is fragmented across multiple platforms (booking engines, review sites, blogs, weather apps).
- Manually synthesizing flights, hotel options, daily sightseeing routes, and budgets is time-consuming and often leads to unrealistic itineraries.
- **Solution:** An autonomous agent that ingests natural language requests, dynamically gathers data across specialized tools, and outputs a complete, validated itinerary.

### Slide 3: System Architecture
- Decoupled 4-tier design:
  1. **User Interface:** Interactive Streamlit dashboard.
  2. **Agent Core:** LangChain Tool-Calling Agent (`create_agent` state graph) with deterministic fallback.
  3. **Tool Layer:** 5 specialized tools for Flights, Hotels, Places, Weather, and Budget.
  4. **Data Layer:** 300K+ flight records, 4K hotel records, 323 attractions, and live Open-Meteo REST API.

### Slide 4: Specialized Agent Tools
- **Flight Search:** Evaluates routes, cheapest vs. fastest flights, and travel class.
- **Hotel Recommendation:** Filters properties by rating, location, and verified baseline prices.
- **Places Discovery:** Recommends curated tourist attractions with entry fees and duration caps.
- **Live Weather Lookup:** Fetches real-time multi-day temperature and rain probabilities from Open-Meteo.
- **Budget Estimation:** Transparent itemized cost accounting with customizable daily rates.

### Slide 5: Quality Assurance & Testing
- 32 comprehensive automated test cases with `pytest` (100% pass rate).
- Validated Pydantic models guaranteeing structured JSON exports.
- Robust error handling for network timeouts, missing fields, and identical origin/destination cities.

### Slide 6: Dataset Limitations & Future Scope
- **Limitations:** Historical flight and hotel crawl data serve as planning baselines rather than live booking engines.
- **Future Scope:** Integration of live GDS flight booking APIs (Amadeus/Sabre), dynamic route distance optimization via Google Maps API, and multi-modal voice interactions.
