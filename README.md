# Agentic AI-Based Travel Planning Assistant Using LangChain

[![Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-1.4+-green.svg)](https://www.langchain.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.65+-red.svg)](https://streamlit.io/)
[![Pytest](https://img.shields.io/badge/Tests-32%20Passed-brightgreen.svg)](https://pytest.org/)

An academic-grade, autonomous travel concierge that accepts natural-language trip requests and dynamically orchestrates specialized tools to plan personalized, realistically paced day-wise itineraries with real-time weather and transparent budget breakdowns.

---

## 1. Project Overview & Problem Statement

Planning a trip requires searching flights, evaluating hotel options, discovering sightseeing spots, verifying opening hours and entry tickets, checking destination weather, and calculating a cohesive budget. Travelers frequently juggle dozens of disjointed tabs, compare conflicting information, and manually draft schedules that are either rushed or impractical.

This project delivers an **autonomous Agentic AI assistant** that:
1. Understands unstructured natural-language travel requests.
2. Autonomously selects and executes tools for flight search, hotel recommendations, tourist attraction discovery, weather forecasts, and budget calculations.
3. Produces a cohesive, realistic day-wise schedule with morning, afternoon, and evening activities.
4. Explains the rationale behind every selection.
5. Provides both an interactive **Streamlit** user interface and structured **Pydantic** JSON exports.

---

## 2. Key Objectives

- **Autonomous Tool-Calling Reasoning:** Implement LangChain's `create_agent` tool-calling state graph where the LLM dynamically decides which tools to invoke based on user prompts.
- **Multi-Source Integration:** Combine 300,000+ historical domestic flight records, a 4,000-hotel catalogue, and 323 curated Indian tourist destinations with the live **Open-Meteo** weather API.
- **Budget Integrity:** Calculate total trip expenses with itemized transparency according to the academic formula:
  $$\text{Total Budget} = \text{Flight Cost} + \text{Hotel Cost} + \text{Food Estimate} + \text{Local Transport} + \text{Sightseeing Fees} + \text{Contingency Buffer}$$
- **Data Transparency & Honesty:** Clearly label historical/static datasets and never fabricate missing hotel prices, flight schedules, or weather forecasts.
- **Graceful Fallback / Demo Mode:** Provide an honest, fully functional deterministic demo mode when an OpenAI API key is not configured.

---

## 3. System Architecture

```
User / Traveler
      │ (Natural Language Prompt + Route & Preference Settings)
      ▼
Streamlit Interface (app.py)
      │
TravelAgentRunner Orchestration
      ├───────────────────────────────┬───────────────────────────────┐
      ▼                               ▼                               ▼
Autonomous LangChain Agent       Deterministic Planner          Singleton DataLoader
(ChatOpenAI + create_agent)           (Demo Mode)             (In-Memory CSV/JSON Cache)
      │                               │                               │
      └───────────────────────────────┴───────────────────────────────┘
                                      │
              ┌───────────────────────┼───────────────────────┐
              ▼                       ▼                       ▼
    flight_search_tool      hotel_recommendation     places_discovery_tool
    (300K+ Flight CSV)       (4K Hotel Catalogue)     (323 Tourist Spots)
              │                       │                       │
              └───────────────────────┼───────────────────────┘
                                      │
              ┌───────────────────────┴───────────────────────┐
              ▼                                               ▼
     weather_lookup_tool                            budget_estimation_tool
   (Live Open-Meteo REST API)                       (Itemized Math Engine)
                                      │
                                      ▼
                        Pydantic Schema Validation
                     (CompleteTravelItinerary Model)
                                      │
                                      ▼
                 Interactive UI Display & JSON Export
```

---

## 4. Technology Stack

- **Core Runtime:** Python 3.10.11
- **Agentic AI Framework:** LangChain 1.4+, LangGraph, `langchain-openai`, `langchain-core`
- **User Interface:** Streamlit 1.65+
- **Data Engineering:** Pandas 2.3+, NumPy 2.2+
- **Data Validation & Schemas:** Pydantic v2
- **External Live API:** Open-Meteo Free Forecast API (no API key required)
- **Automated Testing:** Pytest 9.2+

---

## 5. Dataset Descriptions & Limitations

| Dataset | File Path | Records | Description & Attribution |
| :--- | :--- | :--- | :--- |
| **Flights (CSV)** | `data/processed/flights_clean.csv` | 300,153 | Historical domestic fares across major Indian hubs (Indian Airlines dataset). |
| **Flights (JSON)** | `reference/flights.json` | 30 | Reference baseline connecting non-hub leisure destinations like Goa and Jaipur. |
| **Hotels (CSV)** | `data/processed/hotels_clean.csv` | 4,000 | Goibibo crawl (2016) with facilities, coordinates, addresses, and ratings. |
| **Hotels (JSON)** | `reference/hotels.json` | 40 | Verified nightly room rates across key tourist cities for honest budgeting. |
| **Places (CSV)** | `data/processed/places_clean.csv` | 323 | Curated Indian destinations with categories, visit durations, and ticket fees. |
| **Places (JSON)** | `reference/places.json` | 40 | Baseline attraction records across key cities. |

> **Data Notice:** Flight and hotel datasets are historical reference datasets for research and simulation. They do not represent live commercial booking inventory. Missing hotel room rates in the Goibibo crawl are supplemented with verified baseline values or city medians, explicitly tagged as reference estimates.

---

## 6. Directory Structure

```
AI-Travel-Planner/
│
├── app.py                      # Interactive Streamlit Web Application
├── config.py                   # Centralized configuration & city coordinates
├── requirements.txt            # Python package dependencies
├── .env                        # Local environment variables (API keys)
├── .env.example                # Sample environment template
├── .gitignore                  # Git ignore rules
│
├── data/
│   ├── raw/                    # Untouched source datasets
│   └── processed/              # Normalized CSVs (flights, hotels, places)
│
├── reference/
│   ├── flights.json            # Baseline flight references
│   ├── hotels.json             # Baseline hotel prices
│   └── places.json             # Baseline attraction spots
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py          # Singleton cached data loader
│   ├── flight_tool.py          # Flight search tool & @tool
│   ├── hotel_tool.py           # Hotel recommendation tool & @tool
│   ├── places_tool.py          # Tourist attraction tool & @tool
│   ├── weather_tool.py         # Open-Meteo live forecast tool & @tool
│   ├── budget_tool.py          # Itemized budget calculation tool & @tool
│   ├── prompts.py              # Pydantic schemas & Agent system prompts
│   ├── itinerary_generator.py  # Itinerary synthesis engine
│   └── travel_agent.py         # LangChain Agent runner & Demo Mode
│
├── tests/
│   ├── __init__.py
│   ├── test_flights.py         # Flight search & route tests
│   ├── test_hotels.py          # Hotel recommendation & pricing tests
│   ├── test_places.py          # Attraction discovery & category tests
│   ├── test_weather.py         # Open-Meteo API & code parsing tests
│   ├── test_budget.py          # Itemized budget formula tests
│   └── test_itinerary.py       # End-to-end itinerary & schema tests
│
├── docs/
│   ├── system_architecture.md   # Architectural design documentation
│   ├── data_dictionary.md       # Dataset schemas, types & nulls
│   ├── project_workflow.md      # Workflow lifecycle & sequence diagrams
│   └── mock_interview_questions.md # Top 5 evaluation Q&As & PPT outline
│
├── outputs/                    # Exported itinerary JSON files
└── README.md                   # Project documentation
```

---

## 7. Installation & Setup

### Prerequisites
- Python 3.10 installed on your system.

### Step 1: Clone / Navigate to Project Directory
```bash
cd D:\INTERNSHIP
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables (Optional)
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Add your OpenAI API key if you wish to run the LLM autonomous agent:
```ini
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4o-mini
```
*(If no key is configured, the application automatically runs in full Demo Mode with live Open-Meteo weather and dataset filtering).*

---

## 8. How to Run the Application

Launch the Streamlit web dashboard:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 9. Running Automated Tests

Run the complete test suite with `pytest`:
```bash
pytest tests -v
```

All **32 tests** execute locally without requiring an OpenAI API key (external network calls are mocked or handled with fallback resilience).

---

## 10. Sample Output Preview

```json
{
  "trip_title": "3-Day Personalized Tour: Delhi to Goa",
  "source_city": "Delhi",
  "destination_city": "Goa",
  "trip_duration_days": 3,
  "num_travelers": 2,
  "flight": {
    "airline": "Air India",
    "flight_number": "AI-902",
    "source": "Delhi",
    "destination": "Goa",
    "departure_time": "07:30",
    "arrival_time": "10:00",
    "duration_hours": 2.5,
    "stops": "zero",
    "travel_class": "Economy",
    "price_inr": 4200.0,
    "selection_reason": "Standard hub connection fare (historical baseline)"
  },
  "hotel": {
    "hotel_name": "Comfort Suites",
    "city": "Goa",
    "address": "Central Goa",
    "star_rating": 5.0,
    "review_rating": 5.0,
    "price_per_night_inr": 3650.0,
    "amenities": ["gym", "breakfast", "wifi", "parking"]
  },
  "budget": {
    "grand_total_inr": 31756.0,
    "per_person_inr": 15878.0,
    "flights_total_inr": 16800.0,
    "hotel_total_inr": 7300.0,
    "food_total_inr": 4800.0,
    "local_transport_total_inr": 1500.0,
    "sightseeing_total_inr": 356.0,
    "other_buffer_inr": 1000.0
  }
}
```

---

## 11. Academic Evaluation Resources

- [System Architecture Guide](docs/system_architecture.md)
- [Data Dictionary & Null Distributions](docs/data_dictionary.md)
- [Project Workflow & Sequence Diagrams](docs/project_workflow.md)
- [Mock Interview Questions & PPT Outline](docs/mock_interview_questions.md)

---

## 12. Future Scope

1. **Live Booking Engine Integration:** Incorporate real-time GDS flight reservation APIs (e.g., Amadeus or Sabre) and live hotel booking endpoints.
2. **Geographic Route Optimization:** Use the Google Maps Distance Matrix API to sequence daily attractions with minimal in-city transit times.
3. **Multi-Modal Conversational Input:** Enable speech-to-text voice interaction for mobile travelers.
