# Data Dictionary: Agentic Travel Planning Assistant

This document outlines the schemas, data types, null distributions, transformations, and limitations of all datasets used across the project.

---

## 1. Flights Datasets

### A. Processed Flights (`data/processed/flights_clean.csv`)
- **Source:** `data/raw/Indian Airlines.csv` (Domestic flight fare observations).
- **Row Count:** 300,153 rows.
- **Duplicates:** 0 duplicate records.

| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `airline` | `string` | 0 | Operating airline (AirAsia, Air_India, GO_FIRST, Indigo, SpiceJet, Vistara) |
| `flight` | `string` | 0 | Flight identification number (e.g. `SG-8709`) |
| `source_city` | `string` | 0 | Departure city (`Delhi`, `Mumbai`, `Bangalore`, `Kolkata`, `Hyderabad`, `Chennai`) |
| `departure_time` | `string` | 0 | Time of day departure band (`Early_Morning`, `Morning`, `Afternoon`, `Night`) |
| `stops` | `string` | 0 | Stop categorization (`zero`, `one`, `two_or_more`) |
| `arrival_time` | `string` | 0 | Time of day arrival band |
| `destination_city` | `string` | 0 | Arrival destination city |
| `travel_class` | `string` | 0 | Cabin class (`Economy`, `Business`) |
| `days_left` | `integer` | 0 | Number of days between booking and flight date (1 to 49) |
| `duration_hours` | `float` | 0 | Total flight duration in hours |
| `price_inr` | `float` | 0 | Historical reference ticket price in Indian Rupees (INR) |

### B. Baseline Reference Flights (`reference/flights.json`)
- **Row Count:** 30 records.
- **Purpose:** Fast baseline flight references connecting non-hub leisure destinations (such as Goa and Jaipur).
- **Schema:** `flight_id` (str), `airline` (str), `from` (str), `to` (str), `departure_time` (ISO-8601), `arrival_time` (ISO-8601), `price` (float).

---

## 2. Hotels Datasets

### A. Processed Hotels (`data/processed/hotels_clean.csv`)
- **Source:** `data/raw/goibibo_com-travel_sample.csv` (Goibibo hotel crawl, 2016).
- **Row Count:** 4,000 unique properties.
- **Duplicates:** 0 duplicate property IDs.

| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `property_id` | `string` | 0 | Unique hotel identifier |
| `hotel_name` | `string` | 0 | Commercial name of property |
| `city` | `string` | 0 | Location city |
| `state` | `string` | 0 | Location state |
| `address` | `string` | 0 | Street address |
| `property_type` | `string` | 0 | Accommodation type (Hotel, Resort, Guest House) |
| `hotel_category` | `string` | 0 | Category tier |
| `star_rating` | `float` | 1,936 | Verified star rating (1.0 to 5.0). Zero values converted to Null. |
| `review_rating` | `float` | 1,584 | Guest review score on a 5.0 scale |
| `review_count` | `float` | 1,584 | Number of user reviews recorded |
| `hotel_facilities` | `string` | 194 | Pipe-separated list of hotel amenities |
| `room_facilities` | `string` | 270 | Pipe-separated list of in-room amenities |
| `room_type` | `string` | 0 | Room category (Deluxe, Standard, Suite) |
| `latitude` | `float` | 24 | GPS latitude coordinate |
| `longitude` | `float` | 24 | GPS longitude coordinate |
| `crawl_date` | `string` | 0 | Historical crawl date timestamp |
| `pageurl` | `string` | 0 | Original Goibibo listing URL |

### B. Baseline Reference Hotels (`reference/hotels.json`)
- **Row Count:** 40 records.
- **Purpose:** Provides verified nightly prices and standard amenities across key tourist cities for honest budget calculation.
- **Schema:** `hotel_id` (str), `name` (str), `city` (str), `stars` (int), `price_per_night` (float), `amenities` (list of str).

---

## 3. Places and Tourist Attractions Datasets

### A. Processed Destinations (`data/processed/places_clean.csv`)
- **Source:** `data/raw/destinations_clean(1).csv`.
- **Row Count:** 323 curated attraction records.
- **Duplicates:** 0 duplicate places.

| Column Name | Data Type | Null Count | Description |
| :--- | :--- | :--- | :--- |
| `zone` | `string` | 0 | Geographic region of India (Northern, Western, Southern, etc.) |
| `state` | `string` | 0 | Indian state or union territory |
| `city` | `string` | 0 | Destination city |
| `place_name` | `string` | 0 | Attraction title (e.g. `Baga Beach`, `Qutub Minar`, `Amber Palace`) |
| `type` | `string` | 0 | Category (Fort, Beach, Temple, Lake, National Park, Museum, Palace) |
| `establishment_year`| `float` | 131 | Historical construction or founding year |
| `visit_duration_hours` | `float` | 0 | Recommended visit time in hours |
| `rating` | `float` | 0 | Public Google rating on a 5.0 scale |
| `entrance_fee_inr` | `float` | 0 | Standard ticket fee in INR (0 for free public attractions) |
| `airport_within_50km` | `string` | 0 | Proximity indicator (`Yes` / `No`) |
| `weekly_off` | `string` | 291 | Day of week closed (e.g. `Monday`, `Friday`, or `None`) |
| `significance` | `string` | 0 | Cultural, architectural, or religious importance |
| `dslr_allowed` | `string` | 0 | Photography permission (`Yes` / `No`) |
| `review_count_lakhs` | `float` | 0 | Volume of reviews in Lakhs (100,000s) |
| `best_time_to_visit` | `string` | 0 | Recommended seasonal or daytime window |

### B. Baseline Reference Places (`reference/places.json`)
- **Row Count:** 40 records.
- **Schema:** `place_id` (str), `name` (str), `city` (str), `type` (str), `rating` (float).

---

## 4. Live API: Open-Meteo Weather Forecast

- **Endpoint:** `https://api.open-meteo.com/v1/forecast`
- **Authentication:** Free public API (no API key required).
- **Parameters:** `latitude`, `longitude`, `daily`, `timezone=auto`.
- **Fields Processed:**
  - `time`: Forecast date (`YYYY-MM-DD`).
  - `weathercode`: Numerical World Meteorological Organization (WMO) code.
  - `temperature_2m_max`: Maximum daily temperature in °C.
  - `temperature_2m_min`: Minimum daily temperature in °C.
  - `precipitation_probability_max`: Maximum chance of rain (0-100%).
  - `windspeed_10m_max`: Peak wind speed in km/h.

---

## 5. Data Integrity and Limitations

1. **Static / Historical Fares:** Domestic flight records and Goibibo listings are historical research samples. They provide realistic fare ranges and property profiles for discovery, but do not represent live reservation inventory.
2. **Missing Hotel Nightly Prices:** The original Goibibo crawl did not capture reliable room rates. The project applies baseline prices from `hotels.json` or city median reference values for budget calculation, clearly labeling them as reference estimates.
3. **No Price Fabrication:** Missing fields are never artificially invented; Null ratings or fees are handled honestly in both code and user interface.
