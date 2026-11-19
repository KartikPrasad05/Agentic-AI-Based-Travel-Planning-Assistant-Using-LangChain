from pathlib import Path
import pandas as pd
import numpy as np
import json
import shutil

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed"
OUT.mkdir(parents=True, exist_ok=True)

def clean_text(series):
    return series.astype("string").str.strip().replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

# Flights: historical fare observations, not live inventory.
flights = pd.read_csv(RAW / "Indian Airlines.csv")
flights = flights.drop(columns=["Unnamed: 0"], errors="ignore")
flights.columns = [c.strip().lower() for c in flights.columns]
for col in ["airline", "flight", "source_city", "departure_time", "stops",
            "arrival_time", "destination_city", "class"]:
    flights[col] = clean_text(flights[col])
flights["duration_hours"] = pd.to_numeric(flights["duration"], errors="coerce")
flights["price_inr"] = pd.to_numeric(flights["price"], errors="coerce")
flights["days_left"] = pd.to_numeric(flights["days_left"], errors="coerce").astype("Int64")
flights = flights.drop(columns=["duration", "price"]).rename(columns={"class": "travel_class"})
flights.to_csv(OUT / "flights_clean.csv", index=False)

# Hotels: no reliable nightly price is present in this historical 2016 crawl.
hotels = pd.read_csv(RAW / "goibibo_com-travel_sample.csv")
hotels = hotels.rename(columns={
    "property_name": "hotel_name",
    "site_review_rating": "review_rating",
    "site_review_count": "review_count",
    "hotel_star_rating": "raw_star_rating",
})
for col in ["hotel_name", "city", "state", "address", "property_type",
            "hotel_category", "hotel_facilities", "room_facilities",
            "room_type", "pageurl", "crawl_date"]:
    hotels[col] = clean_text(hotels[col])
for col in ["raw_star_rating", "review_rating", "review_count", "latitude", "longitude"]:
    hotels[col] = pd.to_numeric(hotels[col], errors="coerce")
hotels["star_rating"] = hotels["raw_star_rating"].replace(0, np.nan)
hotel_cols = [
    "property_id", "hotel_name", "city", "state", "address", "property_type",
    "hotel_category", "star_rating", "review_rating", "review_count",
    "hotel_facilities", "room_facilities", "room_type", "latitude",
    "longitude", "crawl_date", "pageurl",
]
hotels = hotels[hotel_cols].drop_duplicates(subset=["property_id"], keep="first")
hotels.to_csv(OUT / "hotels_clean.csv", index=False)

# Places: discard undocumented precomputed scores; preserve source attributes.
places = pd.read_csv(RAW / "destinations_clean(1).csv")
places = places.drop(columns=[
    "Unnamed: 0", "popularity_tier", "cost_category", "rating_tier",
    "value_score", "performance_score", "needs_attention",
], errors="ignore")
places = places.rename(columns={
    "destination_name": "place_name",
    "google_rating": "rating",
    "time_needed_hrs": "visit_duration_hours",
    "reviews_lakhs": "review_count_lakhs",
})
for col in ["zone", "state", "city", "place_name", "type", "weekly_off",
            "significance", "best_time_to_visit", "airport_within_50km", "dslr_allowed"]:
    if col in places:
        places[col] = clean_text(places[col])
for col in ["rating", "entrance_fee_inr", "visit_duration_hours",
            "establishment_year", "review_count_lakhs"]:
    places[col] = pd.to_numeric(places[col], errors="coerce")
places = places.drop_duplicates(subset=["place_name", "city", "state"], keep="first")
places.to_csv(OUT / "places_clean.csv", index=False)

report = {}
for name, frame in [("flights", flights), ("hotels", hotels), ("places", places)]:
    report[name] = {
        "rows": int(len(frame)),
        "columns": list(frame.columns),
        "missing_values": {k: int(v) for k, v in frame.isna().sum().items() if v > 0},
        "duplicate_rows": int(frame.duplicated().sum()),
    }
(ROOT / "validation_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps({k: {"rows": v["rows"], "duplicate_rows": v["duplicate_rows"]}
                  for k, v in report.items()}, indent=2))
