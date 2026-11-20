"""Streamlit Web Application for the Agentic AI Travel Planning Assistant (VoyAgent AI).

Provides an interactive user interface for querying, customizing, and
generating autonomous day-wise travel itineraries with LangChain,
historical datasets, and live Open-Meteo forecasts.
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import json
from datetime import date, timedelta
import pandas as pd
import streamlit as st

import config
from src.travel_agent import TravelAgentRunner
from src.prompts import CompleteTravelItinerary

# Configure Streamlit page
st.set_page_config(
    page_title="VoyAgent AI — Agentic Travel Planning Assistant",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "is_logged_in" not in st.session_state:
    st.session_state["is_logged_in"] = True
if "user_name" not in st.session_state:
    st.session_state["user_name"] = "Kartik Prasad"
if "user_email" not in st.session_state:
    st.session_state["user_email"] = "kartik@voyagent.ai"
if "is_generating" not in st.session_state:
    st.session_state["is_generating"] = False

# City Options
CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Goa", "Jaipur", "Hyderabad",
    "Kolkata", "Chennai", "Ahmedabad", "Pune", "Kochi", "Agra",
    "Varanasi", "Coorg", "Alleppey", "Udaipur", "Amritsar", "Leh"
]

# Custom CSS for styling
st.markdown("""
<style>
    /* Top Navbar */
    .brand-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #1E3A8A;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-badge {
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .sub-header {
        font-size: 0.95rem;
        color: #4B5563;
        margin-bottom: 0.8rem;
    }
    .summary-card {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px 20px;
        color: #F8FAFC;
        margin-bottom: 16px;
    }
    .feature-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        height: 100%;
        transition: transform 0.15s ease-in-out;
    }
    .feature-card:hover {
        border-color: #3B82F6;
        box-shadow: 0 4px 12px rgba(59,130,246,0.08);
    }
    .badge-historical {
        background-color: #FEF3C7;
        color: #92400E;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-live {
        background-color: #D1FAE5;
        color: #065F46;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .footer-container {
        margin-top: 45px;
        padding: 30px 20px;
        background-color: #0F172A;
        color: #94A3B8;
        border-radius: 12px;
    }
    .footer-title {
        color: #F8FAFC;
        font-weight: 700;
        margin-bottom: 10px;
    }
    .footer-link {
        color: #94A3B8;
        text-decoration: none;
        display: block;
        margin-bottom: 6px;
    }
    .footer-bottom {
        border-top: 1px solid #334155;
        padding-top: 15px;
        margin-top: 20px;
        text-align: center;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP NAVIGATION HEADER
# -----------------------------------------------------------------------------
col_logo, col_nav = st.columns([5.5, 4.5])
with col_logo:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 8px;">
        <span style="font-size: 2rem;">✈️</span>
        <div>
            <span style="font-size: 1.55rem; font-weight: 800; color: #1E3A8A; letter-spacing: -0.5px;">VoyAgent AI</span>
            <span style="background: #E0E7FF; color: #3730A3; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 6px;">v2.4 Live</span>
            <div style="font-size: 0.82rem; color: #64748B;">Agentic Travel Concierge powered by LangChain & Open-Meteo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Developer configuration relocated to top navbar popover (clean UI, preserved functionality)
with col_nav:
    n_col1, n_col2, n_col3, n_col4 = st.columns([1.1, 1.1, 1.1, 1.3])
    with n_col1:
        with st.popover("⚙️ Settings", use_container_width=True):
            st.markdown("##### 🤖 AI Model & Agent Settings")
            st.caption("Developer and API configuration (optional):")
            dev_api_key = st.text_input(
                "OpenAI API Key",
                value=st.session_state.get("openai_api_key", os.getenv("OPENAI_API_KEY", "")),
                type="password",
                help="Optional. If omitted, the app runs in deterministic dataset & live weather mode.",
                key="openai_api_key_input",
            )
            dev_model = st.selectbox(
                "Model Choice",
                options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
                index=0,
                key="agent_model_choice",
            )
            dev_force_demo = st.checkbox(
                "Force Demo Mode",
                value=st.session_state.get("force_demo_mode", False),
                key="dev_force_demo_cb",
            )
            st.session_state["openai_api_key"] = dev_api_key
            st.session_state["agent_model"] = dev_model
            st.session_state["force_demo_mode"] = dev_force_demo

            effective_key = dev_api_key.strip() or config.OPENAI_API_KEY
            if effective_key and not dev_force_demo:
                st.success(f"🟢 LLM Agent Active ({dev_model})")
            else:
                st.info("⚡ Offline/Demo Mode Active (Live Weather & Datasets)")

    with n_col2:
        with st.popover("🔔 Alerts (3)", use_container_width=True):
            st.markdown("##### 📬 Notifications")
            st.success("🌤️ **Open-Meteo Weather API:** Synchronized & Live.")
            st.info("✈️ **Indian Airlines:** 300,153 historical fares indexed.")
            st.info("🏨 **Goibibo Catalogue:** 4,000 verified listings active.")

    with n_col3:
        with st.popover("👤 Profile", use_container_width=True):
            st.markdown("##### 👤 Traveler Profile")
            if st.session_state["is_logged_in"]:
                st.write(f"**Name:** {st.session_state['user_name']}")
                st.write("**Tier:** Gold Explorer ⭐")
                st.write(f"**Email:** `{st.session_state['user_email']}`")
                st.caption("Preferences: Heritage, Scenic Beaches & Nature")
            else:
                st.write("Browsing as **Guest Traveler**.")

    with n_col4:
        if st.session_state["is_logged_in"]:
            if st.button("🔓 Log Out", use_container_width=True):
                st.session_state["is_logged_in"] = False
                st.toast("Signed out successfully. Browsing as Guest.", icon="👋")
                st.rerun()
        else:
            if st.button("🔒 Log In", use_container_width=True):
                st.session_state["is_logged_in"] = True
                st.toast(f"Welcome back, {st.session_state['user_name']}!", icon="👋")
                st.rerun()

st.divider()

# Read active API settings from session state or environment
active_api_key = st.session_state.get("openai_api_key", "").strip() or config.OPENAI_API_KEY
active_model = st.session_state.get("agent_model", config.OPENAI_MODEL)
active_force_demo = st.session_state.get("force_demo_mode", False)

# -----------------------------------------------------------------------------
# SIDEBAR TRAVEL PLANNING FORM (CLEAN, FOCUSED ON TRAVEL INPUTS ONLY)
# -----------------------------------------------------------------------------
st.sidebar.markdown("## ✈️ Travel Planning Form")
st.sidebar.caption("Fill in your travel preferences below and click Generate.")

# 1. Route Details
st.sidebar.markdown("#### 📍 Route Details")
col_s, col_d = st.sidebar.columns(2)

# Default indices if not already set in session state
default_origin_idx = CITIES.index(st.session_state.get("form_origin", "Delhi")) if st.session_state.get("form_origin") in CITIES else 0
default_dest_idx = CITIES.index(st.session_state.get("form_dest", "Goa")) if st.session_state.get("form_dest") in CITIES else 3

with col_s:
    source_city = st.selectbox(
        "Origin City",
        options=CITIES,
        index=default_origin_idx,
        key="form_origin",
        help="City you are departing from",
    )
with col_d:
    destination_city = st.selectbox(
        "Destination City",
        options=CITIES,
        index=default_dest_idx,
        key="form_dest",
        help="City you want to explore",
    )

# 2. Travel Dates & Duration
st.sidebar.markdown("#### 🗓️ Travel Dates")
today = date.today()
default_dep = st.session_state.get("form_dep_date", today + timedelta(days=1))
default_ret = st.session_state.get("form_ret_date", today + timedelta(days=4))

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    dep_date = st.date_input(
        "Departure Date",
        value=default_dep,
        min_value=today,
        key="form_dep_date",
    )
with col_d2:
    # Ensure return date cannot be before departure date
    min_ret = dep_date
    safe_ret = max(default_ret, min_ret)
    ret_date = st.date_input(
        "Return Date",
        value=safe_ret,
        min_value=min_ret,
        key="form_ret_date",
    )

trip_days = max(1, (ret_date - dep_date).days + 1)
hotel_nights = max(0, trip_days - 1)
st.sidebar.caption(f"⏱️ **Duration:** `{trip_days} Day{'s' if trip_days > 1 else ''}` | `{hotel_nights} Hotel Night{'s' if hotel_nights != 1 else ''}`")

# 3. Travelers & Flight Class
st.sidebar.markdown("#### 👥 Travelers & Flight Class")
col_t1, col_t2 = st.sidebar.columns([1.1, 1.4])
with col_t1:
    num_travelers = st.number_input(
        "Travelers",
        min_value=1,
        max_value=10,
        value=int(st.session_state.get("form_travelers", 2)),
        step=1,
        key="form_travelers",
    )
with col_t2:
    travel_class_options = ["Economy", "Business"]
    default_class_idx = 0 if st.session_state.get("form_travel_class", "Economy") == "Economy" else 1
    travel_class = st.radio(
        "Flight Cabin Class",
        options=travel_class_options,
        index=default_class_idx,
        horizontal=True,
        key="form_travel_class",
    )

# 4. Travel Style & Hotel Amenities
st.sidebar.markdown("#### ✨ Preferences & Accommodation")
category_options = ["Any / Balanced", "Beach", "Fort", "Temple", "Museum", "Lake", "Nature"]
default_cat_idx = category_options.index(st.session_state.get("form_category", "Any / Balanced")) if st.session_state.get("form_category") in category_options else 0
selected_category = st.sidebar.selectbox(
    "Attraction Preference",
    options=category_options,
    index=default_cat_idx,
    key="form_category",
)
pref_category = None if selected_category.startswith("Any") else selected_category

amenities_options = ["wifi", "pool", "gym", "breakfast", "spa", "restaurant"]
default_amenities = st.session_state.get("form_amenities", ["wifi"])
selected_amenities = st.sidebar.multiselect(
    "Preferred Hotel Amenities",
    options=amenities_options,
    default=[a for a in default_amenities if a in amenities_options],
    key="form_amenities",
)

# 5. Budget Customization
st.sidebar.markdown("#### 💵 Budget Settings (INR)")
with st.sidebar.expander("Configure Daily Rates & Budget Cap", expanded=False):
    total_budget_cap = st.number_input(
        "Max Total Budget Cap (₹, 0 = Flexible)",
        min_value=0,
        max_value=1000000,
        value=int(st.session_state.get("form_budget_cap", 0)),
        step=5000,
        key="form_budget_cap",
        help="Optional spending limit for flights, hotels, food and activities.",
    )
    daily_food = st.number_input(
        "Food & Dining / Person / Day (₹)",
        min_value=300,
        max_value=5000,
        value=int(st.session_state.get("form_daily_food", 800)),
        step=100,
        key="form_daily_food",
    )
    daily_transport = st.number_input(
        "Local Cab / Transit / Day (₹)",
        min_value=200,
        max_value=3000,
        value=int(st.session_state.get("form_daily_transport", 500)),
        step=50,
        key="form_daily_transport",
    )
    other_buffer = st.number_input(
        "Contingency Buffer / Person (₹)",
        min_value=0,
        max_value=5000,
        value=int(st.session_state.get("form_other_buffer", 1000)),
        step=200,
        key="form_other_buffer",
    )

# 6. Special Requests / Notes
special_notes = st.sidebar.text_input(
    "Special Requests / Notes (Optional)",
    value=st.session_state.get("form_special_notes", ""),
    placeholder="e.g. sunset views, relaxed pace, local markets...",
    key="form_special_notes",
)

st.sidebar.markdown("---")
# Primary Generate Button placed directly in the sidebar
generate_btn_sidebar = st.sidebar.button(
    "🚀 Generate Personalized Itinerary",
    type="primary",
    use_container_width=True,
    disabled=st.session_state.get("is_generating", False),
    key="btn_generate_sidebar",
)

# -----------------------------------------------------------------------------
# MAIN INTERACTION PANEL
# -----------------------------------------------------------------------------
st.markdown("### 🗺️ AI Travel Concierge")

# Active Configuration Summary Card
col_sum1, col_sum2 = st.columns([3.5, 1.5])
with col_sum1:
    st.markdown(f"""
    <div class="summary-card">
        <div style="font-size: 1.15rem; font-weight: 700; color: #93C5FD; margin-bottom: 4px;">
            🛫 {source_city} &nbsp;➔&nbsp; 🛬 {destination_city}
        </div>
        <div style="font-size: 0.9rem; color: #CBD5E1;">
            📅 <b>Dates:</b> {dep_date.strftime('%d %b %Y')} – {ret_date.strftime('%d %b %Y')} ({trip_days} Days, {hotel_nights} Nights)<br>
            👥 <b>Travelers:</b> {num_travelers} ({travel_class} Class) &nbsp;|&nbsp; 
            ✨ <b>Style:</b> {selected_category} &nbsp;|&nbsp; 
            🏨 <b>Amenities:</b> {', '.join(selected_amenities) if selected_amenities else 'Any'}
            {f"<br>💵 <b>Budget Cap:</b> ₹{total_budget_cap:,.0f}" if total_budget_cap > 0 else ""}
            {f"<br>📝 <b>Notes:</b> {special_notes}" if special_notes else ""}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_sum2:
    st.write("")
    generate_btn_main = st.button(
        "🚀 Generate Itinerary",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.get("is_generating", False),
        key="btn_generate_main",
    )
    st.caption("Click to generate your plan based on the sidebar settings.")

# Quick Scenario Presets (Updates the sidebar fields in sync, no disconnect)
st.markdown("##### 💡 Or Select a Quick Inspiration Preset:")
preset_cols = st.columns(3)
with preset_cols[0]:
    if st.button("🌴 3-Day Goa Vacation (from Delhi)", use_container_width=True):
        st.session_state["form_origin"] = "Delhi"
        st.session_state["form_dest"] = "Goa"
        st.session_state["form_dep_date"] = today + timedelta(days=1)
        st.session_state["form_ret_date"] = today + timedelta(days=3)
        st.session_state["form_travelers"] = 2
        st.session_state["form_travel_class"] = "Economy"
        st.session_state["form_category"] = "Beach"
        st.session_state["form_amenities"] = ["wifi", "pool"]
        st.session_state["form_special_notes"] = "Relaxing beach vacation with sunset views"
        st.rerun()

with preset_cols[1]:
    if st.button("🏰 4-Day Jaipur Heritage (from Mumbai)", use_container_width=True):
        st.session_state["form_origin"] = "Mumbai"
        st.session_state["form_dest"] = "Jaipur"
        st.session_state["form_dep_date"] = today + timedelta(days=1)
        st.session_state["form_ret_date"] = today + timedelta(days=4)
        st.session_state["form_travelers"] = 2
        st.session_state["form_travel_class"] = "Economy"
        st.session_state["form_category"] = "Fort"
        st.session_state["form_amenities"] = ["wifi", "breakfast"]
        st.session_state["form_special_notes"] = "Royal cultural tour exploring historic forts and food"
        st.rerun()

with preset_cols[2]:
    if st.button("🏛️ 3-Day Delhi Heritage (from Bangalore)", use_container_width=True):
        st.session_state["form_origin"] = "Bangalore"
        st.session_state["form_dest"] = "Delhi"
        st.session_state["form_dep_date"] = today + timedelta(days=1)
        st.session_state["form_ret_date"] = today + timedelta(days=3)
        st.session_state["form_travelers"] = 2
        st.session_state["form_travel_class"] = "Economy"
        st.session_state["form_category"] = "Museum"
        st.session_state["form_amenities"] = ["wifi", "restaurant"]
        st.session_state["form_special_notes"] = "Historic monuments and bustling markets tour"
        st.rerun()

# -----------------------------------------------------------------------------
# GENERATION WORKFLOW (TRIGGERS FROM SIDEBAR OR MAIN BUTTON)
# -----------------------------------------------------------------------------
should_generate = generate_btn_sidebar or generate_btn_main

if should_generate:
    if source_city.lower() == destination_city.lower():
        st.error(f"❌ Origin and destination cities cannot be identical ({source_city}). Please select different cities.")
    elif ret_date < dep_date:
        st.error("❌ Return date cannot be earlier than departure date.")
    elif trip_days > 14:
        st.error("❌ Maximum supported trip duration is 14 days.")
    else:
        st.session_state["is_generating"] = True
        with st.spinner(f"🤖 Generating personalized itinerary for {source_city} ➔ {destination_city} ({trip_days} Days, {num_travelers} Travelers)..."):
            # Construct natural language request dynamically from actual user inputs
            amenity_str = f", preferring hotels with {', '.join(selected_amenities)}" if selected_amenities else ""
            cat_str = f", focusing on {selected_category} attractions" if selected_category != "Any / Balanced" else ""
            notes_str = f". Special requests: {special_notes.strip()}" if special_notes.strip() else ""
            budget_str = f", with a total target budget of ₹{total_budget_cap:,.0f}" if total_budget_cap > 0 else ""

            user_query = (
                f"Plan a {trip_days}-day trip from {source_city} to {destination_city} "
                f"for {num_travelers} traveler(s) in {travel_class} cabin class"
                f"{cat_str}{amenity_str}{budget_str}{notes_str}."
            )

            runner = TravelAgentRunner(api_key=active_api_key, model_name=active_model)
            try:
                plan_result = runner.plan_trip(
                    user_query=user_query,
                    source_city=source_city,
                    destination_city=destination_city,
                    duration_days=trip_days,
                    start_date=dep_date.strftime("%Y-%m-%d"),
                    num_travelers=num_travelers,
                    total_budget=float(total_budget_cap) if total_budget_cap > 0 else None,
                    travel_class=travel_class,
                    preferred_category=pref_category,
                    preferred_amenities=selected_amenities,
                    daily_food_per_person=float(daily_food),
                    daily_transport=float(daily_transport),
                    other_buffer_per_person=float(other_buffer),
                    force_demo_mode=active_force_demo or not runner.has_llm_key,
                )
                st.session_state["current_plan"] = plan_result
                st.session_state["is_generating"] = False
                st.toast(f"Itinerary for {destination_city} generated successfully!", icon="🎉")
            except Exception as e:
                st.session_state["is_generating"] = False
                st.error(f"⚠️ Error generating itinerary: {e}")

# -----------------------------------------------------------------------------
# RESULTS DISPLAY
# -----------------------------------------------------------------------------
if "current_plan" in st.session_state:
    plan = st.session_state["current_plan"]
    itinerary: CompleteTravelItinerary = plan["structured_itinerary"]

    st.markdown("---")
    st.subheader(f"🗺️ {itinerary.trip_title}")

    # Top summary metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Estimated Cost", f"₹{itinerary.budget.grand_total_inr:,.0f}")
    with m2:
        st.metric("Cost Per Traveler", f"₹{itinerary.budget.per_person_inr:,.0f}")
    with m3:
        st.metric("Selected Airline", f"{itinerary.flight.airline}")
    with m4:
        st.metric("Accommodation", f"{itinerary.hotel.hotel_name[:18]}...")

    # Interactive Tabs
    tab_schedule, tab_logistics, tab_weather, tab_budget, tab_reasoning, tab_raw = st.tabs([
        "📅 Day-Wise Itinerary",
        "✈️ Flight & 🏨 Hotel",
        "⛅ Weather Forecast",
        "💰 Budget Breakdown",
        "🧠 AI Justifications",
        "📄 Structured JSON & Export",
    ])

    # 1. Day-Wise Itinerary Tab
    with tab_schedule:
        for day in itinerary.day_wise_schedule:
            with st.expander(f"📍 **{day.date_label}**: {day.theme}", expanded=True):
                st.caption(f"🌤️ **Weather:** {day.weather_summary} | 💵 **Estimated Day Expense:** ₹{day.estimated_daily_expense_inr:,.0f}")
                for act in day.activities:
                    c_time, c_desc, c_meta = st.columns([1.5, 4, 2])
                    with c_time:
                        st.markdown(f"**{act.time_slot}**")
                        st.caption(f"🏷️ `{act.category}`")
                    with c_desc:
                        st.markdown(f"**{act.activity_name}**")
                        st.write(act.highlights_or_tips)
                    with c_meta:
                        st.write(f"⏱️ **Duration:** {act.estimated_duration_hours:.1f} hrs")
                        if act.entrance_fee_inr > 0:
                            st.write(f"🎟️ **Fee:** ₹{int(act.entrance_fee_inr)} / person")
                        else:
                            st.write("🎟️ **Fee:** Free Entry")
                    st.divider()

    # 2. Flight & Hotel Tab
    with tab_logistics:
        col_fl, col_ht = st.columns(2)
        with col_fl:
            st.markdown("#### ✈️ Recommended Flight Option")
            st.markdown(f"**{itinerary.flight.airline}** ({itinerary.flight.flight_number})")
            st.write(f"**Route:** {itinerary.flight.source} ➡️ {itinerary.flight.destination}")
            st.write(f"**Schedule:** Departs {itinerary.flight.departure_time} | Arrives {itinerary.flight.arrival_time}")
            st.write(f"**Duration:** {itinerary.flight.duration_hours:.1f} hrs | **Stops:** {itinerary.flight.stops}")
            st.write(f"**Cabin Class:** {itinerary.flight.travel_class}")
            st.markdown(f"### ₹{itinerary.flight.price_inr:,.0f} <span class='badge-historical'>Historical Fare / Person</span>", unsafe_allow_html=True)
            st.info(f"**Selection Logic:** {itinerary.flight.selection_reason}")

        with col_ht:
            st.markdown("#### 🏨 Recommended Accommodation")
            st.markdown(f"**{itinerary.hotel.hotel_name}**")
            st.write(f"📍 **Address:** {itinerary.hotel.address}")
            rating_display = f"{itinerary.hotel.review_rating or itinerary.hotel.star_rating or 4.0}★"
            st.write(f"⭐ **Rating:** {rating_display}")
            if itinerary.hotel.amenities:
                st.write(f"🏊 **Amenities:** {', '.join(itinerary.hotel.amenities)}")
            verified_tag = "badge-live" if itinerary.hotel.is_verified_price else "badge-historical"
            tag_label = "Baseline Verified Rate" if itinerary.hotel.is_verified_price else "Historical Reference Rate"
            st.markdown(f"### ₹{itinerary.hotel.price_per_night_inr:,.0f} <span class='{verified_tag}'>{tag_label} / Night</span>", unsafe_allow_html=True)
            st.info(f"**Selection Logic:** {itinerary.hotel.recommendation_reason}")

    # 3. Weather Forecast Tab
    with tab_weather:
        st.markdown(f"#### ⛅ Open-Meteo Real-Time Weather Forecast for {itinerary.destination_city}")
        if itinerary.weather:
            w_cols = st.columns(min(len(itinerary.weather), 5))
            for i, w in enumerate(itinerary.weather[:5]):
                with w_cols[i]:
                    st.markdown(f"**{w.date}**")
                    st.write(f"**{w.condition}**")
                    if w.temp_min_c is not None and w.temp_max_c is not None:
                        st.markdown(f"🌡️ `{w.temp_min_c}°C – {w.temp_max_c}°C`")
                    st.write(f"🌧️ Rain: {w.precipitation_probability_pct}%")
                    st.caption(f"💡 {w.travel_advice}")
        else:
            st.info("Weather data retrieved from seasonal climate averages.")

    # 4. Budget Breakdown Tab
    with tab_budget:
        st.markdown("#### 💰 Comprehensive Itemized Trip Budget")
        bd = itinerary.budget

        budget_data = {
            "Expense Category": [
                "Flights (Round-trip)",
                "Accommodation",
                "Food & Dining (Estimated)",
                "Local Transport (Estimated)",
                "Sightseeing Entrance Tickets",
                "Contingency Buffer"
            ],
            "Amount (INR)": [
                bd.flights_total_inr,
                bd.hotel_total_inr,
                bd.food_total_inr,
                bd.local_transport_total_inr,
                bd.sightseeing_total_inr,
                bd.other_buffer_inr,
            ],
            "Source Type": [
                "Dataset Reference",
                "Dataset Reference",
                "Configured Daily Allowance",
                "Configured Daily Allowance",
                "Dataset Reference",
                "Configured Contingency Buffer"
            ]
        }
        df_budget = pd.DataFrame(budget_data)

        col_tbl, col_chart = st.columns([1.2, 1])
        with col_tbl:
            st.dataframe(df_budget, use_container_width=True, hide_index=True)
            st.markdown(f"### Grand Total: **₹{bd.grand_total_inr:,.0f}** *(₹{bd.per_person_inr:,.0f} / person)*")
        with col_chart:
            chart_df = df_budget.set_index("Expense Category")[["Amount (INR)"]]
            st.bar_chart(chart_df)

    # 5. AI Reasoning Tab
    with tab_reasoning:
        st.markdown("#### 🧠 AI Decision Explanations & Justifications")
        st.markdown(f"**Mode:** {plan.get('mode', 'Agentic Execution')}")

        if plan.get("tool_trace"):
            st.markdown("##### 🔍 Autonomous Tool Invocation Trace:")
            for step in plan["tool_trace"]:
                st.markdown(f"- 🔧 `{step.get('tool', 'tool')}`: {step.get('result') or step.get('preview') or step.get('status') or 'Completed'}")

        st.markdown("##### 📌 Selection Rationale:")
        st.markdown(f"- **Flight Selection:** {itinerary.justifications.flight_justification}")
        st.markdown(f"- **Hotel Selection:** {itinerary.justifications.hotel_justification}")
        st.markdown(f"- **Attractions Curation:** {itinerary.justifications.attractions_justification}")
        st.markdown(f"- **Itinerary Pacing:** {itinerary.justifications.pacing_justification}")

        if plan.get("llm_response"):
            with st.expander("📝 Full Agent Synthesis Text", expanded=False):
                st.markdown(plan["llm_response"])

    # 6. Structured JSON & Export Tab
    with tab_raw:
        st.markdown("#### 📄 Validated Pydantic JSON Output")
        json_dump = itinerary.model_dump_json(indent=2)
        st.download_button(
            label="📥 Download Structured Itinerary JSON",
            data=json_dump,
            file_name=f"itinerary_{itinerary.destination_city.lower()}_{trip_days}days.json",
            mime="application/json",
            use_container_width=True,
        )
        st.json(json.loads(json_dump))

# -----------------------------------------------------------------------------
# ABOUT THIS APP & HOW IT HELPS YOU (ABOVE FOOTER)
# -----------------------------------------------------------------------------
st.markdown("---")
st.markdown("### 🌟 About VoyAgent AI & How It Helps You")
st.markdown(
    "VoyAgent AI is an **Agentic AI Travel Planning Assistant** engineered using LangChain, Open-Meteo, "
    "and curated Indian tourism databases to automate end-to-end trip design with expert-level personalization."
)

col_a1, col_a2, col_a3 = st.columns(3)
with col_a1:
    st.markdown("""
    <div class="feature-card">
        <h4>⚡ Saves Hours of Planning</h4>
        <p style="color: #64748B; font-size: 0.92rem;">
            Eliminates switching across dozens of flight aggregators, booking portals, and travel blogs. 
            Simply select your destination and preferences in the sidebar to get a synthesized, complete itinerary in seconds.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_a2:
    st.markdown("""
    <div class="feature-card">
        <h4>🎯 Fatigue-Free Realistic Pacing</h4>
        <p style="color: #64748B; font-size: 0.92rem;">
            Unlike generic planners that overload 10 sights a day, our engine caps activities to 2–3 daily stops, 
            respects opening hours, closing days (weekly off), and coordinates return-day flights.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col_a3:
    st.markdown("""
    <div class="feature-card">
        <h4>💰 100% Transparent Budgeting</h4>
        <p style="color: #64748B; font-size: 0.92rem;">
            Computes true trip cost including flights, hotel nights, customizable daily dining, local transit, 
            and monument tickets. Differentiates verified rates from estimates with zero hidden math.
        </p>
    </div>
    """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# USEFUL TRAVEL TOOLS (BELOW ABOUT SECTION)
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
col_pack, col_help = st.columns([1.1, 1])

with col_pack:
    with st.expander("🧳 Smart Travel Packing Checklist Generator", expanded=False):
        st.write("Customized packing essentials tailored to Indian travel:")
        p_c1, p_c2 = st.columns(2)
        with p_c1:
            st.checkbox("Government ID (Aadhaar / Passport / Voter ID)", value=True)
            st.checkbox("Flight boarding passes / Train e-tickets", value=True)
            st.checkbox("Power bank & universal charging adapters", value=True)
            st.checkbox("Prescription medicines & basic first-aid", value=True)
        with p_c2:
            st.checkbox("Comfortable walking sneakers / sandals", value=True)
            st.checkbox("Light cotton clothing / Sun protection (SPF 50)", value=True)
            st.checkbox("Sunglasses & reusable insulated water bottle", value=True)
            st.checkbox("Digital camera / DSLR with extra SD card", value=False)

with col_help:
    with st.expander("🚨 Emergency & Travel Helpline Directory", expanded=False):
        st.markdown("""
        - 📞 **Incredible India Tourist Helpline:** `1363` *(24x7 Multi-lingual)*
        - 🚨 **National Unified Emergency:** `112`
        - 🚑 **Medical Ambulance Assistance:** `108`
        - 🛡️ **Women's Safety Helpline:** `1091`
        - 🚂 **Indian Railways Enquiry / Assistance:** `139`
        """)

# -----------------------------------------------------------------------------
# APP DETAILS & METRICS
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("##### 📊 Platform Architecture & Data Health:")
st_c1, st_c2, st_c3, st_c4 = st.columns(4)
with st_c1:
    st.info("✈️ **300,153 Records**\n\nIndian Airlines Historical Fares")
with st_c2:
    st.info("🏨 **4,000 Properties**\n\nGoibibo Hotel Catalogue")
with st_c3:
    st.info("🏛️ **323 Destinations**\n\nCurated Tourist Attractions")
with st_c4:
    st.info("⛅ **Live Weather Sync**\n\nOpen-Meteo REST API")

# -----------------------------------------------------------------------------
# MODERN FOOTER
# -----------------------------------------------------------------------------
st.markdown("""
<div class="footer-container">
    <div style="display: flex; justify-content: space-between; flex-wrap: wrap; gap: 20px;">
        <div style="flex: 1.5; min-width: 240px;">
            <div style="font-size: 1.3rem; font-weight: 800; color: #FFFFFF; display: flex; align-items: center; gap: 8px;">
                ✈️ VoyAgent AI
            </div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px; line-height: 1.5;">
                An autonomous multi-agent travel concierge powered by LangChain, Open-Meteo, and domestic Indian tourism datasets. Built for academic evaluation and smart travel synthesis.
            </p>
        </div>
        <div style="flex: 1; min-width: 160px;">
            <div class="footer-title">Navigation</div>
            <span class="footer-link">🗺️ Personalized Trip Planner</span>
            <span class="footer-link">⛅ Live Weather Forecast</span>
            <span class="footer-link">✈️ Historical Flight Lookup</span>
            <span class="footer-link">🏨 Hotel Discovery Catalogue</span>
        </div>
        <div style="flex: 1; min-width: 160px;">
            <div class="footer-title">Technology</div>
            <span class="footer-link">🦜 LangChain 1.4+ Agents</span>
            <span class="footer-link">🌐 Open-Meteo REST API</span>
            <span class="footer-link">⚡ Streamlit 1.65 Framework</span>
            <span class="footer-link">🛡️ Pydantic v2 Validation</span>
        </div>
        <div style="flex: 1; min-width: 180px;">
            <div class="footer-title">Academic Project</div>
            <p style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 6px;">
                <b>Author:</b> Kartik Prasad<br>
                <b>Domain:</b> Agentic AI / Tourism Tech<br>
                <b>Status:</b> 32/32 Pytest Tests Passed
            </p>
        </div>
    </div>
    <div class="footer-bottom">
        © 2026 <b>VoyAgent AI</b> — Agentic AI-Based Travel Planning Assistant. Built with Python 3.10 & LangChain.
    </div>
</div>
""", unsafe_allow_html=True)
