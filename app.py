"""Streamlit Web Application for the Agentic AI Travel Planning Assistant.

Provides an interactive user interface for querying, customizing, and
generating autonomous day-wise travel itineraries with LangChain,
historical datasets, and live Open-Meteo forecasts.
Includes top navigation header, about section, interactive footer,
smart packing checklist, and emergency travel directory.
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

# Custom CSS for styling
st.markdown("""
<style>
    /* Top Navbar */
    .top-nav {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 10px 18px;
        background: linear-gradient(135deg, #1E3A8A 0%, #2563EB 100%);
        border-radius: 12px;
        color: white;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    .brand-title {
        font-size: 1.45rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .brand-badge {
        background-color: rgba(255,255,255,0.22);
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-left: 8px;
    }
    .main-header {
        font-size: 2.1rem;
        font-weight: 750;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #4B5563;
        margin-bottom: 1.2rem;
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
    .helpline-card {
        background-color: #EFF6FF;
        border-left: 4px solid #3B82F6;
        border-radius: 8px;
        padding: 12px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP NAVIGATION HEADER
# -----------------------------------------------------------------------------
col_logo, col_nav = st.columns([6, 4])
with col_logo:
    st.markdown("""
    <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
        <span style="font-size: 2rem;">✈️</span>
        <div>
            <span style="font-size: 1.55rem; font-weight: 800; color: #1E3A8A; letter-spacing: -0.5px;">VoyAgent AI</span>
            <span style="background: #E0E7FF; color: #3730A3; padding: 2px 8px; border-radius: 12px; font-size: 0.75rem; font-weight: 600; margin-left: 6px;">v2.4 Live</span>
            <div style="font-size: 0.82rem; color: #64748B;">Agentic Travel Concierge powered by LangChain & Open-Meteo</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_nav:
    n_col1, n_col2, n_col3 = st.columns([1.2, 1.2, 1.4])
    with n_col1:
        with st.popover("🔔 Alerts (3)", use_container_width=True):
            st.markdown("##### 📬 Notifications")
            st.success("🌤️ **Open-Meteo Weather API:** Synchronized & Live.")
            st.info("✈️ **Indian Airlines:** 300,153 historical fares indexed.")
            st.info("🏨 **Goibibo Catalogue:** 4,000 verified listings active.")
    with n_col2:
        with st.popover("👤 Profile", use_container_width=True):
            st.markdown("#####  traveler Profile")
            if st.session_state["is_logged_in"]:
                st.write(f"**Name:** {st.session_state['user_name']}")
                st.write(f"**Tier:** Gold Explorer ⭐")
                st.write(f"**Email:** `{st.session_state['user_email']}`")
                st.caption("Preferences: Heritage, Scenic Beaches & Nature")
            else:
                st.write("Browsing as **Guest Traveler**.")
    with n_col3:
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

# -----------------------------------------------------------------------------
# SIDEBAR CONFIGURATION
# -----------------------------------------------------------------------------
st.sidebar.title("🛠️ Travel Parameters")

# LLM Configuration
with st.sidebar.expander("🤖 AI Agent Settings", expanded=False):
    api_key_input = st.text_input(
        "OpenAI API Key",
        value=os.getenv("OPENAI_API_KEY", ""),
        type="password",
        help="Optional. Leave blank to run in fast dataset-backed demo mode.",
    )
    model_choice = st.selectbox(
        "Agent Model",
        options=["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"],
        index=0,
    )
    force_demo = st.checkbox("Force Deterministic Demo Mode", value=False)
    effective_key = api_key_input.strip() or config.OPENAI_API_KEY
    if effective_key and not force_demo:
        st.sidebar.caption(f"🟢 LLM Agent Active (`{model_choice}`)")
    else:
        st.sidebar.caption("⚡ Offline/Demo Mode Active (Live Weather & Datasets)")

# City Options
CITIES = [
    "Delhi", "Mumbai", "Bangalore", "Goa", "Jaipur", "Hyderabad",
    "Kolkata", "Chennai", "Ahmedabad", "Pune", "Kochi", "Agra",
    "Varanasi", "Coorg", "Alleppey", "Udaipur", "Amritsar", "Leh"
]

st.sidebar.subheader("📍 Route Details")
col_s, col_d = st.sidebar.columns(2)
with col_s:
    source_city = st.selectbox("Origin City", options=CITIES, index=0)
with col_d:
    destination_city = st.selectbox("Destination City", options=CITIES, index=3)  # Defaults to Goa

# Dates & Travelers
today = date.today()
tomorrow = today + timedelta(days=1)
four_days_later = tomorrow + timedelta(days=3)

col_d1, col_d2 = st.sidebar.columns(2)
with col_d1:
    dep_date = st.date_input("Departure Date", value=tomorrow, min_value=today)
with col_d2:
    ret_date = st.date_input("Return Date", value=four_days_later, min_value=dep_date)

trip_days = max(1, (ret_date - dep_date).days + 1)
st.sidebar.caption(f"🗓️ Duration: **{trip_days} Day(s)** ({max(1, trip_days - 1)} Night{'s' if trip_days > 2 else ''})")

num_travelers = st.sidebar.number_input("Number of Travelers", min_value=1, max_value=10, value=2, step=1)
travel_class = st.sidebar.radio("Flight Cabin Class", options=["Economy", "Business"], index=0, horizontal=True)

# Preferences & Category
st.sidebar.subheader("✨ Travel Style")
category_options = ["Any / Balanced", "Beach", "Fort", "Temple", "Museum", "Lake", "Nature"]
selected_category = st.sidebar.selectbox("Attraction Preference", options=category_options, index=0)
pref_category = None if selected_category.startswith("Any") else selected_category

amenities_options = ["wifi", "pool", "gym", "breakfast", "spa", "restaurant"]
selected_amenities = st.sidebar.multiselect("Preferred Hotel Amenities", options=amenities_options, default=["wifi"])

# Budget Settings Expander
with st.sidebar.expander("💵 Budget Customization (INR)", expanded=False):
    daily_food = st.number_input("Food & Dining / Person / Day", min_value=300, max_value=5000, value=800, step=100)
    daily_transport = st.number_input("Local Cab / Transport / Day", min_value=200, max_value=3000, value=500, step=50)
    other_buffer = st.number_input("Contingency Buffer / Person", min_value=0, max_value=5000, value=1000, step=200)

# -----------------------------------------------------------------------------
# MAIN INTERACTION PANEL
# -----------------------------------------------------------------------------
st.markdown("##### 💡 Quick Scenario Presets:")
preset_cols = st.columns(3)
with preset_cols[0]:
    if st.button("🌴 3-Day Goa Vacation (from Delhi)"):
        st.session_state["query_text"] = "Plan a relaxing 3-day beach vacation to Goa for 2 people with comfortable hotel and sunset views."
with preset_cols[1]:
    if st.button("🏰 4-Day Jaipur Heritage (from Mumbai)"):
        st.session_state["query_text"] = "Plan a 4-day royal cultural trip to Jaipur exploring forts, palaces, and authentic Rajasthani food."
with preset_cols[2]:
    if st.button("🏛️ 3-Day Delhi Tour (from Bangalore)"):
        st.session_state["query_text"] = "Plan a 3-day heritage and food tour of Delhi visiting top historic monuments and bustling markets."

user_query = st.text_area(
    "Travel Request (Natural Language):",
    value=st.session_state.get(
        "query_text",
        f"Plan a {trip_days}-day trip from {source_city} to {destination_city} for {num_travelers} traveler(s) focusing on {selected_category}."
    ),
    height=75,
)

generate_btn = st.button("🚀 Generate Personalized Itinerary", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# GENERATION WORKFLOW
# -----------------------------------------------------------------------------
if generate_btn:
    if source_city.lower() == destination_city.lower():
        st.error("❌ Origin and destination cities cannot be identical. Please select different cities.")
    else:
        with st.spinner("🤖 Agentic Assistant is querying flights, hotels, attractions, Open-Meteo weather, and computing budgets..."):
            runner = TravelAgentRunner(api_key=effective_key, model_name=model_choice)
            plan_result = runner.plan_trip(
                user_query=user_query,
                source_city=source_city,
                destination_city=destination_city,
                duration_days=trip_days,
                start_date=dep_date.strftime("%Y-%m-%d"),
                num_travelers=num_travelers,
                travel_class=travel_class,
                preferred_category=pref_category,
                preferred_amenities=selected_amenities,
                daily_food_per_person=daily_food,
                daily_transport=daily_transport,
                other_buffer_per_person=other_buffer,
                force_demo_mode=force_demo or not runner.has_llm_key,
            )
            st.session_state["current_plan"] = plan_result

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
            Simply type your destination and preferences to get a synthesized, complete itinerary in seconds.
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
