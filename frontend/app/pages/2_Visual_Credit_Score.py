import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
import numpy as np

# Ensure utils is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "frontend", "app"))
from utils.data_loader import load_data, load_scorer

st.set_page_config(page_title="Visual Credit Score | TerraTrust", layout="wide")

# PREMIUM CUSTOM CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif !important;
    }
    .stApp {
        background: radial-gradient(circle at top left, #0e1628, #05080f);
        color: #e2e8f0;
    }
    h1, h2, h3 {
        color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.5px;
    }
    /* Cards and Glassmorphism */
    div[data-testid="stMetricValue"] {
        font-size: 2.2rem !important;
        background: -webkit-linear-gradient(45deg, #00FFAA, #00B8FF);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }
    /* Buttons with hover effects */
    .stButton > button {
        background: linear-gradient(90deg, #00FFAA 0%, #00B8FF 100%) !important;
        color: #05080f !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(0, 255, 170, 0.3) !important;
    }
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(0, 255, 170, 0.5) !important;
        filter: brightness(1.1);
    }
    /* Selectboxes and inputs */
    div[data-baseweb="select"] > div {
        background-color: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
        border-radius: 8px !important;
        color: white !important;
    }
    div[data-baseweb="select"]:hover > div {
        border-color: #00FFAA !important;
    }
    /* Expander / Containers */
    .streamlit-expanderHeader {
        background-color: rgba(255, 255, 255, 0.03) !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

df = load_data()
scorer = load_scorer()

st.title("📊 Micro Evaluation – Visual Credit Scorecard")
st.markdown("Operational dashboard combining ML predictions with transparent visual evidence to automate loan decisions.")

if df.empty:
    st.error("Data not loaded correctly. Please run the data pipeline.")
    st.stop()

# ---------------------------------------------------------
# Sidebar: Application Input & Search
# ---------------------------------------------------------
st.sidebar.title("Application Details")
districts = sorted(df['district'].dropna().unique())
selected_dist = st.sidebar.selectbox("Target District", districts)

taluks = sorted(df[df['district'] == selected_dist]['taluk'].dropna().unique())
selected_taluk = st.sidebar.selectbox("Target Taluk", taluks)

taluk_farms = df[(df['district'] == selected_dist) & (df['taluk'] == selected_taluk)]

if taluk_farms.empty:
    st.sidebar.warning("No applications found in this taluk.")
    st.stop()

selected_farm_id = st.sidebar.selectbox("Select Point ID", taluk_farms['point_id_yr'].unique())
farm_data = taluk_farms[taluk_farms['point_id_yr'] == selected_farm_id].iloc[0].to_dict()

# Optional inputs (for UI completeness as requested)
declared_crop = st.sidebar.text_input("Declared Crop Type", value=farm_data.get('declared_crop', 'Cotton'))
loan_amount = st.sidebar.number_input("Requested Loan Amount (₹)", min_value=10000, max_value=1000000, value=50000, step=10000)

# Run the scoring engine
result = scorer.generate_credit_score(farm_data)

# Extract scores
score = result['heuristic_credit_score']
risk = result['risk_category']
comps = result['components']
repay_prob = result['repayment_probability']

# ---------------------------------------------------------
# Main Dashboard
# ---------------------------------------------------------
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Risk Score for Loan Approval")
    # Modern Thin Gauge Chart for Score
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Credit Score (0-100)", 'font': {'color': 'white', 'size': 18}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#00FFAA"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 0,
            'steps': [
                {'range': [0, 100], 'color': "rgba(255,255,255,0.1)"}
            ],
        }
    ))
    fig.update_layout(
        height=250, 
        margin=dict(l=20, r=20, t=50, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    st.plotly_chart(fig, width='stretch')
    
    st.markdown(f"<h3 style='text-align: center;'>Risk Classification: <b>{risk}</b></h3>", unsafe_allow_html=True)
    st.markdown(f"<p style='text-align: center;'>{result['recommendation']}</p>", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------
# Supporting Evidence & Explainability
# ---------------------------------------------------------
col_map, col_trend = st.columns(2)

lat = farm_data.get('latitude', None)
lon = farm_data.get('longitude', None)

with col_map:
    st.subheader("2. Supporting Geospatial Evidence (Maps + Imagery)")
    if pd.notna(lat) and pd.notna(lon):
        # Bound the map strictly to Karnataka (Approx bounds: South, West, North, East)
        karnataka_bounds = [[11.5, 74.0], [18.5, 78.5]]
        m = folium.Map(
            location=[lat, lon], 
            zoom_start=14, 
            tiles=None,
            max_bounds=True,
            min_lat=11.0, max_lat=19.0, min_lon=73.5, max_lon=79.0
        )
        # Add Google Satellite tiles
        folium.TileLayer(
            tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
            attr='Google',
            name='Google Satellite',
            overlay=False,
            control=True
        ).add_to(m)
        m.fit_bounds(karnataka_bounds)
        
        # Add Mask to cover the rest of the world (blackout)
        mask_path = os.path.join(PROJECT_ROOT, "data", "processed", "inverted_mask.geojson")
        if os.path.exists(mask_path):
            folium.GeoJson(
                mask_path,
                style_function=lambda x: {
                    'fillColor': '#05080f',
                    'color': '#05080f',
                    'weight': 1,
                    'fillOpacity': 1.0
                }
            ).add_to(m)
        
        # Add a more premium marker
        icon_color = "green" if risk == "Low" else ("orange" if risk == "Moderate" else "red")
        folium.Marker(
            location=[lat, lon],
            popup=f"<b>Farm ID:</b> {selected_farm_id}<br><b>Risk:</b> {risk}",
            icon=folium.Icon(color=icon_color, icon="info-sign")
        ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        folium_static(m, width=500, height=400)
    else:
        st.warning("Coordinates not available for this farm.")

with col_trend:
    st.subheader("3. Historical Trends for Decision Support")
    st.markdown("Time-series satellite data for yield consistency.")
    
    # Load raw NDVI from satellite file if possible
    sat_path = os.path.join(PROJECT_ROOT, "data", "satellite", "karnataka_ndvi_real.csv")
         
    if os.path.exists(sat_path) and pd.notna(lat) and pd.notna(lon):
        try:
            sat_df = pd.read_csv(sat_path)
            # Find closest taluk/point case-insensitive
            sat_df['taluk_clean'] = sat_df['taluk'].str.lower().str.strip()
            target_taluk = str(farm_data.get('taluk', '')).lower().strip()
            closest_farms = sat_df[sat_df['taluk_clean'] == target_taluk]
            
            if not closest_farms.empty:
                # Group by year
                trend_df = closest_farms.groupby('year')['ndvi_mean'].mean().reset_index()
                fig2 = px.line(trend_df, x='year', y='ndvi_mean', markers=True, 
                               title="Historical NDVI Mean (Satellite)",
                               line_shape="spline", render_mode="svg")
                fig2.update_traces(line_color="#00FFAA", marker=dict(size=8, color="white"))
                fig2.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font=dict(color='white')
                )
                st.plotly_chart(fig2, width='stretch')
            else:
                st.info("No localized historical satellite data found.")
        except Exception as e:
            st.error("Could not process historical trends.")
    else:
         st.info("Satellite time-series data file not available.")

st.divider()

# ---------------------------------------------------------
# Local Explainability (Heuristic Fallback)
# ---------------------------------------------------------
with st.expander("🔍 View Local Explainability (Why this score?)", expanded=False):
    st.markdown("**(Fallback Mode: Due to system Application Control policies blocking native SHAP libraries, this is a simulated heuristic breakdown of the risk factors).**")
    
    # Simulate waterfall breakdown
    base_score = 50
    final = score
    diff = final - base_score
    
    # Create arbitrary but logical attributions
    health_impact = (comps['pred_crop_health'] - 0.5) * 40
    soil_impact = (comps['pred_soil_q'] - 1.0) * 20
    water_impact = (50 - comps['pred_water_depth']) * 0.5
    
    # Normalize to match actual diff
    total_sim = health_impact + soil_impact + water_impact
    if total_sim != 0:
        health_impact = health_impact / total_sim * diff
        soil_impact = soil_impact / total_sim * diff
        water_impact = water_impact / total_sim * diff
    else:
        health_impact = diff / 3
        soil_impact = diff / 3
        water_impact = diff / 3

    fig_waterfall = go.Figure(go.Waterfall(
        name="Credit Score",
        orientation="v",
        measure=["absolute", "relative", "relative", "relative", "total"],
        x=["Base Score", "Crop Health", "Soil Quality", "Water Depth", "Final Score"],
        textposition="outside",
        y=[base_score, health_impact, soil_impact, water_impact, final],
        connector={"line": {"color": "rgb(63, 63, 63)"}},
    ))
    fig_waterfall.update_layout(
        title="Simulated Feature Impact on Credit Score",
        showlegend=False,
        height=400
    )
    st.plotly_chart(fig_waterfall, width='stretch')
