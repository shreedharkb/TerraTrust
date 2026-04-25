import streamlit as st
import pandas as pd
import folium
from streamlit_folium import folium_static
import plotly.express as px
import sys
import os

# Ensure utils is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils.data_loader import load_data, load_geojson, load_metrics

st.set_page_config(page_title="Regional Intelligence | TerraTrust", layout="wide")

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
</style>
""", unsafe_allow_html=True)

# Load data
df = load_data()
geojson = load_geojson()
metrics = load_metrics()

st.title("🌟 Welcome to TerraTrust")
st.markdown("""
<div style='background: rgba(0, 255, 170, 0.05); border: 1px solid rgba(0, 255, 170, 0.2); padding: 1.5rem; border-radius: 10px; margin-bottom: 2rem;'>
    <h3 style='margin-top: 0; color: #00FFAA !important;'>Project Overview</h3>
    <p style='font-size: 1.1rem; line-height: 1.6; color: #e2e8f0; margin-bottom: 0;'>
        TerraTrust is a state-of-the-art agricultural credit intelligence system. We eliminate the need for manual, subjective loan evaluations by combining <b>high-resolution satellite imagery (Sentinel-2)</b>, <b>NASA climate data</b>, and <b>KGIS soil profiles</b>. Using advanced Machine Learning, TerraTrust generates automated, transparent, and highly accurate risk assessments for agricultural loans across the entire state of Karnataka.
    </p>
</div>
""", unsafe_allow_html=True)

st.header("🌍 Macro Intelligence – Regional Risk & Data Overview")
st.markdown("Assess regional risk and understand the underlying data driving the credit models across Karnataka.")

if df.empty or not geojson:
    st.error("Data or GeoJSON not loaded correctly. Please run the data pipeline.")
    st.stop()

# Aggregate data by Taluk for the map
taluk_agg = df.groupby('taluk').agg({
    'ndvi_annual_mean': 'mean',
    'avg_root_zone_wetness': 'mean',
    'organic_carbon_dg_per_kg': 'mean'
}).reset_index()

# ---------------------------------------------------------
# Sidebar: Model Architecture & Performance
# ---------------------------------------------------------
st.sidebar.title("Model Performance")
if metrics:
    st.sidebar.markdown("### Accuracy Metrics")
    m_a = metrics.get('Model_A_CropHealth', {}).get('spatial', {})
    st.sidebar.metric("Crop Health (XGBoost)", f"{m_a.get('test_accuracy', 0)*100:.1f}%")
    
    m_b = metrics.get('Model_B_Soil', {}).get('spatial', {})
    st.sidebar.metric("Soil Quality (RF)", f"{m_b.get('test_accuracy', 0)*100:.1f}%")
    
    m_c = metrics.get('Model_C_Water', {}).get('spatial', {})
    st.sidebar.metric("Water Depth (GradBoost R²)", f"{m_c.get('test_r2', 0)*100:.1f}%")
    
    m_d = metrics.get('Model_D_Credit_Risk', {}).get('spatial', {})
    st.sidebar.metric("Credit Risk (XGBoost)", f"{m_d.get('test_accuracy', 0)*100:.1f}%")

# ---------------------------------------------------------
# Main Content: Map & Feature Importance
# ---------------------------------------------------------
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Interactive KGIS Layer Map")
    st.markdown("*Note: Taluks without color represent regions without training data in the current `karnataka_master_dataset.csv`.*")
    layer_option = st.selectbox("Select Spatial Layer", [
        "Crop Health (NDVI)", 
        "Groundwater/Root Zone Wetness", 
        "Soil Suitability (Organic Carbon)"
    ])
    
    # Map selection to dataframe column
    if layer_option == "Crop Health (NDVI)":
        value_col = 'ndvi_annual_mean'
        color_scale = 'YlGn'
        legend_name = 'Mean NDVI'
    elif layer_option == "Groundwater/Root Zone Wetness":
        value_col = 'avg_root_zone_wetness'
        color_scale = 'Blues'
        legend_name = 'Mean Root Zone Wetness'
    else:
        value_col = 'organic_carbon_dg_per_kg'
        color_scale = 'Oranges'
        legend_name = 'Mean Organic Carbon (dg/kg)'

    # Create Base Map focused on Karnataka strictly
    karnataka_bounds = [[11.5, 74.0], [18.5, 78.5]]
    m = folium.Map(
        location=[15.3173, 75.7139], 
        zoom_start=6, 
        tiles=None,  # Do not use default tiles
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
    mask_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "processed", "inverted_mask.geojson")
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

    # Add Choropleth
    folium.Choropleth(
        geo_data=geojson,
        name='choropleth',
        data=taluk_agg,
        columns=['taluk', value_col],
        key_on='feature.properties.KGISTalukN',
        fill_color=color_scale,
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name,
        nan_fill_color='white'
    ).add_to(m)

    folium_static(m, width=700, height=500)

with col2:
    st.subheader("Global Feature Importance")
    st.markdown("Variables driving risk assessments state-wide.")
    
    if metrics and 'Model_D_Credit_Risk' in metrics:
        importances = metrics['Model_D_Credit_Risk'].get('feature_importances', {})
        if importances:
            imp_df = pd.DataFrame(list(importances.items()), columns=['Feature', 'Importance'])
            imp_df = imp_df.sort_values(by='Importance', ascending=True)
            
            # Clean feature names for display
            imp_df['Feature'] = imp_df['Feature'].str.replace('_', ' ').str.title()
            
            fig = px.bar(
                imp_df, 
                x='Importance', 
                y='Feature', 
                orientation='h',
                color='Importance',
                color_continuous_scale=px.colors.sequential.Sunsetdark,
                text_auto='.3f'
            )
            fig.update_traces(
                marker_line_width=0, 
                textfont_size=14, 
                textangle=0, 
                textposition="outside", 
                cliponaxis=False
            )
            fig.update_layout(
                height=550, 
                margin=dict(l=0, r=40, t=20, b=0),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(color='white', size=14),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
                yaxis=dict(title="", tickfont=dict(size=13, color="#e2e8f0")),
                coloraxis_showscale=False
            )
            st.plotly_chart(fig, width='stretch')
        else:
            st.info("No feature importance data available.")
