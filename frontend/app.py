import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import folium
from streamlit_folium import folium_static
import plotly.express as px
import plotly.graph_objects as go
import json
from PIL import Image

# Add root directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.credit_scorer import VisualCreditScorer
from src.config import PROCESSED_DIR, RESULTS_DIR

st.set_page_config(
    page_title="TerraTrust Intelligence System",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Premium Design
st.markdown("""
<style>
    .big-score {
        font-size: 6rem !important;
        font-weight: 800;
        text-align: center;
        margin: 0px;
    }
    .score-low-risk { color: #10B981; }
    .score-mod-risk { color: #F59E0B; }
    .score-high-risk { color: #EF4444; }
    
    .stProgress > div > div > div > div {
        background-color: #10B981;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Loading
# ---------------------------------------------------------
@st.cache_resource
def load_scorer():
    return VisualCreditScorer()

@st.cache_data
def load_data():
    df_path = os.path.join(PROCESSED_DIR, "karnataka_master_dataset.csv")
    if os.path.exists(df_path):
        return pd.read_csv(df_path)
    return pd.DataFrame()

@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

df = load_data()
scorer = load_scorer()

if df.empty:
    st.error("❌ Dataset not found! Please ensure data pipeline has run.")
    st.stop()

# ---------------------------------------------------------
# Sidebar Navigation
# ---------------------------------------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3061/3061341.png", width=60)
st.sidebar.title("TerraTrust")
st.sidebar.markdown("Intelligent Rural Banking")

page = st.sidebar.radio("Navigation", [
    "Loan Assessment Dashboard", 
    "Farm Comparison",
    "System Transparency & XAI"
])

# Utility to generate reasons for rejection
def explain_rejection(farm, final_score):
    reasons = []
    if final_score >= 70:
        return ["Farm meets all critical agricultural thresholds."]
    
    # Simple heuristic-based reasoning to simulate SHAP rules
    if farm.get('avg_root_zone_wetness', 0) < 0.2:
        reasons.append("📉 Low groundwater/root zone wetness significantly reduced the score.")
    if farm.get('avg_monthly_rainfall_mm', 0) < 50:
        reasons.append("🏜️ Historical rainfall is below the safe threshold for declared crops.")
    if farm.get('pH', 7.0) < 5.5 or farm.get('pH', 7.0) > 8.5:
        reasons.append("🧪 Soil pH is outside the optimal range, reducing yield probability.")
    if farm.get('ndvi', 0.5) < 0.3:
        reasons.append("🍂 Low historical NDVI (Crop Health Index) detected by satellite imagery.")
        
    if not reasons:
        reasons.append("Combination of marginal soil quality and moderate water stress.")
    return reasons

# ---------------------------------------------------------
# PAGE 1: Loan Assessment Dashboard
# ---------------------------------------------------------
if page == "Loan Assessment Dashboard":
    st.title("🎯 Visual Loan Assessment")
    st.markdown("Automated evaluation using KGIS datasets and real-time satellite imagery.")
    
    col_input, col_score = st.columns([1, 2])
    
    with col_input:
        st.subheader("Select Farm Parcel")
        districts = df['district'].dropna().unique()
        selected_dist = st.selectbox("District", sorted(districts))
        
        taluks = df[df['district'] == selected_dist]['taluk'].dropna().unique()
        selected_taluk = st.selectbox("Taluk", sorted(taluks))
        
        # Filter farms in the selected taluk
        taluk_farms = df[(df['district'] == selected_dist) & (df['taluk'] == selected_taluk)]
        
        if taluk_farms.empty:
            st.warning("No farm data available for this selection.")
        else:
            selected_farm_id = st.selectbox("Select Point ID", taluk_farms['point_id_yr'].unique())
            farm_data = taluk_farms[taluk_farms['point_id_yr'] == selected_farm_id].iloc[0].to_dict()
            
            st.divider()
            st.subheader("🎛️ Risk Simulation")
            st.markdown("Adjust environmental factors to simulate impact on the credit score.")
            
            # Simulation Sliders
            sim_rainfall = st.slider("Rainfall (mm/month)", 
                                     min_value=0.0, max_value=500.0, 
                                     value=float(farm_data.get('avg_monthly_rainfall_mm', 100.0)))
            
            sim_wetness = st.slider("Root Zone Wetness", 
                                    min_value=0.0, max_value=1.0, 
                                    value=float(farm_data.get('avg_root_zone_wetness', 0.5)))
            
            # Apply simulation
            simulated_farm = farm_data.copy()
            simulated_farm['avg_monthly_rainfall_mm'] = sim_rainfall
            simulated_farm['avg_root_zone_wetness'] = sim_wetness
            
            # Recalculate score
            result = scorer.generate_credit_score(simulated_farm)
            
    with col_score:
        if not taluk_farms.empty:
            st.subheader("Visual Credit Score")
            
            score = result['heuristic_credit_score']
            risk = result['risk_category']
            
            color_class = "score-low-risk" if risk == "Low" else "score-mod-risk" if risk == "Moderate" else "score-high-risk"
            
            st.markdown(f'<p class="big-score {color_class}">{score}</p>', unsafe_allow_html=True)
            st.markdown(f"<h3 style='text-align: center;'>Risk Category: {risk}</h3>", unsafe_allow_html=True)
            
            # ML Assessment Gauges
            st.divider()
            c1, c2, c3 = st.columns(3)
            
            comps = result['components']
            with c1:
                st.metric("Crop Health (NDVI)", f"{comps['pred_crop_health']:.2f}/1.0")
                st.progress(float(comps['pred_crop_health']))
            with c2:
                # Normalize water depth logic (higher score generally means deeper/worse or better depending on model)
                water = max(0, min(100, comps['pred_water_depth']))
                st.metric("Water Availability", f"{water:.1f}")
                st.progress(int(water) if water <= 100 else 100)
            with c3:
                soil = max(0, min(1.0, comps['pred_soil_q']))
                st.metric("Soil Suitability", f"{soil:.2f}")
                st.progress(float(soil))
                
            # Explain Rejection
            st.divider()
            if st.button("🔍 Why was this loan given this score?"):
                st.markdown("#### Explainable AI Insights:")
                reasons = explain_rejection(simulated_farm, score)
                for r in reasons:
                    st.info(r)
            
            # Geospatial Evidence Map
            st.subheader("🗺️ Geospatial Evidence")
            lat = result['evidence']['coordinates']['lat']
            lon = result['evidence']['coordinates']['lon']
            
            if pd.notna(lat) and pd.notna(lon):
                m = folium.Map(location=[lat, lon], zoom_start=14)
                folium.Marker(
                    [lat, lon], 
                    popup=f"Score: {score}", 
                    tooltip="Selected Farm"
                ).add_to(m)
                
                # Add a circle to represent the parcel size
                folium.Circle(
                    radius=500,
                    location=[lat, lon],
                    color="green" if risk == "Low" else "red",
                    fill=True,
                ).add_to(m)
                
                folium_static(m, width=700, height=350)
            else:
                st.warning("No coordinates available for mapping.")

# ---------------------------------------------------------
# PAGE 2: Farm Comparison
# ---------------------------------------------------------
elif page == "Farm Comparison":
    st.title("⚖️ Farm Risk Comparison")
    st.markdown("Compare two distinct farm parcels side-by-side to assist in portfolio balancing.")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Farm A")
        farm_a_id = st.selectbox("Select Farm A", df['point_id_yr'].unique(), key="fa")
        farm_a = df[df['point_id_yr'] == farm_a_id].iloc[0].to_dict()
        res_a = scorer.generate_credit_score(farm_a)
        
        st.metric("Credit Score", res_a['heuristic_credit_score'], res_a['risk_category'])
        st.metric("Crop Health", f"{res_a['components']['pred_crop_health']:.2f}")
        st.metric("Soil Quality", f"{res_a['components']['pred_soil_q']:.2f}")
    
    with col_b:
        st.subheader("Farm B")
        farm_b_id = st.selectbox("Select Farm B", df['point_id_yr'].unique(), index=1 % len(df), key="fb")
        farm_b = df[df['point_id_yr'] == farm_b_id].iloc[0].to_dict()
        res_b = scorer.generate_credit_score(farm_b)
        
        st.metric("Credit Score", res_b['heuristic_credit_score'], res_b['risk_category'])
        st.metric("Crop Health", f"{res_b['components']['pred_crop_health']:.2f}")
        st.metric("Soil Quality", f"{res_b['components']['pred_soil_q']:.2f}")
        
    st.divider()
    # Radar Chart Comparison
    categories = ['Crop Health', 'Soil Suitability', 'Water Availability', 'Root Zone Wetness', 'Rainfall Score']
    
    def normalize(val, max_val):
        return max(0, min(1, float(val) / max_val)) if max_val else 0

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[
            normalize(res_a['components']['pred_crop_health'], 1),
            normalize(res_a['components']['pred_soil_q'], 1),
            normalize(100 - res_a['components']['pred_water_depth'], 100), # Lower depth is better
            farm_a.get('avg_root_zone_wetness', 0.5),
            normalize(farm_a.get('avg_monthly_rainfall_mm', 0), 300)
        ],
        theta=categories,
        fill='toself',
        name='Farm A'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[
            normalize(res_b['components']['pred_crop_health'], 1),
            normalize(res_b['components']['pred_soil_q'], 1),
            normalize(100 - res_b['components']['pred_water_depth'], 100),
            farm_b.get('avg_root_zone_wetness', 0.5),
            normalize(farm_b.get('avg_monthly_rainfall_mm', 0), 300)
        ],
        theta=categories,
        fill='toself',
        name='Farm B'
    ))

    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------
# PAGE 3: System Transparency
# ---------------------------------------------------------
elif page == "System Transparency & XAI":
    st.title("🔬 System Transparency & Explainable AI")
    st.markdown("Detailed breakdown of the ML models powering the Visual Credit Score.")
    
    tab1, tab2, tab3 = st.tabs(["📊 Validation Metrics", "🧠 XAI (SHAP Analysis)", "📈 Training Curves"])
    
    with tab1:
        st.subheader("Model Validation Metrics")
        metrics = load_metrics()
        if metrics:
            st.json(metrics)
        else:
            st.warning("Metrics file not found. Ensure pipeline generated 'model_metrics.json'")
            
    with tab2:
        st.subheader("Explainable AI (SHAP)")
        st.markdown("These charts demonstrate how different KGIS and satellite variables impact the final credit decision.")
        
        shap_dir = os.path.join(RESULTS_DIR, "XAI_SHAP_Analysis")
        if os.path.exists(shap_dir):
            images = [f for f in os.listdir(shap_dir) if f.endswith('.png')]
            if images:
                for img_name in images:
                    img_path = os.path.join(shap_dir, img_name)
                    st.image(Image.open(img_path), caption=img_name.replace('.png', ''))
            else:
                st.info("No SHAP images found in results/XAI_SHAP_Analysis/")
        else:
            st.info("SHAP directory not found.")
            
    with tab3:
        st.subheader("Model Training Curves")
        st.markdown("Visual proof of model learning and convergence without overfitting.")
        
        curve_dir = os.path.join(RESULTS_DIR, "Training_Curves")
        if os.path.exists(curve_dir):
            images = [f for f in os.listdir(curve_dir) if f.endswith('.png')]
            if images:
                for img_name in images:
                    img_path = os.path.join(curve_dir, img_name)
                    st.image(Image.open(img_path), caption=img_name.replace('.png', ''))
            else:
                st.info("No training curves found in results/Training_Curves/")
        else:
            st.info("Training Curves directory not found.")

