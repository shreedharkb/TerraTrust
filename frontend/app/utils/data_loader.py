import os
import sys
import pandas as pd
import json
import streamlit as st

# Add root directory to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.credit_scorer import VisualCreditScorer
from src.config import PROCESSED_DIR, RESULTS_DIR

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
def load_geojson():
    geojson_path = os.path.join(PROCESSED_DIR, "karnataka_taluks.geojson")
    if os.path.exists(geojson_path):
        with open(geojson_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

@st.cache_data
def load_metrics():
    path = os.path.join(RESULTS_DIR, "model_metrics.json")
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}
