"""
TerraTrust Configuration
========================
Central configuration for paths, constants, and target region settings.
"""
import os

# ============================================================
# Project Paths
# ============================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
KGIS_DIR = os.path.join(DATA_DIR, "kgis_boundaries")
TABULAR_DIR = os.path.join(DATA_DIR, "kgis_tabular")
SATELLITE_DIR = os.path.join(DATA_DIR, "satellite")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Create directories if they don't exist
for d in [TABULAR_DIR, SATELLITE_DIR, PROCESSED_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# KGIS Shapefile Paths
# ============================================================
SHAPEFILES_DIR = os.path.join(DATA_DIR, "raw", "shapefiles")
DISTRICT_SHP = os.path.join(SHAPEFILES_DIR, "District.shp")
TALUK_SHP = os.path.join(SHAPEFILES_DIR, "Taluk.shp")

# ============================================================
# CRS Configuration
# ============================================================
TARGET_CRS = "EPSG:4326"  # WGS84 - needed for all API calls

# ============================================================
# Target Region: Davangere District
# ============================================================
TARGET_DISTRICT = "Davanagere"  # Spelling as used in KGIS shapefiles
TARGET_DISTRICT_ALT = ["Davangere", "Davanagere", "DAVANAGERE", "DAVANGERE"]

# Davangere Taluks (6 taluks)
TARGET_TALUKS = [
    "Davanagere",
    "Harihara",
    "Jagalur",
    "Honnali",
    "Channagiri",
    "Nyamathi"
]

# ============================================================
# Crop Configuration
# ============================================================
# Davangere is known for: Maize, Cotton, Rice (Paddy), Jowar, Sunflower
CROPS = ["Maize", "Cotton", "Rice", "Jowar", "Sunflower"]
PRIMARY_CROP = "Maize"  # Davangere is called the "Maize City of Karnataka"

# Crop-specific soil requirements (NPK ranges in kg/ha)
CROP_REQUIREMENTS = {
    "Maize":     {"N_min": 120, "N_max": 180, "P_min": 50, "P_max": 80, "K_min": 40, "K_max": 60, "pH_min": 5.5, "pH_max": 7.5},
    "Cotton":    {"N_min": 80,  "N_max": 120, "P_min": 40, "P_max": 60, "K_min": 40, "K_max": 60, "pH_min": 6.0, "pH_max": 8.0},
    "Rice":      {"N_min": 100, "N_max": 150, "P_min": 40, "P_max": 60, "K_min": 40, "K_max": 80, "pH_min": 5.0, "pH_max": 7.0},
    "Jowar":     {"N_min": 60,  "N_max": 100, "P_min": 30, "P_max": 50, "K_min": 25, "K_max": 50, "pH_min": 6.0, "pH_max": 8.5},
    "Sunflower": {"N_min": 60,  "N_max": 90,  "P_min": 40, "P_max": 60, "K_min": 30, "K_max": 50, "pH_min": 6.0, "pH_max": 7.5},
}

# ============================================================
# NDVI Thresholds for Crop Health Classification
# ============================================================
NDVI_THRESHOLDS = {
    "Barren/Water":   (-1.0, 0.1),
    "Stressed":       (0.1, 0.25),
    "Moderate":       (0.25, 0.45),
    "Healthy":        (0.45, 0.65),
    "Very Healthy":   (0.65, 1.0),
}

# Crop growth stage NDVI profiles (approximate)
GROWTH_STAGES = {
    "Germination":    (0.10, 0.20),
    "Vegetative":     (0.25, 0.45),
    "Flowering":      (0.45, 0.65),
    "Maturity":       (0.55, 0.75),
    "Harvest-ready":  (0.30, 0.45),
}

# ============================================================
# Credit Score Configuration
# ============================================================
CREDIT_SCORE_WEIGHTS = {
    "crop_health":       0.30,   # NDVI-based crop health score
    "water_availability": 0.25,  # Groundwater + NDWI
    "soil_suitability":   0.25,  # Soil NPK match to crop requirements
    "historical_trend":   0.20,  # Historical NDVI trend stability
}

RISK_CATEGORIES = {
    "Low Risk (Approve)":    (75, 100),
    "Moderate Risk (Review)": (50, 74),
    "High Risk (Reject)":    (0, 49),
}

# ============================================================
# API Endpoints (Genuine Data Sources)
# ============================================================
# ISRIC SoilGrids API - Real global soil data
SOILGRIDS_API = "https://rest.isric.org/soilgrids/v2.0/properties/query"

# NASA POWER API - Real climate/weather data
NASA_POWER_API = "https://power.larc.nasa.gov/api/temporal/monthly/point"

# Microsoft Planetary Computer STAC API - Real Sentinel-2 imagery
PLANETARY_COMPUTER_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
