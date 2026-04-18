<div align="center">
  
# 🌱 TerraTrust 
**Advanced Agro-Ecological Intelligence & Satellite Pipeline**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![React](https://img.shields.io/badge/React-20232A?style=flat&logo=react&logoColor=61DAFB)](#)
[![Geopandas](https://img.shields.io/badge/Geopandas-Geospatial-green)](#)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine%20Learning-orange)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

*An objective, data-driven Machine Learning engine designed to evaluate independent agricultural suitability and credit risk using 100% genuine real-world geographic bounds, authentic satellite spectral analysis, and historical government records.*

---

</div>

## 📖 Overview

TerraTrust bridges the gap between raw spatial data and actionable agricultural insight. Constructed originally as a rigorous academic course project, the system performs end-to-end geospatial intelligence gathering across the **Davangere District** in Karnataka.

Rather than relying on abstract simulations, TerraTrust enforces an architecture built exclusively on **100% genuine APIs and physical properties**, forcing its Machine Learning ensemble to learn the genuine messy statistical correlations of real-world farming environments.

## ✨ Key Features

- 🛰️ **Authentic Satellite Pipeline**: Natively queries the **Microsoft Planetary Computer STAC** endpoint to download authentic Sentinel-2 L2A Cloud-Optimized GeoTIFF arrays. Performs mathematical pixel analysis to extract true NDVI and NDWI indices while aggressively rendering out cloud/shadow noise utilizing the native Scene Classification Layer (SCL).
- 🌍 **Genuine Geographical Sampling**: Consumes authentic KGIS (Karnataka Geographic Information System) vector shapefiles (`.shp`). Generates randomized, rigorous geometric coordinate points strictly bonded within the real boundaries of regional Taluks.
- 📡 **Live Environmental Ingestion**: Seamlessly pings verified scientific catalogs—including **ISRIC SoilGrids v2** and **NASA POWER**—to fetch the explicit historical micro-climate characteristics (Rainfall, Humidity, Soil pH, Clay Density) bound exactly to the sampled geometric coordinates.
- 📉 **Independent ML Evaluation**: Prevents arbitrary deterministic data leaks by training its predictive ML ensemble (Random Forest, Gradient Boosting, XGBoost) strictly against independently injected historical benchmarks (Yield Records & Repayment Histories) mapped to the Taluk boundaries from genuine CSV databases.

## ⚙️ Architecture

TerraTrust consists of three primary, sequential computational engines:

1. **`src/data_pipeline.py`**
   The extraction layer. Sub-samples authentic geographical bounding boxes and pulls raw API metadata. Joins historical reference databases (`land_records.csv`, `groundwater.csv`) natively onto the extracted coordinate grid to build the raw truth metrics.
2. **`src/satellite_pipeline.py`** 
   The spatial physics layer. Performs REST-based downloads and aggregations of massive Sentinel-2 TIFF items. Translates spectral bands (B04, B08, B03) into human-readable vegetation/water index arrays.
3. **`src/models.py`** & **`src/credit_scorer.py`**
   The intelligence layer. Ingests all merged spatial data. Trains an XGBoost classifier and regressor system to independently score the relationship between micro-climate features and historic yield. Generates an objective, transparent Agro-Ecological **Visual Credit Score** mapped directly to the original physical boundaries.

## 🚀 Getting Started

### Prerequisites
* Python 3.9+
* Required Libraries: `geopandas`, `odc.stac`, `pystac-client`, `rasterio`, `xarray`, `xgboost`, `scikit-learn`, `pandas`.

### Installation & Execution

1. **Clone the repository:**
   ```bash
   git clone https://github.com/shreedharkb/TerraTrust.git
   cd TerraTrust
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the sequential pipelines:**
   ```bash
   # 1. Fetch geographic attributes and API catalogs
   python src/data_pipeline.py
   
   # 2. Train the intelligence models on independent historical metrics 
   python src/models.py
   
   # 3. Generate the conclusive Visual Credit Reports 
   python src/credit_scorer.py
   ```

## 📊 Evaluation Matrix

TerraTrust evaluates properties based on composite scoring thresholds mapping exclusively physical metrics to historical performance:
* **Low Risk**: Strong agricultural indicators (Ideal Soil pH, High Rainfall, Steady NDVI). Loan authorization is highly recommended.
* **Medium Risk**: Moderate indicators. Requires manual field verification or irrigation advisory.
* **High Risk**: Severe physical disqualifications (Barren NDWI, Unsuitable Clay %, Low Rainfall Trends).

<div align="center">
  <br>
  <i>Built with dedication for course curriculum exploring advanced geospatial ML infrastructure.</i>
</div>