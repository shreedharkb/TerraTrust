# TerraTrust: Geospatial Credit Intelligence
## Project Execution Procedure & Setup Guide

This document outlines the standard operating procedure (SOP) for evaluators to set up, run, and validate the TerraTrust project on a local machine.

### 1. Prerequisites
Ensure you have the following installed:
* **Python 3.10+**
* **Git** (optional, for cloning)
* A modern web browser (for the Streamlit dashboard)

### 2. Environment Setup
1. Open a terminal (Command Prompt or PowerShell) and navigate to the project directory:
   ```bash
   cd TerraTrust
   ```
2. (Recommended) Create a virtual environment:
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```
3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### 3. Executing the Data & ML Pipeline
The project is built as a sequential data pipeline. To regenerate the models and data from scratch, run the scripts in the following order:

1. **Master Dataset Generation:**
   *This script handles spatial merging, physics-based imputation, and feature engineering from raw geospatial sources.*
   ```bash
   python src/build_master_dataset.py
   ```
2. **Machine Learning Model Training:**
   *This script trains the hierarchical ML models (Crop Health, Soil Quality, Water Depth, and the final Meta-Learner). It enforces spatial cross-validation and saves the `model_metrics.json`.*
   ```bash
   python src/models.py
   ```
3. **Credit Scoring Engine:**
   *This script applies the trained models to the entire state of Karnataka to compute the deterministic heuristic credit index for all 240 taluks.*
   ```bash
   python src/credit_scorer.py
   ```
4. **Visualization & Report Generation:**
   *This script generates all the high-resolution Matplotlib graphs, SHAP explainability plots, and architecture diagrams found in the research paper.*
   ```bash
   python src/generate_all_results.py
   ```
   *Note: Check the `results_and_visualizations/` folder to view the generated plots.*

### 4. Running the Interactive Dashboard
The primary user interface for TerraTrust is an interactive spatial dashboard built with Streamlit.

1. Start the Streamlit application:
   ```bash
   streamlit run app.py
   ```
2. The dashboard will automatically open in your default web browser (usually at `http://localhost:8501`).
3. **Usage:** Click on any Taluk (region) within the Karnataka map to instantly trigger the ML inference pipeline and view the real-time Credit Intelligence Report, Environmental Baseline, and Suggested Actions.

### 5. Compiling the LaTeX Report
The repository contains the soft copy of the research paper (`TerraTrust_Report.tex`).
* **Online Compilation (Recommended):** The easiest way to compile the PDF is to upload `TerraTrust_Report.tex` and the `results_and_visualizations` folder into an online LaTeX editor like [Overleaf](https://www.overleaf.com/).
* **Local Compilation:** If you have a LaTeX distribution (like MiKTeX or TeX Live) installed locally, you can compile it via command line:
  ```bash
  pdflatex TerraTrust_Report.tex
  pdflatex TerraTrust_Report.tex
  ```

### 6. Code & Dataset Access
* **Codebase:** All source code is located in the `src/` directory and `app.py`.
* **Datasets:** The generated datasets and KGIS shapefiles are located in the `data/` directory.

---
**Author:** Shreedhar K B (23BCS126)  
**Institution:** Indian Institute of Information Technology, Dharwad
