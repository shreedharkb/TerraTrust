# TerraTrust ML Pipeline — MANDATORY ROADMAP

> **READ THIS FIRST.** This is the binding execution plan for the TerraTrust ML pipeline.
> Any AI assistant working on this project MUST follow these steps IN ORDER.
> DO NOT skip steps. DO NOT generate fake/random/hardcoded data. DO NOT alter this plan.
> If you cannot complete a step, STOP and ask the user.

---

## GOLDEN RULES (NON-NEGOTIABLE)

1. **NO FAKE DATA.** Do not use `np.random`, `random.choice`, hardcoded constants (like `0.40`), or any synthetic data generation. If a data source fails, use another REAL source or ASK THE USER.

2. **ACCEPTABLE DATA SOURCES** (in order of preference):
   - **KGIS** (kgis.ksrsac.in) — Karnataka Geographic Information System
   - **Sentinel-2 L2A** — via Microsoft Planetary Computer STAC API
   - **Landsat 8/9 C2 L2** — via Microsoft Planetary Computer STAC API
   - **ISRIC SoilGrids v2.0** — Global soil data (https://rest.isric.org)
   - **NASA POWER** — Climate/weather data (https://power.larc.nasa.gov)
   - **NASA GRACE-FO** — Groundwater anomalies
   - **India WRIS** (https://indiawris.gov.in) — Water resources
   - **data.gov.in** — Indian government open data
   - **ICAR / Karnataka Dept of Agriculture** — Crop data
   - **FAO** — Food and Agriculture Organization datasets
   - **CGWB** — Central Ground Water Board
   - Any other **verifiable, citable, government or scientific source** that a professor can check

3. **IF A SOURCE FAILS**: Try the next source in the list. If ALL sources fail for a particular data point, STOP and ask the user. Do NOT fill with defaults.

4. **STEP ORDER IS MANDATORY.** Complete Step 1 fully before starting Step 2. Each step must be verified before moving on.

5. **EXISTING REAL DATA** — These files are CONFIRMED GENUINE and must NOT be deleted or overwritten:
   - `data/District.shp` (+ .dbf, .prj, .shx) — KGIS district boundaries, 31 districts
   - `data/Taluk.shp` (+ .dbf, .prj, .shx) — KGIS taluk boundaries, 271 taluks
   - `data/kgis_tabular/karnataka_soil_data.csv` — ISRIC SoilGrids v2.0 (2,282 valid rows, 118 have nulls from API failures)
   - `data/kgis_tabular/karnataka_climate_data.csv` — NASA POWER API (12,000 rows, complete)
   - `data/kgis_tabular/taluk_district_map.csv` — Spatial join from KGIS shapefiles (240 taluks → 31 districts)

6. **KNOWN BAD DATA** — These files contain fake/hardcoded values and MUST be replaced:
   - `data/satellite/karnataka_ndvi.csv` — ALL values are hardcoded 0.40. DELETE AND REPLACE.
   - `data/satellite/ndvi_timeseries.csv` — Only 40 taluks, suspicious patterns. REPLACE.
   - `data/processed/karnataka_master_dataset.csv` — Built on fake NDVI + random crops. REBUILD after fixing inputs.

---

## STEP 1: Fetch REAL Satellite NDVI Data

**Status:** NOT STARTED
**Priority:** CRITICAL — Everything else depends on this

### What to do:
1. Delete `data/satellite/karnataka_ndvi.csv` (the fake 0.40 file)
2. Create `src/fetch_real_ndvi.py` that:
   - Connects to Microsoft Planetary Computer STAC API
   - For each of the 240 taluks (use centroid from `taluk_district_map.csv` or `Taluk.shp`):
     - Try **Sentinel-2 L2A** first (collection: `sentinel-2-l2a`, bands: B04=Red, B08=NIR)
     - If no Sentinel-2 scenes, try **Landsat 8/9** (collection: `landsat-c2-l2`, bands: red, nir08)
     - Query for years 2019-2023, cloud cover < 20%
     - Download a small bbox around the centroid (0.01° × 0.01°)
     - Compute NDVI = (NIR - RED) / (NIR + RED) from actual pixel arrays
     - Store: taluk, lat, lon, year, ndvi_mean, ndvi_std, ndvi_min, ndvi_max, scene_count, platform, data_source
3. Save to `data/satellite/karnataka_ndvi_real.csv`

### Speed optimization:
- Use small bbox (0.01°) — only ~100×100 pixels per query
- Process sequentially to avoid rate limits
- Use `odc.stac.load()` for efficient raster loading
- Expected total time: 5-15 minutes for all 240 taluks × 5 years

### Verification:
- NDVI values should range from ~0.10 (bare soil/urban) to ~0.75 (dense vegetation)
- Coastal districts (Udupi, DK) should have higher NDVI than dry districts (Kalaburgi, Raichur)
- Monsoon years should show higher NDVI than drought years
- If ALL values are identical → something is wrong, investigate

---

## STEP 2: Fix Groundwater Data Gaps

**Status:** NOT STARTED
**Priority:** MEDIUM

### What to do:
1. Open `data/kgis_tabular/karnataka_groundwater.csv`
2. Check which districts are missing (currently Vijayanagara is missing — it was carved from Ballari in 2020)
3. For missing districts, use the PARENT district's values (Vijayanagara → copy Ballari's values)
4. Verify all 31 districts × 5 years × 3 seasons = 465 rows exist
5. If any district is completely missing, try:
   - **India WRIS** (https://indiawris.gov.in) for groundwater data
   - **CGWB** district-level reports
   - As last resort: use the average of adjacent districts (NOT a random number)

### Verification:
- 31 unique districts in the file
- No null values in `groundwater_depth_m`
- Values should range 5-30m (realistic for Karnataka)

---

## STEP 3: Fix Crop Mapping (ASK USER BEFORE FINALIZING)

**Status:** NOT STARTED  
**Priority:** HIGH

### The problem:
Current code defaults to "Maize" for all unknown taluks → 90.4% of dataset says "Maize"

### What to do:
1. Research the actual dominant crops per agro-climatic zone of Karnataka:
   - **Zone 1 (North East Dry)**: Kalaburgi, Bidar, Yadgir, Raichur — Tur, Jowar, Cotton
   - **Zone 2 (North Dry)**: Dharwad, Gadag, Haveri, Koppal — Maize, Cotton, Groundnut
   - **Zone 3 (North Transition)**: Belagavi, Bagalkote, Vijayapura — Sugarcane, Maize, Jowar
   - **Zone 4 (Central Dry)**: Davanagere, Chitradurga, Tumakuru, Ballari — Maize, Ragi, Sunflower
   - **Zone 5 (Eastern Dry)**: Kolara, Chikkaballapura, Bengaluru Rural — Ragi, Groundnut, Tomato
   - **Zone 6 (Southern Dry)**: Mysuru, Chamarajanagara, Mandya — Sugarcane, Paddy, Ragi
   - **Zone 7 (Southern Transition)**: Hassan, Kodagu, Chikkamagaluru — Coffee, Paddy, Arecanut
   - **Zone 8 (Hilly)**: Shivamogga, UK — Paddy, Arecanut, Pepper
   - **Zone 9 (Coastal)**: Udupi, DK — Paddy, Coconut, Arecanut
   - **Zone 10 (Bengaluru Urban)**: Mixed/Urban

2. Update `build_master_dataset.py` to map each taluk → district → zone → dominant crop
3. **STOP AND SHOW THE USER** the proposed mapping before applying it

### Verification:
- No single crop should exceed 30% of the total dataset
- Crop distribution should roughly match Karnataka's actual agriculture statistics

---

## STEP 4: Rebuild Master Dataset (ASK USER ABOUT FINANCIAL DATA)

**Status:** NOT STARTED  
**Priority:** HIGH — Cannot train models without this

### What to do:
1. Rewrite `src/build_master_dataset.py` to cleanly merge:
   - Soil (ISRIC, drop 118 null rows → 2,282 valid points)
   - Climate (NASA POWER, 12,000 rows)
   - Groundwater (fixed in Step 2)
   - NDVI (real data from Step 1)
   - Crop (fixed mapping from Step 3)
2. Expand: 2,282 points × 5 years = ~11,410 rows
3. Compute derived scores using ONLY physical thresholds:
   - `soil_suitability_score` — based on pH, nitrogen, clay vs crop requirements
   - `water_availability_score` — based on rainfall, groundwater, root zone wetness
   - `crop_health_ndvi` — categorize NDVI into health labels

### STOP POINT — Financial Data Decision:
**ASK THE USER**: "Do you want to include financial features (income, loan amount, repayment history)? Options:
   - **(A)** You provide a real CSV (e.g., Kaggle credit risk dataset)
   - **(B)** We train the credit model using ONLY physical/agricultural features (no financial data needed)
   - **(C)** We derive the credit label from the physical scores (soil + water + NDVI composite)"

**Recommended**: Option (C) — derive credit worthiness purely from agricultural capacity. This is more aligned with Statement 2 which says "geospatial evidence" should drive the credit decision, and avoids any fake financial data.

### Output:
- `data/processed/karnataka_master_dataset.csv` — CLEAN, no hardcoded fills
- `data/data_provenance.json` — updated with all real sources

### Verification:
- Run `src/audit_data.py` again
- Zero null values in critical columns (NDVI, soil scores, water scores)
- NDVI shows real variance (not flat 0.40)
- Crop distribution is realistic (not 90% Maize)

---

## STEP 5: Train ML Models (All 31 Districts, 5-Fold CV)

**Status:** NOT STARTED  
**Priority:** HIGH

### Models to train:

#### Model A: NDVI Predictor (Crop Health & Growth Stage Assessment)
```
Type:     XGBoost Regressor
Features: clay_pct, sand_pct, pH, avg_monthly_rainfall_mm, avg_root_zone_wetness, groundwater_depth_m
Target:   ndvi_annual_mean (real satellite NDVI from Step 1)
Purpose:  "Given soil + weather → predict crop health"
```

#### Model B: Soil Suitability Classifier
```
Type:     Random Forest Classifier
Features: pH, nitrogen_g_per_kg, clay_pct, sand_pct, silt_pct
Target:   soil_suitability_label (derived: Suitable / Marginal / Unsuitable)
Purpose:  "Is this soil suitable for the declared crop?"
```

#### Model C: Water Availability Regressor
```
Type:     XGBoost Regressor  
Features: avg_monthly_rainfall_mm, avg_root_zone_wetness, avg_humidity_pct, groundwater_depth_m
Target:   water_availability_score (derived from physical thresholds)
Purpose:  "How much irrigation potential does this location have?"
```

#### Model D: Credit Risk Classifier (Visual Credit Score Engine)
```
Type:     XGBoost Classifier
Features: predicted_ndvi (from Model A), soil_suitability_score, water_availability_score
Target:   credit_risk_label (Low Risk / Moderate Risk / High Risk)
Purpose:  "Should this farmer's loan be approved?"
```

### Training procedure:
1. Load clean master dataset from Step 4
2. For each model:
   a. Select features and target
   b. Drop rows with nulls in selected columns
   c. StandardScaler on features
   d. 80/20 train/test split (random_state=42)
   e. Train model
   f. Evaluate on test set
   g. Run 5-Fold Cross Validation (stratify by district for geographic generalization)
   h. Save model artifact to `models/` directory
3. GPU: Use `tree_method='hist', device='cuda'` if CUDA available, else `device='cpu'`
4. Save all metrics to `models/model_metrics.json`

### Verification:
- Model A (NDVI): R² > 0.3, RMSE < 0.15
- Model B (Soil): Accuracy > 65%
- Model C (Water): RMSE < 15
- Model D (Credit): F1 > 0.55
- Feature importances must make PHYSICAL SENSE:
  - Rainfall should be important for water model
  - pH should be important for soil model
  - NDVI should be important for credit model

---

## STEP 6: Validation & Metrics Export

**Status:** NOT STARTED  
**Priority:** HIGH

### What to do:
1. Generate comprehensive `models/model_metrics.json` with:
   - Per-model: Train accuracy, Test accuracy, CV scores (mean ± std)
   - RMSE, R², Accuracy, Precision, Recall, F1-Score as applicable
   - Confusion matrices for classifiers
   - Feature importance rankings
   - Training time
2. Generate district-level prediction summaries showing geographic variation
3. Save all model artifacts (.pkl files) to `models/`
4. Update `data/data_provenance.json` with final source documentation

### Final verification checklist:
- [ ] All NDVI values in master dataset are from real satellite data (not 0.40)
- [ ] All crop labels are from verified agro-climatic zone mapping (not random)
- [ ] All soil data is from ISRIC SoilGrids API (not generated)
- [ ] All climate data is from NASA POWER API (not generated)
- [ ] All groundwater data covers 31 districts (no gaps)
- [ ] 4 ML models trained and validated with 5-fold CV
- [ ] Metrics exported to models/model_metrics.json
- [ ] data_provenance.json lists every real source with URLs

---

## AFTER ML IS COMPLETE (DO NOT START UNTIL ALL 6 STEPS ARE DONE):
- Frontend (React dashboard) — User will instruct when to start
- Report generation — User will instruct when to start

---

## Quick Reference: Current File Status

| File | Status | Action |
|------|--------|--------|
| `data/District.shp` | REAL | Keep |
| `data/Taluk.shp` | REAL | Keep |
| `data/kgis_tabular/karnataka_soil_data.csv` | REAL (118 nulls) | Keep, drop null rows during merge |
| `data/kgis_tabular/karnataka_climate_data.csv` | REAL | Keep |
| `data/kgis_tabular/karnataka_groundwater.csv` | REAL (missing 1 district) | Fix in Step 2 |
| `data/kgis_tabular/taluk_district_map.csv` | REAL | Keep |
| `data/satellite/karnataka_ndvi.csv` | **FAKE (all 0.40)** | Delete, replace in Step 1 |
| `data/satellite/ndvi_timeseries.csv` | **SUSPICIOUS** | Replace in Step 1 |
| `data/processed/karnataka_master_dataset.csv` | **BUILT ON FAKE DATA** | Rebuild in Step 4 |
