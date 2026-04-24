"""
TerraTrust — Master Dataset Rebuild (v2 – Audit-Compliant)
============================================================
Merges all real data sources into the final training dataset:
  - Soil data (ISRIC SoilGrids)       → karnataka_soil_data.csv
  - Climate data (NASA POWER)         → karnataka_climate_data.csv
  - Groundwater (NASA GRACE-FO model) → karnataka_groundwater.csv
  - NDVI / Satellite (Landsat C2 L2)  → karnataka_ndvi_real.csv
  - Taluk/District boundaries (KGIS)  → Taluk.shp

AUDIT FIX v2:
  - Phase 1: Data cleaning (missing values, negative NDVI, imputation log)
  - Phase 2: New physically-meaningful interaction features
  - Deterministic scores KEPT for credit_scorer.py but FLAGGED as non-training columns
  - New column `_is_training_feature` metadata in provenance

Outputs:
  - data/processed/karnataka_master_dataset.csv (full training set)
  - data/processed/data_provenance.json (updated)
  - data/processed/data_audit_log.csv (all imputation decisions)
"""

import os
import sys
import json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

BASE_DIR     = os.path.join(os.path.dirname(__file__), '..')
TABULAR_DIR  = os.path.normpath(os.path.join(BASE_DIR, 'data', 'kgis_tabular'))
SATELLITE_DIR= os.path.normpath(os.path.join(BASE_DIR, 'data', 'satellite'))
PROCESSED_DIR= os.path.normpath(os.path.join(BASE_DIR, 'data', 'processed'))
DATA_DIR     = os.path.normpath(os.path.join(BASE_DIR, 'data'))

SOIL_CSV      = os.path.join(TABULAR_DIR,   'karnataka_soil_data.csv')
CLIMATE_CSV   = os.path.join(TABULAR_DIR,   'karnataka_climate_data.csv')
GW_CSV        = os.path.join(TABULAR_DIR,   'karnataka_groundwater.csv')
NDVI_CSV      = os.path.join(SATELLITE_DIR, 'karnataka_ndvi_real.csv')
MASTER_CSV    = os.path.join(PROCESSED_DIR, 'karnataka_master_dataset.csv')
AUDIT_LOG_CSV = os.path.join(PROCESSED_DIR, 'data_audit_log.csv')


# ─── Crop declared per taluk (mapped from district agro-climatic zones) ─────────
with open(os.path.join(BASE_DIR, 'src', 'taluk_dist_map.json'), 'r') as f:
    TALUK_TO_DIST = json.load(f)

DISTRICT_CROPS = {
    'Kalaburgi': 'Tur', 'Bidar': 'Jowar', 'Yadgir': 'Cotton', 'Raichur': 'Cotton',
    'Dharwad': 'Maize', 'Gadag': 'Groundnut', 'Haveri': 'Maize', 'Koppal': 'Cotton',
    'Belagavi': 'Sugarcane', 'Bagalkote': 'Jowar', 'Vijayapura': 'Maize',
    'Davanagere': 'Maize', 'Chitradurga': 'Sunflower', 'Tumakuru': 'Ragi',
    'Ballari': 'Ragi', 'Vijayanagara': 'Maize',
    'Kolara': 'Tomato', 'Chikkaballapura': 'Groundnut', 'Bengaluru (Rural)': 'Ragi',
    'Mysuru': 'Paddy', 'Chamarajanagara': 'Ragi', 'Mandya': 'Sugarcane',
    'Hassan': 'Coffee', 'Kodagu': 'Coffee', 'Chikkamagaluru': 'Arecanut',
    'Shivamogga': 'Arecanut', 'Uttara Kannada': 'Pepper',
    'Udupi': 'Paddy', 'Dakshina Kannada': 'Coconut',
    'Bengaluru (Urban)': 'Mixed/Urban',
    'Bengaluru South': 'Ragi'
}

def get_declared_crop(taluk):
    """Map taluk -> district -> dominant crop type."""
    district = TALUK_TO_DIST.get(taluk)
    if district in DISTRICT_CROPS:
        return DISTRICT_CROPS[district]
    if any(k in taluk for k in ['Mangaluru','Udupi','Sirsi','Bhatkal','Honnavar']):
        return 'Paddy'
    return 'Maize'


# ══════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════

def load_soil():
    print("  [1/5] Loading soil data...", end=" ")
    df = pd.read_csv(SOIL_CSV)
    numeric_cols = ['clay_pct','sand_pct','silt_pct','pH','nitrogen_g_per_kg']
    df = df.dropna(subset=numeric_cols, how='all')
    print(f"OK — {len(df)} rows, {df['taluk'].nunique()} taluks")
    return df


def load_climate():
    print("  [2/5] Loading climate data...", end=" ")
    df = pd.read_csv(CLIMATE_CSV)
    print(f"OK — {len(df)} rows")
    return df


def load_groundwater():
    print("  [3/5] Loading groundwater data...", end=" ")
    if not os.path.exists(GW_CSV) or os.path.getsize(GW_CSV) < 100:
        print("MISSING — using Karnataka state average 14m proxy")
        return None
    df = pd.read_csv(GW_CSV)
    if df.empty:
        print("EMPTY — using proxy")
        return None
    print(f"OK — {len(df)} rows, {df['district'].nunique()} districts")
    return df


def load_ndvi():
    print("  [4/5] Loading NDVI satellite data...", end=" ")
    if not os.path.exists(NDVI_CSV) or os.path.getsize(NDVI_CSV) < 100:
        print("MISSING — NDVI column will be estimated from crop calendar")
        return None
    df = pd.read_csv(NDVI_CSV)
    if df.empty:
        print("EMPTY")
        return None
    print(f"OK — {len(df)} rows, {df['taluk'].nunique()} taluks")
    return df


# ══════════════════════════════════════════════════════════════════
# PHASE 1: DATA CLEANING FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def clean_and_audit(master):
    """
    Phase 1 — Data cleaning with full audit trail.
    Returns cleaned dataframe and audit log list.
    """
    audit_log = []

    # 1) Fix missing bulk_density_cg_per_cm3 — taluk-median imputation
    if 'bulk_density_cg_per_cm3' in master.columns:
        missing_bd = master['bulk_density_cg_per_cm3'].isna()
        if missing_bd.any():
            taluk_medians = master.groupby('taluk')['bulk_density_cg_per_cm3'].transform('median')
            global_median = master['bulk_density_cg_per_cm3'].median()
            fill_vals = taluk_medians.fillna(global_median)
            for idx in master[missing_bd].index:
                pid = master.loc[idx, 'point_id_yr'] if 'point_id_yr' in master.columns else str(idx)
                audit_log.append({
                    'point_id_yr': pid,
                    'column': 'bulk_density_cg_per_cm3',
                    'action': 'imputed_taluk_median',
                    'original_value': 'NaN',
                    'new_value': round(fill_vals.loc[idx], 1)
                })
            master.loc[missing_bd, 'bulk_density_cg_per_cm3'] = fill_vals[missing_bd]
            print(f"    AUDIT: Imputed {missing_bd.sum()} missing bulk_density values (taluk-median)")

    # 2) Fix missing ndvi_annual_std — set to 0 (single annual observation)
    if 'ndvi_annual_std' in master.columns:
        missing_std = master['ndvi_annual_std'].isna()
        if missing_std.any():
            for idx in master[missing_std].index:
                pid = master.loc[idx, 'point_id_yr'] if 'point_id_yr' in master.columns else str(idx)
                audit_log.append({
                    'point_id_yr': pid,
                    'column': 'ndvi_annual_std',
                    'action': 'filled_zero_single_obs',
                    'original_value': 'NaN',
                    'new_value': 0.0
                })
            master.loc[missing_std, 'ndvi_annual_std'] = 0.0
            print(f"    AUDIT: Filled {missing_std.sum()} missing ndvi_annual_std with 0")

    # 3) Handle negative NDVI — cap at 0 and flag
    if 'ndvi_annual_mean' in master.columns:
        neg_ndvi = master['ndvi_annual_mean'] < 0
        if neg_ndvi.any():
            master['ndvi_flagged'] = neg_ndvi.astype(int)
            for idx in master[neg_ndvi].index:
                pid = master.loc[idx, 'point_id_yr'] if 'point_id_yr' in master.columns else str(idx)
                audit_log.append({
                    'point_id_yr': pid,
                    'column': 'ndvi_annual_mean',
                    'action': 'capped_negative_to_zero',
                    'original_value': round(master.loc[idx, 'ndvi_annual_mean'], 4),
                    'new_value': 0.0
                })
            master.loc[neg_ndvi, 'ndvi_annual_mean'] = 0.0
            print(f"    AUDIT: Capped {neg_ndvi.sum()} negative NDVI values to 0 (flagged in ndvi_flagged)")
        else:
            master['ndvi_flagged'] = 0

    return master, audit_log


# ══════════════════════════════════════════════════════════════════
# PHASE 2: PHYSICALLY-MEANINGFUL INTERACTION FEATURES
# ══════════════════════════════════════════════════════════════════

def add_interaction_features(master):
    """
    Phase 2 — Create physically meaningful interaction features
    that break the leakage from deterministic scores.
    These give the ML models real agricultural signals to learn from.
    """
    print("    Computing interaction features...")

    # 1) Soil water retention capacity (clay traps water, organic matter holds it)
    master['soil_water_retention'] = (
        master['clay_pct'] * master['organic_carbon_dg_per_kg'] / 100.0
    ).round(4)

    # 2) Aridity index (FAO-inspired: higher = wetter regime)
    temp_range = (master['max_temp_c'] - master['min_temp_c']).replace(0, np.nan)
    master['aridity_index'] = (
        master['avg_monthly_rainfall_mm'] / temp_range
    ).round(4)

    # 3) Vegetation-moisture coupling (NDVI should correlate with root zone wetness)
    master['vegetation_stress_index'] = (
        master['ndvi_annual_mean'] * master['avg_root_zone_wetness']
    ).round(4)

    # 4) Soil fertility index (nitrogen + organic carbon normalized by pH stress)
    ph_stress = np.abs(master['pH'] - 6.8)  # optimal pH ~6.8 for most crops
    master['soil_fertility_index'] = (
        (master['nitrogen_g_per_kg'] + master['organic_carbon_dg_per_kg'] / 100.0) /
        (1 + ph_stress)
    ).round(4)

    # 5) Water table pressure (shallow groundwater + high rainfall = waterlogging risk)
    master['water_table_pressure'] = (
        master['avg_monthly_rainfall_mm'] / (master['groundwater_depth_m'] + 1)
    ).round(4)

    # 6) Thermal comfort index for crops (deviation from optimal 25-30°C range)
    avg_temp = (master['max_temp_c'] + master['min_temp_c']) / 2.0
    master['thermal_stress'] = np.abs(avg_temp - 27.5).round(4)

    # 7) Soil texture ratio (sand/clay ratio indicates drainage vs retention)
    master['sand_clay_ratio'] = (
        master['sand_pct'] / master['clay_pct'].replace(0, np.nan)
    ).round(4)

    n_features = 7
    print(f"    Added {n_features} physically-meaningful interaction features")
    return master


# ══════════════════════════════════════════════════════════════════
# LEGACY SCORING (kept for credit_scorer.py but NOT for ML training)
# ══════════════════════════════════════════════════════════════════

def soil_suitability(row):
    """Deterministic soil score — used ONLY for heuristic credit scoring, NOT ML training."""
    score = 70.0
    crop = str(row.get('declared_crop', 'Mixed/Urban'))
    ph = row.get('pH', np.nan)
    clay = row.get('clay_pct', np.nan)
    n = row.get('nitrogen_g_per_kg', np.nan)

    if pd.notna(ph):
        if crop in ['Coffee', 'Pepper'] and (ph > 6.5 or ph < 4.5): score -= 25
        elif crop in ['Sugarcane', 'Cotton'] and (ph < 6.0 or ph > 8.5): score -= 25
        elif crop in ['Paddy', 'Rice'] and (ph < 5.0 or ph > 8.5): score -= 20
        elif (ph < 5.5 or ph > 8.5): score -= 20
        else: score += 10
    if pd.notna(clay):
        if crop in ['Cotton', 'Paddy', 'Sugarcane'] and clay < 30: score -= 20
        elif crop in ['Groundnut'] and clay > 40: score -= 25
        elif clay > 60 or clay < 10: score -= 15
        else: score += 5
    if pd.notna(n):
        if n < 0.5: score -= 25
        elif n > 1.2: score += 10
    return max(0, min(100, round(score, 1)))


def water_availability(row):
    """Deterministic water score — used ONLY for heuristic credit scoring, NOT ML training."""
    score = 50.0
    crop = str(row.get('declared_crop', 'Mixed/Urban'))
    rain = row.get('avg_monthly_rainfall_mm', np.nan)
    gw  = row.get('groundwater_depth_m', np.nan)
    min_temp = row.get('min_temp_c', np.nan)

    if pd.notna(rain):
        if crop in ['Paddy', 'Sugarcane', 'Coffee']:
            if rain < 60: score -= 25
            elif rain > 100: score += 20
        elif crop in ['Jowar', 'Ragi', 'Maize', 'Groundnut']:
            if rain > 80: score -= 10
            elif 30 <= rain <= 60: score += 15
            elif rain < 15: score -= 20
        else:
            if rain > 80: score += 15
            elif rain < 20: score -= 15

    if pd.notna(gw):
        if gw < 5: score += 20
        elif gw < 15: score += 5
        elif gw > 25: score -= 20

    ndvi = row.get('ndvi_annual_mean', np.nan)
    if pd.notna(ndvi): score += ndvi * 10

    temp = row.get('max_temp_c', 35.0)
    if pd.notna(temp): score -= abs(temp - 30.0) * 1.5
    if pd.notna(min_temp): score -= abs(min_temp - 20.0) * 0.8

    return max(0, min(100, round(score, 1)))


def credit_score_heuristic(row):
    """Deterministic credit score — used ONLY for heuristic baseline, NOT ML training."""
    base = 430.0

    soil = row.get('soil_suitability_score', 50)
    water = row.get('water_availability_score', 50)
    ndvi = row.get('ndvi_annual_mean', 0.4)
    gw = row.get('groundwater_depth_m', 14)
    rain = row.get('avg_monthly_rainfall_mm', 0)
    ph = row.get('pH', 7.0)
    n = row.get('nitrogen_g_per_kg', 1.0)
    wet = row.get('avg_root_zone_wetness', 0.5)
    max_temp = row.get('max_temp_c', np.nan)
    min_temp = row.get('min_temp_c', np.nan)

    base += soil * 1.15
    base += water * 0.85
    base += ndvi * 95
    base += wet * 45

    if gw > 20: base -= 100
    elif gw > 15: base -= 55
    if soil < 50: base -= 100
    elif soil < 65: base -= 45
    if ndvi < 0.20: base -= 100
    elif ndvi < 0.30: base -= 50
    if water < 40: base -= 75
    elif water < 55: base -= 30
    if rain < 20: base -= 35
    elif rain > 95: base += 10
    if ph < 5.5 or ph > 8.5: base -= 30
    elif 6.2 <= ph <= 7.8: base += 10
    if n < 0.6: base -= 30
    elif n > 1.2: base += 10
    if pd.notna(max_temp):
        if max_temp > 37: base -= 18
        elif max_temp < 31: base += 6
    if pd.notna(min_temp):
        if min_temp < 14: base -= 10
        elif 18 <= min_temp <= 23: base += 4

    return max(300, min(900, round(base)))


def ndvi_health(ndvi):
    """NDVI-based crop health label."""
    if ndvi >= 0.60: return 'Excellent'
    elif ndvi >= 0.45: return 'Good'
    elif ndvi >= 0.30: return 'Moderate'
    elif ndvi >= 0.15: return 'Poor'
    else: return 'Bare/Stressed'


# ══════════════════════════════════════════════════════════════════
# MAIN BUILDER
# ══════════════════════════════════════════════════════════════════

def build_master_dataset():
    print("=" * 65)
    print("TerraTrust — Master Dataset Builder (v2 Audit-Compliant)")
    print("Merging: Soil + Climate + Groundwater + NDVI")
    print("=" * 65)

    soil_df    = load_soil()
    climate_df = load_climate()
    gw_df      = load_groundwater()
    ndvi_df    = load_ndvi()

    # ── Step 1: Expand soil × year (multi-year schema) ─────────
    print("\n  [5/5] Building spatio-temporal master table...")
    years = [2019, 2020, 2021, 2022, 2023]
    expanded = []
    for _, row in soil_df.iterrows():
        for year in years:
            r = row.to_dict()
            r['year'] = year
            r['point_id_yr'] = f"{row['point_id']}_{year}"
            r['declared_crop'] = get_declared_crop(row['taluk'])
            expanded.append(r)
    master = pd.DataFrame(expanded)
    print(f"    Expanded to {len(master)} rows (soil × 5 years)")

    # ── Step 2: Merge climate ───────────────────────────────────
    climate_numeric = [
        c for c in [
            'avg_monthly_rainfall_mm', 'max_temp_c', 'min_temp_c',
            'avg_humidity_pct', 'avg_root_zone_wetness'
        ] if c in climate_df.columns
    ]
    climate_agg = {c: 'mean' for c in climate_numeric}
    if 'api_source' in climate_df.columns:
        climate_agg['api_source'] = 'first'

    climate_dedup = climate_df.groupby('point_id_yr', as_index=False).agg(climate_agg)
    print(f"    Climate dedup: {len(climate_df)} -> {len(climate_dedup)} rows (unique point_id_yr)")

    master = master.merge(climate_dedup, on='point_id_yr', how='left')
    print(f"    After climate merge: {len(master)} rows")

    # ── Step 3: Merge groundwater ───────────────────────────────
    if gw_df is not None:
        try:
            with open(os.path.join(BASE_DIR, 'src', 'taluk_dist_map.json'), 'r') as f:
                taluk_map = json.load(f)
        except Exception:
            taluk_map = {}

        master['district'] = master['taluk'].map(taluk_map).fillna('Unknown')
        gw_annual = (
            gw_df.groupby(['district', 'year'])['groundwater_depth_m']
            .mean()
            .reset_index()
        )
        master = master.merge(gw_annual, on=['district', 'year'], how='left')
        gw_source_df = gw_df[['district','year','source']].drop_duplicates(subset=['district','year'])
        master = master.merge(gw_source_df.rename(columns={'source':'gw_data_source'}),
                               on=['district','year'], how='left')
        missing_count = master['groundwater_depth_m'].isna().sum()
        master['gw_data_source'] = master['gw_data_source'].fillna('Missing after district-year merge')
        print(f"    Groundwater merged | {missing_count} rows missing groundwater_depth_m")
    else:
        master['groundwater_depth_m'] = np.nan
        master['gw_data_source'] = 'Missing groundwater source file'
        print("    Groundwater source missing; groundwater_depth_m left as NaN")

    # ── Step 4: Merge NDVI ──────────────────────────────────────
    if ndvi_df is not None:
        ndvi_annual = (
            ndvi_df.groupby(['taluk', 'year'])['ndvi_mean']
            .mean()
            .reset_index()
            .rename(columns={'ndvi_mean': 'ndvi_annual_mean'})
        )
        ndvi_std = (
            ndvi_df.groupby(['taluk', 'year'])['ndvi_mean']
            .std()
            .reset_index()
            .rename(columns={'ndvi_mean': 'ndvi_annual_std'})
        )
        ndvi_merged = ndvi_annual.merge(ndvi_std, on=['taluk','year'], how='left')

        if 'season' in ndvi_df.columns:
            kharif = ndvi_df[ndvi_df['season'] == 'Kharif'].groupby(['taluk','year'])['ndvi_mean'].mean().reset_index()
            kharif = kharif.rename(columns={'ndvi_mean': 'ndvi_kharif'})
            ndvi_merged = ndvi_merged.merge(kharif, on=['taluk','year'], how='left')

        master = master.merge(ndvi_merged, on=['taluk', 'year'], how='left')
        missing_ndvi = master['ndvi_annual_mean'].isna().sum()
        print(f"    NDVI merged | {missing_ndvi} rows missing ndvi_annual_mean")
    else:
        master['ndvi_annual_mean'] = np.nan
        master['ndvi_annual_std'] = np.nan
        print("    NDVI source missing; ndvi_annual_mean left as NaN")

    # Remove duplicates
    before_dedup = len(master)
    master = master.drop_duplicates()
    row_dups_removed = before_dedup - len(master)
    before_pid_dedup = len(master)
    master = master.drop_duplicates(subset=['point_id_yr'], keep='first')
    pid_dups_removed = before_pid_dedup - len(master)
    print(f"    De-dup check | removed {row_dups_removed} full-row dups, {pid_dups_removed} point_id_yr dups")

    # Drop rows missing critical physical data
    before_drop = len(master)
    master = master.dropna(subset=['groundwater_depth_m', 'ndvi_annual_mean'])
    dropped_rows = before_drop - len(master)
    print(f"    Dropped {dropped_rows} rows due to missing groundwater_depth_m/ndvi_annual_mean")

    # ══════════════════════════════════════════════════════════════
    # PHASE 1: DATA CLEANING
    # ══════════════════════════════════════════════════════════════
    print("\n  PHASE 1: Data Cleaning & Integrity Audit")
    master, audit_log = clean_and_audit(master)

    # ══════════════════════════════════════════════════════════════
    # PHASE 2: INTERACTION FEATURES
    # ══════════════════════════════════════════════════════════════
    print("\n  PHASE 2: Interaction Feature Engineering")
    master = add_interaction_features(master)

    # ══════════════════════════════════════════════════════════════
    # LEGACY SCORES (kept for credit_scorer heuristic, NOT for ML)
    # ══════════════════════════════════════════════════════════════
    print("\n    Computing legacy heuristic scores (for reference only)...")
    master['soil_suitability_score']   = master.apply(soil_suitability, axis=1)
    master['water_availability_score'] = master.apply(water_availability, axis=1)
    master['crop_health_ndvi']         = master['ndvi_annual_mean'].apply(ndvi_health)
    master['credit_score']             = master.apply(credit_score_heuristic, axis=1)
    master['loan_risk_category'] = pd.cut(
        master['credit_score'],
        bins=[0, 441, 459, 533, 577, 900],
        labels=['Very High Risk', 'High Risk', 'Moderate Risk', 'Low Risk', 'Very Low Risk']
    )

    # ── Save ───────────────────────────────────────────────────
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    master.to_csv(MASTER_CSV, index=False)

    # Save audit log
    if audit_log:
        pd.DataFrame(audit_log).to_csv(AUDIT_LOG_CSV, index=False)
        print(f"    Audit log saved: {AUDIT_LOG_CSV} ({len(audit_log)} entries)")

    # Define which columns are ML training features vs heuristic-only
    ml_training_features = [
        'clay_pct', 'sand_pct', 'silt_pct', 'pH', 'nitrogen_g_per_kg',
        'organic_carbon_dg_per_kg', 'bulk_density_cg_per_cm3',
        'avg_monthly_rainfall_mm', 'max_temp_c', 'min_temp_c',
        'avg_humidity_pct', 'avg_root_zone_wetness',
        'ndvi_annual_mean', 'ndvi_annual_std', 'groundwater_depth_m',
        # Interaction features (Phase 2)
        'soil_water_retention', 'aridity_index', 'vegetation_stress_index',
        'soil_fertility_index', 'water_table_pressure', 'thermal_stress',
        'sand_clay_ratio', 'ndvi_flagged'
    ]
    heuristic_only_columns = [
        'soil_suitability_score', 'water_availability_score',
        'credit_score', 'loan_risk_category', 'crop_health_ndvi'
    ]

    # Update provenance
    gw_source = gw_df['source'].iloc[0] if gw_df is not None else 'NASA POWER GWETROOT Model'
    ndvi_source_str = ndvi_df['data_source'].iloc[0] if (ndvi_df is not None and 'data_source' in ndvi_df.columns) else 'Landsat C2 L2 + Crop Calendar Model'

    provenance = {
        "project": "TerraTrust",
        "version": "v2_audit_compliant",
        "region": "Karnataka State, India",
        "total_records": len(master),
        "audit_fixes": {
            "phase_1_cleaning": [
                "Imputed missing bulk_density with taluk-median",
                "Filled missing ndvi_annual_std with 0 (single observation per year)",
                "Capped negative NDVI values to 0 and flagged in ndvi_flagged column"
            ],
            "phase_2_features": [
                "Added soil_water_retention (clay × organic_carbon / 100)",
                "Added aridity_index (rainfall / temp_range, FAO-inspired)",
                "Added vegetation_stress_index (NDVI × root_zone_wetness)",
                "Added soil_fertility_index ((N + OC/100) / (1 + |pH - 6.8|))",
                "Added water_table_pressure (rainfall / (gw_depth + 1))",
                "Added thermal_stress (|avg_temp - 27.5|)",
                "Added sand_clay_ratio (sand / clay)"
            ]
        },
        "ml_training_features": ml_training_features,
        "heuristic_only_columns": heuristic_only_columns,
        "data_sources": {
            "boundaries": {
                "source": "KGIS (kgis.ksrsac.in)",
                "files": ["District.shp", "Taluk.shp"]
            },
            "soil": {
                "source": "ISRIC SoilGrids v2.0",
                "url": "https://rest.isric.org/soilgrids/v2.0/",
                "note": "250m resolution global soil maps, real API extraction"
            },
            "climate": {
                "source": "NASA POWER v2.0",
                "url": "https://power.larc.nasa.gov/",
                "note": "Satellite-derived climate, 2019-2023, 0.5° grid"
            },
            "groundwater": {
                "source": gw_source,
                "note": "District-level seasonal groundwater depth, physics model from GWETROOT"
            },
            "satellite_ndvi": {
                "source": ndvi_source_str,
                "url": "https://planetarycomputer.microsoft.com/",
                "note": "Landsat Collection 2 Level-2 NDVI per taluk centroid, 2019-2023"
            }
        },
        "taluks": sorted(master['taluk'].dropna().unique().tolist()),
        "years": [2019, 2020, 2021, 2022, 2023],
        "features": list(master.columns)
    }
    with open(os.path.join(PROCESSED_DIR, 'data_provenance.json'), 'w') as f:
        json.dump(provenance, f, indent=2)

    print(f"\n{'='*65}")
    print(f"MASTER DATASET COMPLETE (v2 Audit-Compliant)")
    print(f"  Output:   {MASTER_CSV}")
    print(f"  Rows:     {len(master)}")
    print(f"  Columns:  {len(master.columns)}")
    print(f"  Taluks:   {master['taluk'].nunique()}")
    print(f"  Years:    {sorted(master['year'].unique())}")
    print(f"\n  ML Training Features ({len(ml_training_features)}):")
    print(f"    {', '.join(ml_training_features)}")
    print(f"\n  Heuristic-Only Columns (NOT for ML):")
    print(f"    {', '.join(heuristic_only_columns)}")
    print(f"\nCredit Score distribution:")
    print(master['loan_risk_category'].value_counts().to_string())
    print(f"\nSample rows:")
    sample_cols = ['taluk','year','ndvi_annual_mean','ndvi_flagged','groundwater_depth_m',
                   'soil_water_retention','aridity_index','vegetation_stress_index',
                   'credit_score','loan_risk_category']
    print(master[sample_cols].head(6).to_string(index=False))
    print("=" * 65)
    return master


if __name__ == "__main__":
    build_master_dataset()
