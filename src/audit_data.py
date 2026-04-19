"""Complete data audit for TerraTrust - checks every CSV for completeness and authenticity."""
import pandas as pd
import numpy as np
import os, sys, io

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

print("=" * 70)
print("COMPLETE DATA AUDIT - TerraTrust")
print("=" * 70)

# ========== 1. karnataka_groundwater.csv ==========
print("\n" + "-" * 50)
print("FILE: karnataka_groundwater.csv")
print("-" * 50)
gw = pd.read_csv("data/kgis_tabular/karnataka_groundwater.csv")
print(f"Rows: {len(gw)}")
print(f"Columns: {list(gw.columns)}")
districts_in_gw = gw["district"].nunique() if "district" in gw.columns else 0
print(f"Unique districts: {districts_in_gw}")
print(f"Districts list: {sorted(gw['district'].unique().tolist()) if 'district' in gw.columns else 'N/A'}")
print(f"Years: {sorted(gw['year'].unique().tolist()) if 'year' in gw.columns else 'N/A'}")
print(f"Missing values per column:")
for col in gw.columns:
    missing = gw[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing}/{len(gw)} ({100*missing/len(gw):.1f}%)")
if gw.isnull().sum().sum() == 0:
    print("  None - all columns complete")
print(f"Data source column: {gw['gw_data_source'].unique().tolist() if 'gw_data_source' in gw.columns else 'N/A'}")
print(f"\nSample rows:")
print(gw.head(3).to_string())

# ========== 2. karnataka_soil_data.csv ==========
print("\n" + "-" * 50)
print("FILE: karnataka_soil_data.csv")
print("-" * 50)
soil = pd.read_csv("data/kgis_tabular/karnataka_soil_data.csv")
print(f"Rows: {len(soil)}")
print(f"Columns: {list(soil.columns)}")
print(f"Unique taluks: {soil['taluk'].nunique() if 'taluk' in soil.columns else 'N/A'}")
print(f"Missing values per column:")
for col in soil.columns:
    missing = soil[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing}/{len(soil)} ({100*missing/len(soil):.1f}%)")
if soil.isnull().sum().sum() == 0:
    print("  None - all columns complete")
print(f"API source: {soil['api_source'].value_counts().to_dict() if 'api_source' in soil.columns else 'N/A'}")
# Check for rows that are completely empty (only have point_id, taluk, lat, lon)
empty_soil_rows = soil[soil[['clay_pct', 'sand_pct', 'pH']].isnull().all(axis=1)]
print(f"Rows with ALL soil values missing: {len(empty_soil_rows)}")
print(f"\nSample rows:")
print(soil.head(3).to_string())

# ========== 3. karnataka_climate_data.csv ==========
print("\n" + "-" * 50)
print("FILE: karnataka_climate_data.csv")
print("-" * 50)
clim = pd.read_csv("data/kgis_tabular/karnataka_climate_data.csv")
print(f"Rows: {len(clim)}")
print(f"Columns: {list(clim.columns)}")
print(f"Missing values per column:")
for col in clim.columns:
    missing = clim[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing}/{len(clim)} ({100*missing/len(clim):.1f}%)")
if clim.isnull().sum().sum() == 0:
    print("  None - all columns complete")
print(f"API source: {clim['api_source'].value_counts().to_dict() if 'api_source' in clim.columns else 'N/A'}")
# Check for duplicate values (sign of copy-paste/fake)
if 'avg_monthly_rainfall_mm' in clim.columns:
    unique_rainfall = clim['avg_monthly_rainfall_mm'].nunique()
    print(f"Unique rainfall values: {unique_rainfall} out of {len(clim)} rows")
    if unique_rainfall < 20:
        print("  [WARNING] Very few unique rainfall values - could be regional grid snap (not per-point)")
print(f"\nSample rows:")
print(clim.head(3).to_string())

# ========== 4. taluk_district_map.csv ==========
print("\n" + "-" * 50)
print("FILE: taluk_district_map.csv")
print("-" * 50)
tdm = pd.read_csv("data/kgis_tabular/taluk_district_map.csv")
print(f"Rows: {len(tdm)}")
if len(tdm) == 0:
    print("  [CRITICAL] FILE IS EMPTY!")
else:
    print(f"Columns: {list(tdm.columns)}")
    if "KGISDist_1" in tdm.columns:
        print(f"Unique districts: {tdm['KGISDist_1'].nunique()}")
    print(f"\nSample rows:")
    print(tdm.head(5).to_string())

# ========== 5. karnataka_ndvi.csv ==========
print("\n" + "-" * 50)
print("FILE: satellite/karnataka_ndvi_real.csv")
print("-" * 50)
ndvi = pd.read_csv("data/satellite/karnataka_ndvi_real.csv")
print(f"Rows: {len(ndvi)}")
print(f"Columns: {list(ndvi.columns)}")
print(f"Unique taluks: {ndvi['taluk'].nunique() if 'taluk' in ndvi.columns else 'N/A'}")
print(f"Missing values per column:")
for col in ndvi.columns:
    missing = ndvi[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing}/{len(ndvi)} ({100*missing/len(ndvi):.1f}%)")
if ndvi.isnull().sum().sum() == 0:
    print("  None - all columns complete")
# Check if NDVI values look realistic
if 'ndvi_mean' in ndvi.columns:
    print(f"NDVI range: min={ndvi['ndvi_mean'].min():.4f}, max={ndvi['ndvi_mean'].max():.4f}, mean={ndvi['ndvi_mean'].mean():.4f}")
    unique_ndvi = ndvi['ndvi_mean'].nunique()
    print(f"Unique NDVI values: {unique_ndvi} out of {len(ndvi)} rows")
print(f"Data source: {ndvi['data_source'].unique().tolist() if 'data_source' in ndvi.columns else 'N/A'}")
print(f"\nSample rows:")
print(ndvi.head(3).to_string())

# ========== 7. karnataka_master_dataset.csv ==========
print("\n" + "-" * 50)
print("FILE: processed/karnataka_master_dataset.csv")
print("-" * 50)
master = pd.read_csv("data/processed/karnataka_master_dataset.csv")
print(f"Rows: {len(master)}")
print(f"Columns: {list(master.columns)}")
print(f"Unique districts: {master['district'].nunique() if 'district' in master.columns else 'N/A'}")
print(f"Missing values per column:")
for col in master.columns:
    missing = master[col].isnull().sum()
    if missing > 0:
        print(f"  {col}: {missing}/{len(master)} ({100*missing/len(master):.1f}%)")
if master.isnull().sum().sum() == 0:
    print("  None - all columns complete")

# Check for random/fake columns
if 'declared_crop' in master.columns:
    crop_dist = master['declared_crop'].value_counts()
    print(f"\nDeclared crop distribution:")
    for crop, count in crop_dist.items():
        print(f"  {crop}: {count} ({100*count/len(master):.1f}%)")
    # Uniform distribution = random
    if abs(crop_dist.max() - crop_dist.min()) / len(master) < 0.05:
        print("  [WARNING] Near-uniform distribution - likely RANDOMLY GENERATED")

if 'annual_income' in master.columns:
    print(f"\nFinancial data:")
    print(f"  annual_income: min={master['annual_income'].min()}, max={master['annual_income'].max()}, mean={master['annual_income'].mean():.0f}")
    print(f"  loan_principal: min={master['loan_principal'].min()}, max={master['loan_principal'].max()}")
    print(f"  repayment_status distribution: {master['repayment_status'].value_counts().to_dict() if 'repayment_status' in master.columns else 'N/A'}")

print(f"\nSample rows:")
print(master.head(2).to_string())

print("\n" + "=" * 70)
print("AUDIT COMPLETE")
print("=" * 70)
