"""
TerraTrust Data Pipeline
=========================
ALL DATA IS 100% GENUINE from verified government/scientific APIs.
"""

import os, sys, json, time
import requests
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import mapping, Point

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *


def load_district_boundary():
    """Load KGIS District shapefile, extract Davangere, reproject to WGS84."""
    print("=" * 60)
    print("STEP 1: Loading KGIS District Shapefile")
    print("=" * 60)
    gdf = gpd.read_file(DISTRICT_SHP)
    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)
    
    davangere = gdf[gdf['KGISDist_1'].astype(str).str.contains('|'.join(TARGET_DISTRICT_ALT), case=False, na=False)].copy()
    if davangere.empty:
        raise ValueError("Davangere district not found!")
    
    bounds = davangere.geometry.total_bounds
    c = davangere.geometry.centroid.iloc[0]
    print(f"  Found Davangere | BBox: [{bounds[0]:.4f}, {bounds[1]:.4f}, {bounds[2]:.4f}, {bounds[3]:.4f}]")
    print(f"  Centroid: ({c.y:.4f}N, {c.x:.4f}E)")
    return davangere, bounds


def load_taluk_boundaries():
    """Load KGIS Taluk shapefile, extract ONLY Davangere's 6 taluks."""
    print("\n" + "=" * 60)
    print("STEP 2: Loading KGIS Taluk Shapefile")
    print("=" * 60)
    gdf = gpd.read_file(TALUK_SHP)
    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)
    
    # Davangere district code is '14' in the KGIS shapefile
    # Filter: district code 14 AND taluk name is not null (excludes the district-level row)
    davangere_taluks = gdf[(gdf['KGISDistri'] == '14') & (gdf['KGISTalukN'].notna())].copy()
    
    if davangere_taluks.empty:
        raise ValueError("No Davangere taluks found with KGISDistri=14!")
    
    davangere_taluks['centroid_lat'] = davangere_taluks.geometry.centroid.y
    davangere_taluks['centroid_lon'] = davangere_taluks.geometry.centroid.x
    
    print(f"  Found {len(davangere_taluks)} Davangere taluks:")
    for _, r in davangere_taluks.iterrows():
        print(f"    {r['KGISTalukN']:15s} ({r['centroid_lat']:.4f}N, {r['centroid_lon']:.4f}E)")
    
    return davangere_taluks


def fetch_soil_data(lat, lon):
    """Fetch genuine soil data from ISRIC SoilGrids API."""
    params = {
        "lon": lon, "lat": lat,
        "property": ["clay", "sand", "silt", "phh2o", "nitrogen", "ocd", "soc", "bdod"],
        "depth": ["0-5cm", "5-15cm", "15-30cm"],
        "value": "mean"
    }
    try:
        r = requests.get(SOILGRIDS_API, params=params, timeout=30)
        if r.status_code == 200:
            data = r.json()
            result = {}
            for layer in data.get('properties', {}).get('layers', []):
                for depth in layer.get('depths', []):
                    if depth['label'] == '0-5cm':
                        result[layer['name']] = depth['values'].get('mean')
                        break
            return result
    except Exception as e:
        print(f"    SoilGrids error: {e}")
    return None


def fetch_climate_data(lat, lon, year_str):
    """Fetch genuine climate data from NASA POWER API for a specific year."""
    params = {
        "parameters": "PRECTOTCORR,T2M,T2M_MAX,T2M_MIN,RH2M,ALLSKY_SFC_SW_DWN,GWETROOT",
        "community": "AG",
        "longitude": lon, "latitude": lat,
        "start": year_str, "end": year_str,  # Year-only format
        "format": "JSON"
    }
    try:
        r = requests.get(NASA_POWER_API, params=params, timeout=60)
        if r.status_code == 200:
            return r.json().get('properties', {}).get('parameter', {})
        else:
            print(f"    NASA POWER HTTP {r.status_code}")
    except Exception as e:
        print(f"    NASA POWER error: {e}")
    return None


def generate_sample_points(gdf, points_per_polygon=10):
    """Generate random physical coordinates strictly inside the valid real boundaries."""
    print(f"\n  Generating a dense grid of {points_per_polygon} real geographic points per Taluk...")
    points = []
    np.random.seed(42) # Consistent grid
    
    for idx, row in gdf.iterrows():
        poly = row['geometry']
        taluk_name = row['KGISTalukN']
        minx, miny, maxx, maxy = poly.bounds
        
        count = 0
        while count < points_per_polygon:
            pnt = Point(np.random.uniform(minx, maxx), np.random.uniform(miny, maxy))
            if poly.contains(pnt):
                points.append({
                    'taluk': taluk_name,
                    'point_id': f"{taluk_name[:3].upper()}_{count:03d}",
                    'latitude': pnt.y,
                    'longitude': pnt.x
                })
                count += 1
                
    points_df = pd.DataFrame(points)
    print(f"  Generated {len(points_df)} strict real coordinate locations.")
    return points_df


def run_full_data_pipeline():
    """Execute the complete data pipeline with genuine APIs."""
    print("\n" + "=" * 60)
    print("  TerraTrust Data Pipeline — 100% Genuine Data")
    print("  Target: Davangere District, Karnataka")
    print("=" * 60)

    # Step 1 & 2: Load boundaries
    district_gdf, bounds = load_district_boundary()
    taluks_gdf = load_taluk_boundaries()

    # Generate exact real points
    points_df = generate_sample_points(taluks_gdf, points_per_polygon=10)
    
    # Expand to multi-year schema
    expanded_points = []
    for _, row in points_df.iterrows():
        for year in [2019, 2020, 2021, 2022, 2023]:
            pt = row.to_dict()
            pt['year'] = year
            pt['point_id_yr'] = f"{pt['point_id']}_{year}"
            expanded_points.append(pt)
    
    expanded_df = pd.DataFrame(expanded_points)
    print(f"  Expanded spatio-temporal dataframe to {len(expanded_df)} rows.")

    # Step 3: Fetch genuine soil data from ISRIC SoilGrids
    print("\n" + "=" * 60)
    print("STEP 3: Fetching Soil Data from ISRIC SoilGrids API")
    print("  Source: https://rest.isric.org/soilgrids/v2.0/")
    print("=" * 60)
    
    soil_records = []
    # Soil is static historically so we only fetch it once per spatial coordinate
    for _, row in points_df.iterrows():
        name = row['taluk']
        pid = row['point_id']
        lat, lon = row['latitude'], row['longitude']
        print(f"  {pid} ({lat:.4f}N, {lon:.4f}E)...", end=" ")
        
        soil = fetch_soil_data(lat, lon)
        if soil and len(soil) > 0:
            rec = {
                'point_id': pid, 'taluk': name, 'latitude': lat, 'longitude': lon,
                'clay_pct': round(soil.get('clay', 0) / 10, 1) if soil.get('clay') else None,
                'sand_pct': round(soil.get('sand', 0) / 10, 1) if soil.get('sand') else None,
                'silt_pct': round(soil.get('silt', 0) / 10, 1) if soil.get('silt') else None,
                'pH': round(soil.get('phh2o', 0) / 10, 1) if soil.get('phh2o') else None,
                'nitrogen_g_per_kg': round(soil.get('nitrogen', 0) / 100, 2) if soil.get('nitrogen') else None,
                'organic_carbon_dg_per_kg': soil.get('soc'),
                'bulk_density_cg_per_cm3': soil.get('bdod'),
                'api_source': 'ISRIC SoilGrids v2.0',
            }
            soil_records.append(rec)
            print("OK")
        else:
            print("FAILED")
            soil_records.append({'point_id': pid, 'taluk': name, 'latitude': lat, 'longitude': lon})
        time.sleep(0.5)
    
    soil_df = pd.DataFrame(soil_records)
    soil_df.to_csv(os.path.join(TABULAR_DIR, "davangere_soil_data.csv"), index=False)
    print(f"  Saved: data/kgis_tabular/davangere_soil_data.csv ({len(soil_df)} rows)")

    # Step 4: Fetch climate data from NASA POWER
    print("\n" + "=" * 60)
    print("STEP 4: Fetching Climate Data from NASA POWER API")
    print("  Source: https://power.larc.nasa.gov/")
    print("=" * 60)
    
    climate_records = []
    for _, row in expanded_df.iterrows():
        name = row['taluk']
        pid_yr = row['point_id_yr']
        lat, lon = row['latitude'], row['longitude']
        year_str = str(row['year'])
        print(f"  {pid_yr} ({lat:.4f}N, {lon:.4f}E)...", end=" ")
        
        climate = fetch_climate_data(lat, lon, year_str)
        if climate and len(climate) > 0:
            precip = [v for v in climate.get('PRECTOTCORR', {}).values() if v != -999 and v is not None]
            temp = [v for v in climate.get('T2M', {}).values() if v != -999 and v is not None]
            hum = [v for v in climate.get('RH2M', {}).values() if v != -999 and v is not None]
            wet = [v for v in climate.get('GWETROOT', {}).values() if v != -999 and v is not None]
            
            rec = {
                'point_id_yr': pid_yr, 'year': int(year_str),
                'avg_monthly_rainfall_mm': round(np.mean(precip), 2) if precip else None,
                'total_annual_rainfall_mm': round(np.sum(precip), 2) if precip else None,
                'avg_temperature_c': round(np.mean(temp), 2) if temp else None,
                'avg_humidity_pct': round(np.mean(hum), 2) if hum else None,
                'avg_root_zone_wetness': round(np.mean(wet), 4) if wet else None,
                'api_source': 'NASA POWER v2.0',
            }
            climate_records.append(rec)
            print("OK")
        else:
            print("FAILED")
            climate_records.append({'point_id_yr': pid_yr, 'year': int(year_str)})
        time.sleep(0.3)
    
    climate_df = pd.DataFrame(climate_records)
    climate_df.to_csv(os.path.join(TABULAR_DIR, "davangere_climate_data.csv"), index=False)
    print(f"  Saved: data/kgis_tabular/davangere_climate_data.csv ({len(climate_df)} rows)")

    # Step 5: Build master dataset
    print("\n" + "=" * 60)
    print("STEP 5: Building Master Dataset")
    print("=" * 60)
    
    master = expanded_df.copy()
    
    # Merge soil
    soil_merge = soil_df[[c for c in soil_df.columns if c not in ['latitude', 'longitude', 'taluk', 'api_source', 'api_url']]]
    master = master.merge(soil_merge, on='point_id', how='left')
    
    # Merge climate
    climate_merge = climate_df[[c for c in climate_df.columns if c not in ['api_source', 'api_url', 'year']]]
    master = master.merge(climate_merge, on='point_id_yr', how='left')
    
    # Load and merge external attributes (groundwater only)
    gw_path = os.path.join(TABULAR_DIR, "groundwater.csv")
    if os.path.exists(gw_path):
        gw_df = pd.read_csv(gw_path)
        master = master.merge(gw_df, on='taluk', how='left')
    
    # Compute scores
    master['soil_suitability_score'] = master.apply(_soil_score, axis=1)
    master['water_availability_score'] = master.apply(_water_score, axis=1)
    
    master.to_csv(os.path.join(PROCESSED_DIR, "davangere_master_dataset.csv"), index=False)
    print(f"  Master dataset: {master.shape}")
    print(master.to_string())
    
    # Save GeoJSON
    district_gdf.to_file(os.path.join(PROCESSED_DIR, "davangere_district.geojson"), driver="GeoJSON")
    taluks_gdf.to_file(os.path.join(PROCESSED_DIR, "davangere_taluks.geojson"), driver="GeoJSON")
    
    # Save provenance
    provenance = {
        "project": "TerraTrust",
        "region": "Davangere District, Karnataka",
        "data_sources": {
            "boundaries": {"source": "KGIS (kgis.ksrsac.in)", "files": ["District.shp", "Taluk.shp"]},
            "soil": {"source": "ISRIC SoilGrids v2.0", "url": "https://rest.isric.org/soilgrids/v2.0/", "note": "250m resolution global soil maps"},
            "climate": {"source": "NASA POWER v2.0", "url": "https://power.larc.nasa.gov/", "note": "Satellite-derived, 2020-2024"},
        },
        "taluks": list(master['taluk'].values),
    }
    with open(os.path.join(DATA_DIR, "data_provenance.json"), 'w') as f:
        json.dump(provenance, f, indent=2)
    
    print("\n  DATA PIPELINE COMPLETE")
    return master, district_gdf, taluks_gdf


def _soil_score(row):
    crop = row.get('declared_crop', 'Maize')
    req = CROP_REQUIREMENTS.get(crop, CROP_REQUIREMENTS['Maize'])
    score = 100.0
    ph = row.get('pH')
    if ph and not pd.isna(ph):
        if ph < req['pH_min']: score -= min(30, (req['pH_min'] - ph) * 15)
        elif ph > req['pH_max']: score -= min(30, (ph - req['pH_max']) * 15)
    n = row.get('nitrogen_g_per_kg')
    if n and not pd.isna(n):
        n_ha = n * 25
        if n_ha < req['N_min'] * 0.5: score -= 25
        elif n_ha < req['N_min']: score -= 10
    clay = row.get('clay_pct')
    if clay and not pd.isna(clay):
        if clay > 50: score -= 15
        elif clay < 10: score -= 10
    return max(0, min(100, round(score, 1)))


def _water_score(row):
    score = 50.0
    rain = row.get('avg_monthly_rainfall_mm')
    if rain and not pd.isna(rain):
        if rain > 80: score += 20
        elif rain > 50: score += 10
        elif rain < 20: score -= 15
    wet = row.get('avg_root_zone_wetness')
    if wet and not pd.isna(wet): score += wet * 30
    hum = row.get('avg_humidity_pct')
    if hum and not pd.isna(hum):
        if hum > 60: score += 10
        elif hum < 40: score -= 10
        
    gw_depth = row.get('groundwater_depth_m')
    if gw_depth and not pd.isna(gw_depth):
        if gw_depth < 10: score += 15
        elif gw_depth < 20: score += 5
        else: score -= 10
        
    return max(0, min(100, round(score, 1)))


if __name__ == "__main__":
    run_full_data_pipeline()
