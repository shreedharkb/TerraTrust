"""
TerraTrust Satellite Pipeline
==============================
Downloads genuine Sentinel-2 satellite imagery metadata from Microsoft Planetary Computer
and computes NDVI / NDWI indices for crop health and water assessment.

Data Source: Microsoft Planetary Computer (STAC API)
    - Uses real Sentinel-2 Level-2A imagery
    - No API key required for catalog searches
    - Imagery is from ESA Copernicus program

ALL DATA IS 100% GENUINE — scene IDs, dates, and cloud cover come directly
from the Sentinel-2 catalog. NDVI estimates are derived from real scene metadata
and Davangere's known crop calendar.
"""

import os
import sys
import json
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pystac_client
import planetary_computer
import odc.stac
import xarray as xr

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *


# ============================================================
# Sentinel-2 STAC Search
# ============================================================

def search_sentinel2_scenes(bbox, start_date="2024-01-01", end_date="2024-12-31", max_cloud=20):
    """
    Search for Sentinel-2 scenes covering the given bounding box using
    Microsoft Planetary Computer's STAC API.
    
    Args:
        bbox: [minx, miny, maxx, maxy] in EPSG:4326
        start_date: Start of date range (YYYY-MM-DD)
        end_date: End of date range (YYYY-MM-DD)
        max_cloud: Maximum cloud cover percentage
    
    Returns:
        List of pystac.Item representing the scenes
    """
    print(f"\n  Searching Sentinel-2 scenes for Davangere...")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Max cloud cover: {max_cloud}%")
    print(f"  Bounding box: {[round(b, 4) for b in bbox]}")
    
    # Verify bbox is in WGS84 range
    if bbox[0] > 180 or bbox[1] > 90:
        print(f"  ERROR: Bounding box values too large for WGS84!")
        print(f"  Expected: lon in [-180,180], lat in [-90,90]")
        return []

    try:
        catalog = pystac_client.Client.open(
            PLANETARY_COMPUTER_STAC,
            modifier=planetary_computer.sign_inplace,
        )
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{start_date}/{end_date}",
            query={"eo:cloud_cover": {"lt": max_cloud}},
            sortby="datetime",
            limit=50,
        )
        items = list(search.items())
        print(f"  Found {len(items)} Sentinel-2 scenes")
        
        # Print sample scene IDs for verification
        if items:
            for s in items[:3]:
                print(f"    Scene: {s.id} | Date: {s.datetime.strftime('%Y-%m-%d')} | Cloud: {s.properties.get('eo:cloud_cover', 'N/A')}%")
            if len(items) > 3:
                print(f"    ... and {len(items) - 3} more scenes")
        
        return items
    except Exception as e:
        print(f"  STAC search error: {e}")
        return []


def extract_scene_info(scene):
    """Extract key info from a Sentinel-2 scene."""
    props = scene.get('properties', {})
    assets = scene.get('assets', {})
    
    result = {
        'scene_id': scene.get('id', 'unknown'),
        'datetime': props.get('datetime', ''),
        'cloud_cover': props.get('eo:cloud_cover', None),
        'platform': props.get('platform', ''),
        'mgrs_tile': props.get('s2:mgrs_tile', ''),
    }
    
    # Check band availability
    for band_name in ['B02', 'B03', 'B04', 'B08', 'B11']:
        band_key = band_name.lower()
        if band_key not in assets:
            band_key = band_name
        result[f'{band_name}_available'] = band_key in assets
    
    return result


# ============================================================
# NDVI / NDWI Computation from Scene Statistics
# ============================================================

def compute_ndvi_from_scenes(scenes, bbox=None):
    """
    Compute actual NDVI and NDWI statistics for each scene.
    
    NDVI = (NIR - Red) / (NIR + Red) = (B08 - B04) / (B08 + B04)
    NDWI = (Green - NIR) / (Green + NIR) = (B03 - B08) / (B03 + B08)
    
    Downloads the exact pixel arrays over the provided bbox using odc.stac and
    calculates real aggregate data, skipping clouds via the SCL indicator mask.
    """
    print("\n" + "=" * 60)
    print("Computing Real NDVI & NDWI from Sentinel-2 Pixels")
    print("=" * 60)
    print("  NDVI = (B08_NIR - B04_Red) / (B08_NIR + B04_Red)")
    print("  NDWI = (B03_Green - B08_NIR) / (B03_Green + B08_NIR)")
    
    if not scenes or bbox is None:
        return pd.DataFrame()
        
    ndvi_records = []
    
    # Sign items directly one by one for Microsoft Planetary Computer
    signed_items = [planetary_computer.sign(item) for item in scenes]

    try:
        print(f"  Loading pixel data for {len(signed_items)} scenes... this may take a moment.")
        # Load the data using odc.stac. Resolution is set slightly lower to prevent RAM issues on large bboxes.
        ds = odc.stac.load(
            signed_items,
            bands=("B04", "B08", "B03", "SCL"),
            bbox=bbox,
            resolution=0.001,
            chunks={"time": 1, "x": 1000, "y": 1000}
        )
    except Exception as e:
        print(f"  Error loading remote raster data: {e}")
        return pd.DataFrame()
        
    for i, dt_val in enumerate(ds.time.values):
        dt = pd.to_datetime(dt_val)
        scene_date = dt.strftime('%Y-%m-%d')
        month = dt.month
        item = signed_items[i]
        
        img = ds.isel(time=i)
        
        # SCL valid pixels: 4 (Vegetation), 5 (Bare Soil), 6 (Water)
        # Cloud pixels are 8, 9, 10
        valid_mask = (img.SCL >= 4) & (img.SCL <= 7)
        
        # Convert to float and scale (L2A needs scaling by 10000)
        red = img.B04.where(valid_mask).astype(float) / 10000.0
        nir = img.B08.where(valid_mask).astype(float) / 10000.0
        green = img.B03.where(valid_mask).astype(float) / 10000.0
        
        # True pixel calculation
        ndvi_arr = (nir - red) / (nir + red + 1e-8)
        ndwi_arr = (green - nir) / (green + nir + 1e-8)
        
        ndvi_mean = float(ndvi_arr.mean().values)
        if np.isnan(ndvi_mean):
            print(f"  {scene_date} | Skipped (100% Cloud covered or missing data)")
            continue
            
        ndvi_min = float(ndvi_arr.min().values)
        ndvi_max = float(ndvi_arr.max().values)
        ndwi_mean = float(ndwi_arr.mean().values)
        
        record = {
            'scene_id': item.id,
            'date': scene_date,
            'month': month,
            'cloud_cover_pct': item.properties.get('eo:cloud_cover', 100),
            'ndvi_mean': round(ndvi_mean, 4),
            'ndvi_min': round(ndvi_min, 4),
            'ndvi_max': round(ndvi_max, 4),
            'ndwi_mean': round(ndwi_mean, 4),
            'crop_health': classify_ndvi(ndvi_mean),
            'growth_stage': estimate_growth_stage(ndvi_mean, month),
            'platform': item.properties.get('platform', 'Sentinel-2A'),
            'data_source': 'Real Raster via planetary_computer & odc.stac',
        }
        ndvi_records.append(record)
        
        print(f"  {scene_date} | NDVI={record['ndvi_mean']:.3f} "
              f"({record['crop_health']}) | NDWI={record['ndwi_mean']:.3f} "
              f"| Cloud={record['cloud_cover_pct']}%")
    
    return pd.DataFrame(ndvi_records)


def classify_ndvi(ndvi):
    """Classify NDVI value into crop health category."""
    for label, (low, high) in NDVI_THRESHOLDS.items():
        if low <= ndvi < high:
            return label
    return "Unknown"


def estimate_growth_stage(ndvi, month):
    """Estimate crop growth stage from NDVI value and month."""
    for stage, (low, high) in GROWTH_STAGES.items():
        if low <= ndvi < high:
            return stage
    if ndvi < 0.1:
        return "Fallow"
    return "Peak Growth"


# ============================================================
# Historical NDVI Time Series
# ============================================================

def build_ndvi_timeseries(bbox, years=None):
    """
    Build a multi-year NDVI time series for historical trend analysis.
    Uses genuine Sentinel-2 scene catalog from Planetary Computer.
    """
    if years is None:
        years = [2022, 2023, 2024]
    
    print("\n" + "=" * 60)
    print("Building Historical NDVI Time Series")
    print("=" * 60)
    print(f"  Years: {years}")
    
    all_records = []
    
    for year in years:
        print(f"\n  --- Year {year} ---")
        scenes = search_sentinel2_scenes(
            bbox=bbox,
            start_date=f"{year}-01-01",
            end_date=f"{year}-12-31",
            max_cloud=30
        )
        
        if scenes:
            # Take up to 12 scenes per year (roughly 1 per month)
            selected = scenes[:12]
            ndvi_df = compute_ndvi_from_scenes(selected, bbox=bbox)
            if not ndvi_df.empty:
                ndvi_df['year'] = year
                all_records.append(ndvi_df)
        
        import time
        time.sleep(1)  # API courtesy
    
    if all_records:
        timeseries = pd.concat(all_records, ignore_index=True)
        ts_path = os.path.join(SATELLITE_DIR, "ndvi_timeseries.csv")
        timeseries.to_csv(ts_path, index=False)
        print(f"\n  Saved NDVI time series ({len(timeseries)} records) to: {ts_path}")
        return timeseries
    else:
        print("  No time series data collected")
        return pd.DataFrame()


# ============================================================
# Satellite Summary Report
# ============================================================

def generate_satellite_summary(ndvi_df):
    """Generate a satellite assessment summary."""
    print("\n" + "=" * 60)
    print("Generating Satellite Assessment Summary")
    print("=" * 60)
    
    if ndvi_df.empty:
        print("  No NDVI data available for summary")
        return {}
    
    summary = {
        'avg_ndvi': round(float(ndvi_df['ndvi_mean'].mean()), 4),
        'max_ndvi': round(float(ndvi_df['ndvi_mean'].max()), 4),
        'min_ndvi': round(float(ndvi_df['ndvi_mean'].min()), 4),
        'ndvi_std': round(float(ndvi_df['ndvi_mean'].std()), 4),
        'num_scenes': int(len(ndvi_df)),
        'dominant_health': str(ndvi_df['crop_health'].mode().iloc[0]) if len(ndvi_df) > 0 else 'Unknown',
        'avg_cloud_cover': round(float(ndvi_df['cloud_cover_pct'].mean()), 1),
        'date_range': f"{ndvi_df['date'].min()} to {ndvi_df['date'].max()}",
        'data_source': 'Microsoft Planetary Computer / ESA Copernicus Sentinel-2',
    }
    
    # NDVI trend
    if len(ndvi_df) > 2:
        x = np.arange(len(ndvi_df))
        slope = np.polyfit(x, ndvi_df['ndvi_mean'].values, 1)[0]
        summary['ndvi_trend_slope'] = round(float(slope), 6)
        summary['trend_direction'] = 'Improving' if slope > 0.001 else ('Declining' if slope < -0.001 else 'Stable')
    else:
        summary['ndvi_trend_slope'] = 0
        summary['trend_direction'] = 'Insufficient Data'
    
    print(f"  Average NDVI: {summary['avg_ndvi']}")
    print(f"  Health Status: {summary['dominant_health']}")
    print(f"  Trend: {summary['trend_direction']} (slope={summary['ndvi_trend_slope']})")
    print(f"  Scenes Analyzed: {summary['num_scenes']}")
    
    summary_path = os.path.join(SATELLITE_DIR, "satellite_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"  Saved summary to: {summary_path}")
    
    return summary


# ============================================================
# Main Execution
# ============================================================

def run_satellite_pipeline(bbox):
    """Execute the complete satellite data pipeline."""
    print("\n" + "=" * 60)
    print("  TerraTrust Satellite Pipeline")
    print("  Source: Microsoft Planetary Computer / ESA Sentinel-2")
    print("  All imagery metadata is GENUINE Copernicus data")
    print("=" * 60)
    
    # Verify bbox is in WGS84
    if bbox[0] > 180 or bbox[1] > 90:
        print(f"  ERROR: Bounding box {bbox} is not in WGS84!")
        print(f"  This looks like projected coordinates. Cannot proceed.")
        return pd.DataFrame(), {}
    
    print(f"\n  BBox (WGS84): {[round(b, 4) for b in bbox]}")
    
    # Search for recent Sentinel-2 scenes
    scenes = search_sentinel2_scenes(bbox, start_date="2024-01-01", end_date="2024-12-31")
    
    if not scenes:
        print("  No scenes found with <20% cloud. Trying <40%...")
        scenes = search_sentinel2_scenes(bbox, start_date="2023-06-01", end_date="2024-12-31", max_cloud=40)
    
    if not scenes:
        print("  No Sentinel-2 scenes found for this area")
        return pd.DataFrame(), {}
    
    # Compute NDVI from scenes
    ndvi_df = compute_ndvi_from_scenes(scenes[:24], bbox=bbox)
    
    # Save NDVI data
    ndvi_path = os.path.join(SATELLITE_DIR, "ndvi_data.csv")
    ndvi_df.to_csv(ndvi_path, index=False)
    print(f"\n  Saved NDVI data to: {ndvi_path}")
    
    # Build historical time series
    timeseries = build_ndvi_timeseries(bbox)
    
    # Generate summary
    summary = generate_satellite_summary(ndvi_df)
    
    print("\n" + "=" * 60)
    print("  SATELLITE PIPELINE COMPLETE")
    print("=" * 60)
    
    return ndvi_df, summary


if __name__ == "__main__":
    import geopandas as gpd
    
    gdf = gpd.read_file(DISTRICT_SHP)
    
    # Reproject to WGS84
    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)
    
    for col in gdf.columns:
        if gdf[col].dtype == 'object':
            mask = gdf[col].str.contains('|'.join(TARGET_DISTRICT_ALT), case=False, na=False)
            if mask.any():
                davangere = gdf[mask]
                bbox = list(davangere.geometry.total_bounds)
                print(f"Davangere BBox (WGS84): {bbox}")
                run_satellite_pipeline(bbox)
                break
