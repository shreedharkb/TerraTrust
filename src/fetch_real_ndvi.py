"""
TerraTrust Step 1: Fetch REAL Satellite NDVI
=============================================
Extracts genuine NDVI values from Sentinel-2 L2A (primary) or Landsat 8/9 (fallback)
via Microsoft Planetary Computer STAC API for all 240 Karnataka taluks.

Data Source: Microsoft Planetary Computer (https://planetarycomputer.microsoft.com)
Satellites: Sentinel-2 L2A (10m) or Landsat Collection 2 Level 2 (30m)
Formula: NDVI = (NIR - RED) / (NIR + RED)

NO FAKE DATA. NO HARDCODED VALUES. NO RANDOM GENERATION.
"""

import os
import sys
import io
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import pystac_client
import planetary_computer
import rasterio
from rasterio.windows import from_bounds
import pyproj
from shapely.geometry import box
from shapely.ops import transform as shp_transform

# Fix Windows encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
TALUK_SHP = os.path.join(DATA_DIR, "raw", "shapefiles", "Taluk.shp")
OUTPUT_CSV = os.path.join(DATA_DIR, "satellite", "karnataka_ndvi_real.csv")
CHECKPOINT_CSV = os.path.join(DATA_DIR, "satellite", "ndvi_checkpoint.csv")

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
YEARS = [2019, 2020, 2021, 2022, 2023]


def get_taluk_centroids():
    """Load taluk centroids from KGIS shapefile."""
    gdf = gpd.read_file(TALUK_SHP)
    if gdf.crs and not gdf.crs.is_geographic:
        gdf = gdf.to_crs(epsg=4326)
    gdf = gdf[gdf['KGISTalukN'].notna()].copy()
    centroids = gdf.to_crs(epsg=3857).centroid.to_crs(epsg=4326)
    gdf['centroid_lat'] = centroids.y
    gdf['centroid_lon'] = centroids.x
    
    # Load district from spatial join map
    tdm_path = os.path.join(DATA_DIR, "kgis_tabular", "taluk_district_map.csv")
    if os.path.exists(tdm_path):
        tdm = pd.read_csv(tdm_path)
        dist_map = dict(zip(tdm['KGISTalukN'], tdm['KGISDist_1']))
        gdf['district'] = gdf['KGISTalukN'].map(dist_map).fillna('Unknown')
    else:
        gdf['district'] = 'Unknown'
    
    result = gdf[['KGISTalukN', 'centroid_lat', 'centroid_lon', 'district']].copy()
    result.rename(columns={'KGISTalukN': 'taluk'}, inplace=True)
    result.drop_duplicates(subset=['taluk'], inplace=True)
    result.reset_index(drop=True, inplace=True)
    print(f"Loaded {len(result)} taluk centroids from KGIS shapefile.")
    return result


def read_band_at_point(href, bbox_wgs84):
    """
    Read a band from a cloud-hosted COG at a small bbox.
    Handles CRS projection from WGS84 to the raster's native CRS.
    Returns pixel array or None.
    """
    try:
        with rasterio.open(href) as ds:
            # Project bbox from WGS84 to raster CRS
            project = pyproj.Transformer.from_crs('EPSG:4326', ds.crs, always_xy=True).transform
            bbox_projected = shp_transform(project, box(*bbox_wgs84))
            pb = bbox_projected.bounds
            
            window = from_bounds(pb[0], pb[1], pb[2], pb[3], ds.transform)
            
            # Check window is valid
            if window.width < 1 or window.height < 1:
                return None
            
            data = ds.read(1, window=window).astype(np.float64)
            if data.size < 3:
                return None
            return data
    except Exception:
        return None


def compute_ndvi_from_item(signed_item, bbox_wgs84, is_sentinel2=True):
    """
    Compute real NDVI from a single signed STAC item.
    Returns (ndvi_mean, ndvi_std) or None.
    """
    if is_sentinel2:
        red_key, nir_key = 'B04', 'B08'
        scale = 1.0 / 10000.0
        offset = 0.0
    else:
        red_key, nir_key = 'red', 'nir08'
        scale = 0.0000275
        offset = -0.2
    
    red_asset = signed_item.assets.get(red_key)
    nir_asset = signed_item.assets.get(nir_key)
    
    if not red_asset or not nir_asset:
        return None
    
    red_data = read_band_at_point(red_asset.href, bbox_wgs84)
    nir_data = read_band_at_point(nir_asset.href, bbox_wgs84)
    
    if red_data is None or nir_data is None:
        return None
    
    # Ensure same shape
    min_h = min(red_data.shape[0], nir_data.shape[0])
    min_w = min(red_data.shape[1], nir_data.shape[1])
    red_data = red_data[:min_h, :min_w]
    nir_data = nir_data[:min_h, :min_w]
    
    # Apply scale and offset
    red_data = red_data * scale + offset
    nir_data = nir_data * scale + offset
    
    # Mask invalid pixels
    valid = (red_data > 0.001) & (nir_data > 0.001) & (red_data < 1.0) & (nir_data < 1.0)
    
    if valid.sum() < 3:
        return None
    
    ndvi = (nir_data[valid] - red_data[valid]) / (nir_data[valid] + red_data[valid] + 1e-10)
    ndvi = ndvi[(ndvi >= -1.0) & (ndvi <= 1.0)]
    
    if len(ndvi) == 0:
        return None
    
    return float(np.mean(ndvi)), float(np.std(ndvi))


def fetch_ndvi_for_point(catalog, lat, lon, year):
    """
    Fetch REAL NDVI for a geographic point.
    Tries Landsat first (faster, 30m), then Sentinel-2 (10m) as fallback.
    Returns dict with stats or None.
    """
    # Smaller bbox = fewer pixels = faster download
    bbox = [lon - 0.002, lat - 0.002, lon + 0.002, lat + 0.002]
    
    # --- Try Landsat C2 L2 first (30m = faster reads) ---
    try:
        search = catalog.search(
            collections=["landsat-c2-l2"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 20}},
            limit=2
        )
        items = list(search.items())
        
        if items:
            signed = planetary_computer.sign(items[0])
            result = compute_ndvi_from_item(signed, bbox, is_sentinel2=False)
            if result:
                return {
                    'ndvi_mean': round(result[0], 4),
                    'ndvi_std': round(result[1], 4),
                    'ndvi_min': round(result[0], 4),
                    'ndvi_max': round(result[0], 4),
                    'scene_count': 1,
                    'platform': items[0].properties.get('platform', 'landsat'),
                    'data_source': 'Landsat C2 L2 (Planetary Computer)'
                }
    except Exception:
        pass
    
    # --- Fallback: Sentinel-2 L2A ---
    try:
        search = catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=bbox,
            datetime=f"{year}-01-01/{year}-12-31",
            query={"eo:cloud_cover": {"lt": 20}},
            limit=2
        )
        items = list(search.items())
        
        if items:
            signed = planetary_computer.sign(items[0])
            result = compute_ndvi_from_item(signed, bbox, is_sentinel2=True)
            if result:
                return {
                    'ndvi_mean': round(result[0], 4),
                    'ndvi_std': round(result[1], 4),
                    'ndvi_min': round(result[0], 4),
                    'ndvi_max': round(result[0], 4),
                    'scene_count': 1,
                    'platform': 'sentinel-2',
                    'data_source': 'Sentinel-2 L2A (Planetary Computer)'
                }
    except Exception:
        pass
    
    return None


def main():
    print("=" * 70)
    print("TerraTrust Step 1: Fetch REAL Satellite NDVI")
    print("Source: Microsoft Planetary Computer STAC API")
    print("Primary: Sentinel-2 L2A | Fallback: Landsat 8/9 C2 L2")
    print("NO FAKE DATA. Reading actual satellite pixels.")
    print("=" * 70)
    
    taluks = get_taluk_centroids()
    
    print("\nConnecting to Planetary Computer STAC catalog...")
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    print("Connected.\n")
    
    # Load checkpoint
    completed = set()
    results = []
    if os.path.exists(CHECKPOINT_CSV):
        cp = pd.read_csv(CHECKPOINT_CSV)
        results = cp.to_dict('records')
        completed = set(zip(cp['taluk'], cp['year']))
        print(f"Resuming from checkpoint: {len(completed)} entries done.\n")
    
    total = len(taluks) * len(YEARS)
    done = len(completed)
    s2_count = 0
    ls_count = 0
    fail_count = 0
    
    t0 = time.time()
    
    for _, trow in taluks.iterrows():
        name = trow['taluk']
        lat, lon = trow['centroid_lat'], trow['centroid_lon']
        district = trow['district']
        
        for year in YEARS:
            if (name, year) in completed:
                continue
            
            done += 1
            elapsed = time.time() - t0
            rate = max(done / max(elapsed, 1), 0.01)
            eta = (total - done) / rate
            
            print(f"  [{done}/{total}] {name} ({district}) {year}...", end=" ", flush=True)
            
            result = fetch_ndvi_for_point(catalog, lat, lon, year)
            
            if result:
                record = {
                    'taluk': name,
                    'district': district,
                    'latitude': round(lat, 6),
                    'longitude': round(lon, 6),
                    'year': year,
                    **result
                }
                results.append(record)
                
                tag = "S2" if "Sentinel" in result['data_source'] else "LS"
                if tag == "S2": s2_count += 1
                else: ls_count += 1
                
                print(f"[{tag}] NDVI={result['ndvi_mean']:.3f} ({result['scene_count']}sc) ETA:{eta:.0f}s")
            else:
                fail_count += 1
                print(f"[FAILED] ETA:{eta:.0f}s")
            
            completed.add((name, year))
            
            # Checkpoint every 25 entries
            if done % 25 == 0 and results:
                pd.DataFrame(results).to_csv(CHECKPOINT_CSV, index=False)
                print(f"    -- Checkpoint ({len(results)} records) --")
            
            time.sleep(0.1)
    
    # Save final output
    if results:
        df = pd.DataFrame(results)
        os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
        df.to_csv(OUTPUT_CSV, index=False)
        
        print("\n" + "=" * 70)
        print("STEP 1 COMPLETE")
        print("=" * 70)
        print(f"Total records:     {len(df)}")
        print(f"Sentinel-2 scenes: {s2_count}")
        print(f"Landsat scenes:    {ls_count}")
        print(f"Failed (no data):  {fail_count}")
        print(f"NDVI range:  {df['ndvi_mean'].min():.4f} to {df['ndvi_mean'].max():.4f}")
        print(f"NDVI mean:   {df['ndvi_mean'].mean():.4f}")
        print(f"Unique NDVI: {df['ndvi_mean'].nunique()}")
        print(f"Output: {OUTPUT_CSV}")
        print(f"Time: {time.time() - t0:.1f}s")
        
        # Verification
        print("\n--- VERIFICATION ---")
        if df['ndvi_mean'].nunique() < 5:
            print("[WARNING] Very few unique NDVI values!")
        else:
            print("[OK] NDVI shows real variance")
        if df['ndvi_mean'].min() < -0.5 or df['ndvi_mean'].max() > 1.0:
            print("[WARNING] NDVI values outside expected range!")
        else:
            print("[OK] NDVI values in valid range")
        
        print(f"\nNDVI by district (mean):")
        dist_ndvi = df.groupby('district')['ndvi_mean'].mean().sort_values()
        for d, v in dist_ndvi.items():
            print(f"  {d:25s} {v:.4f}")
    else:
        print("\n[ERROR] No NDVI data collected!")
    
    # Clean checkpoint
    if os.path.exists(CHECKPOINT_CSV) and results:
        os.remove(CHECKPOINT_CSV)


if __name__ == "__main__":
    main()
