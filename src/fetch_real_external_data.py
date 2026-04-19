import os
import time
import requests
import pandas as pd
import geopandas as gpd
from urllib3.util.retry import Retry
from requests.adapters import HTTPAdapter

try:
    from selenium import webdriver
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from webdriver_manager.chrome import ChromeDriverManager
except ImportError:
    webdriver = None

# Set absolute paths based on TerraTrust structure
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
TABULAR_DIR = os.path.join(DATA_DIR, 'kgis_tabular')

def setup_kaggle_financial_data():
    """
    Downloads the real Kaggle credit risk dataset.
    Requires kaggle API CLI config (kaggle.json) on the machine.
    """
    print("\n--- Downloading Real Kaggle Financial Data ---")
    dest = os.path.join(DATA_DIR, "credit_risk_dataset.csv")
    if os.path.exists(dest):
        print(f"Data already downloaded at {dest}")
        return True
        
    print("Attempting to download using kaggle CLI...")
    # This requires `pip install kaggle` and API keys setup.
    cmd_result = os.system(f"kaggle datasets download -d laotse/credit-risk-dataset -p {DATA_DIR} --unzip")
    if cmd_result != 0:
        print("Kaggle CLI not configured. Please download 'credit_risk_dataset.csv' from Kaggle and place it in the data/ folder.")
    return os.path.exists(dest)


def fetch_bhoomi_cadastral(survey_number, village_code):
    """
    Fetches real land parcel boundaries using a standard WFS API request.
    Note: Requires the live Government API endpoint to be accessible (not behind VPN).
    """
    print(f"\n--- Fetching Real Bhoomi/Cadastral Boundaries ---")
    # Replace with the actual live GeoServer endpoint when authenticated
    wfs_url = "https://karnataka.gov.in/geoserver/bhoomi/wfs"
    
    params = {
        'service': 'WFS',
        'version': '1.0.0',
        'request': 'GetFeature',
        'typeName': 'bhoomi:land_parcels',
        'outputFormat': 'application/json',
        'CQL_FILTER': f"survey_no='{survey_number}' AND village_code='{village_code}'"
    }

    print(f"Querying WFS: {wfs_url}?survey_no={survey_number}&village_code={village_code}")
    
    try:
        response = requests.get(wfs_url, params=params, timeout=15)
        if response.status_code == 200:
            data = response.json()
            if len(data.get('features', [])) > 0:
                gdf = gpd.GeoDataFrame.from_features(data["features"])
                return gdf
            else:
                print("Survey number not found in WFS response.")
        else:
            print(f"API Error {response.status_code}. Endpoint requires auth or is inaccessible.")
    except Exception as e:
        print(f"WFS Request failed: {e}")
        
    return None


def scrape_wris_groundwater():
    """Scrapes district-wise groundwater data dynamically rendered by JS via Selenium."""
    print("\n--- Scraping India-WRIS Groundwater Data ---")
    
    if webdriver is None:
        print("Selenium not installed. Run: pip install selenium webdriver-manager")
        return pd.DataFrame()

    options = Options()
    options.add_argument('--headless')
    options.add_argument('--window-size=1920,1080')
    
    try:
        print("Launching headless Chrome to parse Javascript DOM...")
        driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)
        driver.get("https://indiawris.gov.in/wris/#/groundWater")
        
        print("Waiting for Angular to render (15s)...")
        time.sleep(15) 
        
        # Scrape logic: Extract tables via Javascript executing to pull HTML.
        # This handles cases where XPATHS change by grabbing all tables.
        html_tables = driver.execute_script("return document.documentElement.outerHTML;")
        dfs = pd.read_html(html_tables)
        
        if len(dfs) > 0:
            df = dfs[0]
            # Clean up columns
            df.columns = [str(c).strip().lower().replace(' ', '_') for c in df.columns]
            
            out_file = os.path.join(TABULAR_DIR, "karnataka_groundwater.csv")
            df.to_csv(out_file, index=False)
            print(f"Successfully saved scraped data to {out_file}")
            return df
        else:
            print("No tables found in the rendered DOM.")
            return pd.DataFrame()

    except Exception as e:
        print(f"Scraping failed (Website structure may have changed): {e}")
        return pd.DataFrame()
    finally:
        try:
            driver.quit()
        except:
            pass


def fetch_satellite_vegetation_water(bbox_coords, start_date='2024-01-01', end_date='2024-12-31'):
    """
    NOTE: Google Earth Engine requires Developer Auth. 
    We are natively using Microsoft Planetary Computer STAC API instead natively in `src/satellite_pipeline.py`.
    Microsoft provides the identical Sentinel-2 L2A datasets entirely openly.
    """
    print("\n--- Querying Real Satellite Imagery (Microsoft STAC API) ---")
    print(f"  Instead of Earth Engine, TerraTrust processes Sentinel-2 optical data natively.")
    print(f"  Please refer to `src/satellite_pipeline.py` which dynamically extracts")
    print(f"  NDVI and NDWI matrices completely auth-free from Azure's live catalog.")
    return True


if __name__ == "__main__":
    print("TerraTrust Real External Data Ingestion Utilities")
    print("=================================================")
    # 1. Financial
    setup_kaggle_financial_data()
    
    # 2. Bhoomi
    fetch_bhoomi_cadastral("123/A", "VIL_442")
    
    # 3. Groundwater Scraper
    # scrape_wris_groundwater() # Uncomment to run (requires chrome/selenium)
    
    # 4. Satellite Imagery (Microsoft Planetary Computer)
    fetch_satellite_vegetation_water([74.0, 11.0, 79.0, 19.0])
