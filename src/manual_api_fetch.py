import requests
import pandas as pd
import os

def fetch_real_groundwater_api():
    print("Manually querying Data.gov.in API for Karnataka Groundwater...")
    
    # This is a public API endpoint for CGWB Ground Water Level Data
    url = "https://api.data.gov.in/resource/4b868d40-0255-46fd-abce-59b4c022ab93"
    
    params = {
        'api-key': '579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b', # Public demo key
        'format': 'json',
        'limit': '1000',
        'filters[State]': 'KARNATAKA'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            records = data.get('records', [])
            
            if records:
                df = pd.DataFrame(records)
                print(f"Success! Downloaded {len(df)} real records from data.gov.in.")
                
                # Standardize column names if needed, but retaining raw data is good for provenance
                out_path = 'D:/TerraTrust/data/kgis_tabular/karnataka_groundwater.csv'
                df.to_csv(out_path, index=False)
                print(f"Saved to {out_path}")
                return df
            else:
                print("API returned no records for Karnataka.")
        else:
            print(f"API Error: {response.status_code}")
            print(response.text)
    except Exception as e:
        print(f"Failed to connect to API: {e}")

if __name__ == "__main__":
    fetch_real_groundwater_api()
