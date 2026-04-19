import requests

def search_india_data_gov():
    print("Searching data.gov.in Catalog for Groundwater Datasets...")
    api_key = "579b464db66ec23bdd000001cdd3946e44ce4aad7209ff7b23ac571b"
    # Query the catalog for datasets titled 'Ground Water'
    catalog_url = f"https://api.data.gov.in/catalog/v1?api-key={api_key}&format=json&limit=50&offset=0&title=Ground%20Water"
    
    r = requests.get(catalog_url)
    if r.status_code != 200:
        print(f"Catalog query failed: {r.status_code}")
        return

    data = r.json()
    records = data.get("records", [])
    
    valid_ids = []
    
    for rec in records:
        title = rec.get("title", "")
        # Look for State-wise or district-wise depth data
        for res in rec.get("resources", []):
            res_id = res.get("id")
            # Test the resource ID
            try:
                test_url = f"https://api.data.gov.in/resource/{res_id}?api-key={api_key}&format=json&limit=1"
                t_r = requests.get(test_url).json()
                
                # Check if it has a 'State' or 'District' column and is returning records
                if t_r.get("count", 0) > 0:
                    fields = [f.get("target", "").lower() for f in t_r.get("field", [])]
                    # We want datasets that contain districts and water level or 'state'
                    if 'district' in fields or 'state_name' in fields or 'state' in fields:
                        print(f"[FOUND] {title[:80]}...")
                        print(f"  -> Resource ID: {res_id}")
                        print(f"  -> Columns: {fields}\n")
                        valid_ids.append(res_id)
            except Exception as e:
                pass
                
    if not valid_ids:
        print("No valid tabular datasets found with District/State columns active right now.")
    else:
        print(f"To download, we can use Resource ID: {valid_ids[0]}")

if __name__ == '__main__':
    search_india_data_gov()
