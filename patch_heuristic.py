import re

with open('src/build_master_dataset.py', 'r', encoding='utf-8') as f:
    content = f.read()

old_logic = '''    def soil_suitability(row):
        score = 100.0
        ph = row.get('pH', np.nan)
        if pd.notna(ph):
            if ph < 5.5: score -= 25
            elif ph > 8.5: score -= 20
            elif 6.0 <= ph <= 7.5: score += 5
        clay = row.get('clay_pct', np.nan)
        if pd.notna(clay):
            if clay > 55: score -= 15
            elif clay < 10: score -= 10
        n = row.get('nitrogen_g_per_kg', np.nan)
        if pd.notna(n):
            if n < 0.5: score -= 20
            elif n < 1.0: score -= 10
        oc = row.get('organic_carbon_dg_per_kg', np.nan)
        if pd.notna(oc):
            if oc > 100: score += 10
        return max(0, min(100, round(score, 1)))

    def water_availability(row):
        score = 50.0
        rain = row.get('avg_monthly_rainfall_mm', np.nan)
        if pd.notna(rain):
            if rain > 80: score += 20
            elif rain > 40: score += 10
            elif rain < 15: score -= 15
        wet = row.get('avg_root_zone_wetness', np.nan)
        if pd.notna(wet): score += wet * 25
        gw  = row.get('groundwater_depth_m', np.nan)
        if pd.notna(gw):
            if gw < 5:  score += 15   # shallow groundwater = good irrigation
            elif gw < 15: score += 5
            elif gw > 25: score -= 10  # very deep = poor
        ndvi = row.get('ndvi_annual_mean', np.nan)
        if pd.notna(ndvi): score += ndvi * 10
        return max(0, min(100, round(score, 1)))'''

new_logic = '''    def soil_suitability(row):
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
        score = 50.0
        crop = str(row.get('declared_crop', 'Mixed/Urban'))
        rain = row.get('avg_monthly_rainfall_mm', np.nan)
        gw  = row.get('groundwater_depth_m', np.nan)
        
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
        return max(0, min(100, round(score, 1)))'''

content = content.replace(old_logic, new_logic)
with open('src/build_master_dataset.py', 'w', encoding='utf-8') as f:
    f.write(content)
print("Dataset builder patched.")
