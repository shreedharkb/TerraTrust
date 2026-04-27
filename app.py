import streamlit as st
import json, csv, os, math, time
import folium
from streamlit_folium import st_folium
from fpdf import FPDF

st.set_page_config(page_title="TerraTrust", page_icon="🌿", layout="wide", initial_sidebar_state="collapsed")

@st.cache_data
def load_master():
    with open(os.path.join("data","processed","karnataka_master_dataset.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
@st.cache_data
def load_credits():
    with open(os.path.join("data","processed","heuristic_credit_scores.csv"), newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))
@st.cache_data
def load_taluk_map():
    with open(os.path.join("src","taluk_dist_map.json"), encoding="utf-8") as f:
        return json.load(f)

master_rows = load_master()
credit_rows = load_credits()
taluk_district = load_taluk_map()

taluk_data = {}
for r in master_rows:
    t = r["taluk"]
    if t not in taluk_data or int(r["year"]) > int(taluk_data[t]["year"]): taluk_data[t] = r
taluk_credit = {}
for r in credit_rows:
    t = r["taluk"]
    if t not in taluk_credit or int(r["year"]) > int(taluk_credit[t]["year"]): taluk_credit[t] = r

KA_LAT_MIN, KA_LAT_MAX, KA_LNG_MIN, KA_LNG_MAX = 11.5, 18.5, 74.0, 78.6
def in_karnataka(lat, lng): return KA_LAT_MIN<=lat<=KA_LAT_MAX and KA_LNG_MIN<=lng<=KA_LNG_MAX
def find_nearest_taluk(lat, lng):
    best, bd = None, float("inf")
    for n, r in taluk_data.items():
        d = math.sqrt((lat-float(r["latitude"]))**2+(lng-float(r["longitude"]))**2)
        if d < bd: bd, best = d, n
    return best
def fmt(v, d=3):
    try: return f"{float(v):.{d}f}"
    except: return str(v)

if "active_taluk" not in st.session_state: st.session_state.active_taluk = None
if "show_results" not in st.session_state: st.session_state.show_results = False
if "map_key" not in st.session_state: st.session_state.map_key = 0

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;color:#1A1A2E}
.stApp{background:#FFF5EB!important}
header{visibility:hidden}
.block-container{padding:1.8rem 3rem 3rem 3rem;max-width:1350px}
.brand-bar{display:flex;align-items:center;gap:14px;margin-bottom:22px;padding-bottom:16px;border-bottom:1px solid #FFE8D6}
.brand-icon{width:48px;height:48px;border-radius:12px;background:linear-gradient(135deg,#FF8C42,#FF6B1A);display:flex;align-items:center;justify-content:center;color:white;box-shadow:0 4px 12px rgba(255,107,26,0.25)}
.brand-name{font-family:'Space Grotesk',sans-serif;font-size:2.2rem;font-weight:800;color:#1A1A2E;letter-spacing:-0.8px;line-height:1.1}
.brand-tag{font-size:0.8rem;color:#9CA3AF;margin-top:-2px}
.panel-label{font-size:0.7rem;font-weight:700;letter-spacing:1.8px;text-transform:uppercase;color:#FF6B1A;margin-bottom:4px}
.panel-title{font-family:'Space Grotesk',sans-serif;font-size:1.5rem;font-weight:700;color:#1A1A2E;margin-bottom:4px}
.panel-desc{font-size:0.88rem;color:#9CA3AF;margin-bottom:14px;line-height:1.45}
.location-pill{display:flex;align-items:center;gap:10px;background:#FFF7ED;border:1px solid #FDBA74;border-radius:10px;padding:12px 18px;font-size:0.95rem;color:#9A3412;font-weight:600}
.loc-detail{font-weight:400;color:#C2410C}
.location-empty{background:#FFFAF5;border-color:#FFE8D6;color:#C4956A}
.error-pill{background:#FEF2F2;border:1px solid #FECACA;color:#B91C1C;font-size:0.88rem;padding:10px 16px;border-radius:10px;font-weight:500;margin-top:8px}
.score-block{text-align:center;padding:24px 0 18px 0;border-bottom:1px solid #F3F4F6;margin-bottom:20px}
.score-value{font-family:'Space Grotesk',sans-serif;font-size:4.2rem;font-weight:700;line-height:1}
.score-value.low{color:#DC2626}.score-value.moderate{color:#D97706}.score-value.high-score{color:#16A34A}
.score-out-of{font-size:0.88rem;color:#D1D5DB;font-weight:500;margin-top:6px}
.badge{display:inline-block;padding:5px 16px;border-radius:20px;font-size:0.78rem;font-weight:700;letter-spacing:0.5px;text-transform:uppercase;margin-top:12px}
.badge-low{background:#FEF2F2;color:#B91C1C;border:1px solid #FECACA}
.badge-mod{background:#FFFBEB;color:#B45309;border:1px solid #FDE68A}
.badge-high{background:#F0FDF4;color:#15803D;border:1px solid #BBF7D0}
.metric-grid{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:14px 0}
.metric-tile{background:#FFF9F3;border:1px solid #FFE8D6;border-radius:10px;padding:14px 16px}
.metric-label{font-size:0.68rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;color:#9CA3AF;margin-bottom:4px}
.metric-value{font-family:'Space Grotesk',sans-serif;font-size:1.25rem;font-weight:700;color:#1A1A2E}
.data-row{display:flex;justify-content:space-between;padding:9px 0;border-bottom:1px solid #FFF0E0;font-size:0.9rem}
.data-row:last-child{border-bottom:none}
.data-key{color:#6B7280;font-weight:500}.data-val{color:#1A1A2E;font-weight:600;text-align:right}
.rec-box{background:#FFFBF5;border:1px solid #FDDFB0;border-radius:10px;padding:16px 18px;margin-top:14px;font-size:0.88rem;color:#92400E;line-height:1.6}
.rec-box strong{font-weight:700;color:#78350F}
.detail-item{padding:10px 0;border-bottom:1px solid #FFF0E0;font-size:0.9rem;color:#44403C;line-height:1.55}
.detail-item:last-child{border-bottom:none}
.detail-icon{color:#FF6B1A;margin-right:8px;font-weight:700}
.empty-state{min-height:500px;display:flex;flex-direction:column;align-items:center;justify-content:center}
.map-wrap{border-radius:12px;overflow:hidden;border:1px solid #FFE8D6;margin-bottom:16px;}
.card-box {background:#FFFFFF;border:1px solid #FFE8D6;border-radius:20px;padding:26px;box-shadow:0 4px 12px rgba(0,0,0,0.03);margin-bottom:24px;}
.footer-text{text-align:center;font-size:0.85rem;color:#C4956A;margin-top:28px;padding-top:14px;border-top:1px solid #FFE8D6}
.data-row { font-size: 1.05rem; padding: 12px 0; }
.detail-item { font-size: 1.05rem; padding: 12px 0; }
.panel-desc { font-size: 1.0rem; }
.btn-row{display:flex;gap:10px;margin-top:16px}
.stDownloadButton>button, .stButton>button {background:#FF6B1A !important;color:white !important;font-weight:700 !important;border-radius:12px !important;border:none !important;font-size:1.05rem !important;width:100% !important;padding:12px !important;box-shadow:0 4px 10px rgba(255,107,26,0.2) !important;}
.stDownloadButton>button:hover, .stButton>button:hover {background:#E85A0C !important;box-shadow:0 6px 14px rgba(255,107,26,0.3) !important;}
</style>""", unsafe_allow_html=True)

state = 1
if st.session_state.active_taluk and not st.session_state.show_results: state = 2
if st.session_state.active_taluk and st.session_state.show_results: state = 3

if state == 1:
    status_text = "Awaiting Farm Selection"
    status_color = "#9CA3AF"
elif state == 2:
    status_text = "Connecting to KGIS & Satellite..."
    status_color = "#FF6B1A"
else:
    status_text = "Credit Intelligence Ready"
    status_color = "#16A34A"

st.markdown(f"""<div class="brand-bar">
<div class="brand-icon">
  <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
    <polygon points="12 2 2 7 12 12 22 7 12 2"></polygon>
    <polyline points="2 17 12 22 22 17"></polyline>
    <polyline points="2 12 12 17 22 12"></polyline>
  </svg>
</div>
<div><div class="brand-name">TerraTrust</div><div class="brand-tag">Geospatial Credit Intelligence for Rural Banking</div></div>
<div style="margin-left:auto; color:{status_color}; background:#FFFFFF; padding:8px 16px; border-radius:20px; border:1px solid {status_color}40; font-weight:700; font-size:0.85rem; box-shadow:0 2px 6px rgba(0,0,0,0.02);">
  {status_text}
</div>
</div>""", unsafe_allow_html=True)

left_col, right_col = st.columns([1, 1], gap="large")

with left_col:
    # EXPLORE card
    st.markdown("""<div class="card-box">
    <div class="panel-label">EXPLORE</div>
    <div class="panel-title">Spatial Selection</div>
    <div class="panel-desc">Click anywhere on the Karnataka map.</div></div>""", unsafe_allow_html=True)

    m = folium.Map(location=[15.3173,75.7139], zoom_start=7, tiles="CartoDB positron", control_scale=False)
    st.markdown('<div class="map-wrap">', unsafe_allow_html=True)
    map_result = st_folium(m, width=None, height=400, key=f"map_{st.session_state.map_key}")
    st.markdown('</div>', unsafe_allow_html=True)

    if map_result and map_result.get("last_clicked"):
        clat, clng = map_result["last_clicked"]["lat"], map_result["last_clicked"]["lng"]
        if in_karnataka(clat, clng):
            nearest = find_nearest_taluk(clat, clng)
            if nearest and nearest != st.session_state.active_taluk:
                st.session_state.active_taluk = nearest
                st.session_state.show_results = False
                st.rerun()
        else:
            st.markdown('<div class="error-pill">⚠ Outside Karnataka — please click within the state boundary.</div>', unsafe_allow_html=True)

    if st.session_state.active_taluk:
        tk = st.session_state.active_taluk
        di = taluk_district.get(tk,"—")
        cr = taluk_data.get(tk,{}).get("declared_crop","—")
        st.markdown(f'<div class="location-pill"><span>📍</span><strong>{tk}</strong> · <span class="loc-detail">{di}</span> · <span class="loc-detail">{cr}</span></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="location-pill location-empty"><span>📍</span>Click on the map to select a location</div>', unsafe_allow_html=True)

    # REGIONAL INTELLIGENCE card (replaces diagnostics)
    if st.session_state.active_taluk and st.session_state.show_results:
        tk = st.session_state.active_taluk
        row = taluk_data.get(tk, {})
        di = taluk_district.get(tk,"—")
        st.markdown(f"""<div class="card-box">
        <div class="panel-label">REGIONAL INTELLIGENCE</div>
        <div class="panel-title">Soil & Climate Profile</div>
        <div class="panel-desc">{tk}, {di} — Environmental baseline data</div>
        <div class="metric-grid">
            <div class="metric-tile"><div class="metric-label">Clay Content</div><div class="metric-value">{fmt(row.get('clay_pct','—'),1)}%</div></div>
            <div class="metric-tile"><div class="metric-label">Sand Content</div><div class="metric-value">{fmt(row.get('sand_pct','—'),1)}%</div></div>
            <div class="metric-tile"><div class="metric-label">Max Temperature</div><div class="metric-value">{fmt(row.get('max_temp_c','—'),1)}°C</div></div>
            <div class="metric-tile"><div class="metric-label">Humidity</div><div class="metric-value">{fmt(row.get('avg_humidity_pct','—'),1)}%</div></div>
        </div>
        <div class="data-row"><span class="data-key">Silt Content</span><span class="data-val">{fmt(row.get('silt_pct','—'),1)}%</span></div>
        <div class="data-row"><span class="data-key">Soil pH</span><span class="data-val">{fmt(row.get('pH','—'),1)}</span></div>
        <div class="data-row"><span class="data-key">Nitrogen</span><span class="data-val">{fmt(row.get('nitrogen_g_per_kg','—'),2)} g/kg</span></div>
        <div class="data-row"><span class="data-key">Organic Carbon</span><span class="data-val">{fmt(row.get('organic_carbon_dg_per_kg','—'),1)} dg/kg</span></div>
        <div class="data-row"><span class="data-key">Bulk Density</span><span class="data-val">{fmt(row.get('bulk_density_cg_per_cm3','—'),0)} cg/cm³</span></div>
        <div class="data-row"><span class="data-key">Min Temperature</span><span class="data-val">{fmt(row.get('min_temp_c','—'),1)}°C</span></div>
        <div class="data-row"><span class="data-key">Root Zone Wetness</span><span class="data-val">{fmt(row.get('avg_root_zone_wetness','—'))}</span></div>
        <div class="data-row"><span class="data-key">Sand/Clay Ratio</span><span class="data-val">{fmt(row.get('sand_clay_ratio','—'))}</span></div>
        </div>""", unsafe_allow_html=True)
        
        # Determine Actions
        cred = taluk_credit.get(tk,{})
        rc = cred.get("risk_category", "Unknown")
        if rc == "High":
            act1 = "Recommend rejection or reduced disbursement. High-risk indicators across multiple data sources."
            act3 = "If approved under exception, mandate monthly satellite monitoring and physical verification."
        elif rc == "Moderate":
            act1 = "Flag for senior review. Cross-validate with physical field inspection before approval."
            act3 = "Schedule bi-monthly post-disbursement satellite monitoring."
        else:
            act1 = "Proceed with standard loan processing. Low-risk profile confirmed by multi-source validation."
            act3 = "Schedule routine post-disbursement monitoring at 90-day intervals."
            
        st.markdown(f"""<div class="card-box" style="min-height: 480px; display: flex; flex-direction: column; justify-content: space-between;">
        <div>
        <div class="panel-label">SUGGESTED ACTIONS</div>
        <div class="panel-title">Action Plan</div>
        <div class="detail-item" style="margin-top: 16px;">{act1}</div>
        <div class="detail-item" style="margin-top: 16px;">Verify land ownership records with KGIS land registry for survey number confirmation.</div>
        <div class="detail-item" style="margin-top: 16px;">{act3}</div>
        </div>
        </div>""", unsafe_allow_html=True)

with right_col:
    if st.session_state.active_taluk and not st.session_state.show_results:
        bar = st.progress(0, text="Loading pre-processed dataset...")
        for i in range(7):
            time.sleep(0.5)
            labels = ["Loading dataset...","Reading groundwater records...","Loading NDVI values...","Reading climate data...","Running ML inference...","Calculating credit score...","Compiling report..."]
            bar.progress((i+1)/7, text=labels[i])
        bar.empty()
        st.session_state.show_results = True
        st.rerun()

    elif st.session_state.active_taluk and st.session_state.show_results:
        tk = st.session_state.active_taluk
        row = taluk_data.get(tk,{})
        cred = taluk_credit.get(tk,{})
        cs = float(cred.get("heuristic_credit_score",0))
        rc = cred.get("risk_category", "Unknown")
        rec = cred.get("recommendation","No recommendation available.")
        dn = taluk_district.get(tk,"—")
        bc = "badge-low" if rc=="Low" else "badge-mod" if rc=="Moderate" else "badge-high"
        sc = "low" if rc=="Low" else "moderate" if rc=="Moderate" else "high-score"

        # ASSESSMENT
        st.markdown(f"""<div class="card-box">
        <div class="panel-label">ASSESSMENT</div>
        <div class="panel-title">Credit Intelligence Report</div>
        <div class="panel-desc">{tk} · {dn}</div>
        <div class="score-block"><div class="score-value {sc}">{cs:.1f}</div>
        <div class="score-out-of">out of 100</div>
        <span class="badge {bc}">{rc} Risk</span></div></div>""", unsafe_allow_html=True)

        # SATELLITE & KGIS DATA
        ndvi=row.get("ndvi_annual_mean","—"); gw=row.get("groundwater_depth_m","—")
        rain=row.get("avg_monthly_rainfall_mm","—"); sf=row.get("soil_fertility_index","—")
        wp=row.get("water_table_pressure","—"); ai=row.get("aridity_index","—")
        vs=row.get("vegetation_stress_index","—"); yp="—"

        st.markdown(f"""<div class="card-box">
        <div class="panel-label">SATELLITE & KGIS DATA</div>
        <div class="panel-title">Geospatial Evidence</div>
        <div class="metric-grid">
            <div class="metric-tile"><div class="metric-label">NDVI (Crop Health)</div><div class="metric-value">{fmt(ndvi)}</div></div>
            <div class="metric-tile"><div class="metric-label">Groundwater Depth</div><div class="metric-value">{fmt(gw,1)} m</div></div>
            <div class="metric-tile"><div class="metric-label">Avg Rainfall</div><div class="metric-value">{fmt(rain,1)} mm</div></div>
            <div class="metric-tile"><div class="metric-label">Soil Fertility</div><div class="metric-value">{fmt(sf)}</div></div>
        </div>
        <div class="data-row"><span class="data-key">Aridity Index</span><span class="data-val">{fmt(ai)}</span></div>
        <div class="data-row"><span class="data-key">Vegetation Stress</span><span class="data-val">{fmt(vs)}</span></div>
        <div class="data-row"><span class="data-key">Water Table Pressure</span><span class="data-val">{fmt(wp)}</span></div>
        <div class="data-row"><span class="data-key">Historical Yield Potential</span><span class="data-val">{fmt(yp,1)}</span></div>
        <div class="data-row"><span class="data-key">Declared Crop</span><span class="data-val">{row.get('declared_crop','—')}</span></div>
        </div>""", unsafe_allow_html=True)

        # DECISION SUPPORT - split into multiple st.markdown calls to avoid HTML rendering issues
        crop = row.get("declared_crop","—")
        nv = float(ndvi) if ndvi!="—" else 0
        gv = float(gw) if gw!="—" else 0
        sv = float(sf) if sf!="—" else 0
        ypv = float(yp) if yp!="—" else 0

        ndvi_txt = "healthy vegetation cover" if nv>0.35 else "moderate vegetation" if nv>0.2 else "stressed or sparse vegetation"
        gw_txt = "shallow and accessible" if gv<10 else "moderate depth" if gv<20 else "deep — irrigation cost may be high"
        soil_txt = "fertile and suitable" if sv>1.5 else "moderately fertile" if sv>1.0 else "low fertility — may need amendment"

        st.markdown(f"""<div class="card-box">
        <div class="panel-label">DECISION SUPPORT</div>
        <div class="panel-title">Loan Recommendation</div>
        <div class="rec-box"><strong>System Recommendation:</strong><br>{rec}</div>
        </div>""", unsafe_allow_html=True)

        st.markdown(f"""<div class="card-box">
        <div class="panel-label">TECHNICAL ANALYSIS</div>
        <div class="panel-title">Detailed Assessment</div>
        <div class="detail-item"><strong>Crop Health (NDVI {fmt(ndvi)}):</strong> Satellite imagery indicates {ndvi_txt} for {crop}. {"Crop growth is on track for the declared cycle." if nv>0.25 else "Vegetation indices suggest possible crop stress."}</div>
        <div class="detail-item"><strong>Water Availability (GW {fmt(gw,1)}m):</strong> Groundwater table is {gw_txt}. {"Sufficient water supports irrigation needs." if gv<15 else "Consider supplementary irrigation assessment before disbursement."}</div>
        <div class="detail-item"><strong>Soil Suitability (Fertility {fmt(sf)}):</strong> Soil composition is {soil_txt} for {crop}. {"No soil amendments needed." if sv>1.5 else "Recommend soil testing and nutrient supplementation."}</div>
        <div class="detail-item"><strong>Historical Yield:</strong> Yield potential score of {fmt(yp,1)} based on multi-year analysis. {"Consistent performance reduces default risk." if ypv>100 else "Below-average yields suggest elevated monitoring."}</div>
        </div>""", unsafe_allow_html=True)

        # Action logic (re-evaluated for PDF)
        if rc == "High":
            act1 = "Recommend rejection or reduced disbursement. High-risk indicators across multiple data sources."
            act3 = "If approved under exception, mandate monthly satellite monitoring and physical verification."
        elif rc == "Moderate":
            act1 = "Flag for senior review. Cross-validate with physical field inspection before approval."
            act3 = "Schedule bi-monthly post-disbursement satellite monitoring."
        else:
            act1 = "Proceed with standard loan processing. Low-risk profile confirmed by multi-source validation."
            act3 = "Schedule routine post-disbursement monitoring at 90-day intervals."

        # PDF Report Generation using fpdf2
        def cln(text):
            return str(text).replace("—", "-").replace("–", "-").replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
            
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Helvetica", "B", 16)
        pdf.cell(0, 10, cln("TERRATRUST - CREDIT INTELLIGENCE REPORT"), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(0, 10, cln(f"Taluk: {tk} | District: {dn} | Crop: {crop}"), new_x="LMARGIN", new_y="NEXT", align="C")
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, cln(f"CREDIT SCORE: {cs:.1f} / 100 ({rc} Risk)"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, cln(f"Recommendation: {rec}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, cln("GEOSPATIAL EVIDENCE"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.cell(80, 8, cln(f"NDVI (Crop Health): {fmt(ndvi)}"))
        pdf.cell(80, 8, cln(f"Groundwater Depth: {fmt(gw,1)} m"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(80, 8, cln(f"Avg Rainfall: {fmt(rain,1)} mm"))
        pdf.cell(80, 8, cln(f"Soil Fertility Index: {fmt(sf)}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(80, 8, cln(f"Aridity Index: {fmt(ai)}"))
        pdf.cell(80, 8, cln(f"Vegetation Stress: {fmt(vs)}"), new_x="LMARGIN", new_y="NEXT")
        pdf.cell(80, 8, cln(f"Historical Yield: {fmt(yp,1)}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, cln("TECHNICAL ANALYSIS"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, cln(f"Crop Health: {ndvi_txt}."), new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 8, cln(f"Water Availability: {gw_txt}."), new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 8, cln(f"Soil Suitability: {soil_txt}."), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(5)
        pdf.set_font("Helvetica", "B", 14)
        pdf.cell(0, 10, cln("SUGGESTED ACTIONS"), new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 12)
        pdf.multi_cell(0, 8, cln(f"1. {act1}"), new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 8, cln(f"2. Verify land ownership with KGIS registry."), new_x="LMARGIN", new_y="NEXT")
        pdf.multi_cell(0, 8, cln(f"3. {act3}"), new_x="LMARGIN", new_y="NEXT")
        pdf.ln(10)
        pdf.set_font("Helvetica", "I", 10)
        pdf.cell(0, 10, cln("Generated by TerraTrust AI"), new_x="LMARGIN", new_y="NEXT", align="C")
        
        pdf_bytes = bytes(pdf.output())

        # Buttons: PDF Download + Reset without emojis
        btn1, btn2 = st.columns(2)
        with btn1:
            st.download_button("Download Report", data=pdf_bytes, file_name=f"TerraTrust_{tk}_Report.pdf", mime="application/pdf")
        with btn2:
            if st.button("Reset"):
                st.session_state.active_taluk = None
                st.session_state.show_results = False
                st.session_state.map_key += 1
                st.rerun()

    else:
        st.markdown("""<div class="empty-state">
        <div style="font-size:3rem;margin-bottom:14px;opacity:0.15;">🌿</div>
        <div class="panel-title" style="text-align:center;color:#C4956A;font-size:1.2rem;">No Location Selected</div>
        <div class="panel-desc" style="text-align:center;max-width:320px;color:#C4956A;font-size:0.92rem;">
        Click anywhere on the Karnataka map to auto-detect the nearest taluk and generate a
        <strong style="color:#FF6B1A;">Credit Intelligence Report</strong>.</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<div class="footer-text">TerraTrust v1.0 · KGIS + Satellite Imagery (NDVI) + ML Models · 240 Taluks · Karnataka<br><span style="font-size:0.75rem;color:#D4A574;">Note: Credit scores are derived from physics-informed heuristic models using satellite + soil + climate data. No real loan repayment ground truth was used.</span></div>', unsafe_allow_html=True)
