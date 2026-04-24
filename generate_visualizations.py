"""
TerraTrust Visualization Generator
=================================================
Generates all figures for Results_and_Visualizations/
"""
import os, sys, json, warnings, joblib
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

BASE = os.path.dirname(os.path.abspath(__file__))
OUT  = os.path.join(BASE, 'results')
METRICS = os.path.join(OUT, 'model_metrics.json')
MASTER  = os.path.join(BASE, 'data', 'processed', 'karnataka_master_dataset.csv')

for d in ['System_Architecture','Training_Curves','Validation_Maps','XAI_SHAP_Analysis']:
    os.makedirs(os.path.join(OUT, d), exist_ok=True)

with open(METRICS, encoding='utf-8') as f:
    M = json.load(f)

COLORS = ['#2563EB','#10B981','#F59E0B','#EF4444','#8B5CF6','#EC4899']
plt.rcParams.update({'font.size':11, 'figure.dpi':150, 'savefig.bbox':'tight',
                      'axes.spines.top':False, 'axes.spines.right':False})

# ═══════════════════════════════════════════════════════════════════════════
# 1. SYSTEM ARCHITECTURE
# ═══════════════════════════════════════════════════════════════════════════

def draw_pipeline_overview():
    fig, ax = plt.subplots(figsize=(16,10))
    ax.set_xlim(0,16); ax.set_ylim(0,10); ax.axis('off')
    ax.set_facecolor('#0F172A'); fig.patch.set_facecolor('#0F172A')

    def box(x,y,w,h,txt,color,fs=10):
        r = mpatches.FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.15",
            facecolor=color, edgecolor='white', linewidth=1.5, alpha=0.9)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, txt, ha='center', va='center', color='white',
                fontsize=fs, fontweight='bold', wrap=True)

    def arrow(x1,y1,x2,y2):
        ax.annotate('', xy=(x2,y2), xytext=(x1,y1),
            arrowprops=dict(arrowstyle='->', color='#94A3B8', lw=2))

    ax.text(8,9.5,'TerraTrust — ML Pipeline Architecture',
            ha='center',color='white',fontsize=16,fontweight='bold')

    # Data sources row
    box(0.3,7.5, 2.5,1.2, 'KGIS\nBoundaries\n(Shapefiles)', '#1E40AF')
    box(3.3,7.5, 2.5,1.2, 'ISRIC\nSoilGrids\n(API)', '#1E40AF')
    box(6.3,7.5, 2.5,1.2, 'NASA POWER\nClimate\n(API)', '#1E40AF')
    box(9.3,7.5, 2.5,1.2, 'Landsat/S2\nNDVI\n(STAC)', '#1E40AF')
    box(12.3,7.5,2.5,1.2, 'GRACE-FO\nGroundwater\n(Model)', '#1E40AF')

    # Arrows to preprocessing
    for x in [1.55, 4.55, 7.55, 10.55, 13.55]:
        arrow(x, 7.5, 8, 6.7)

    box(5.5,5.8, 5,1.0, 'Data Preprocessing & Feature Engineering\n(build_master_dataset.py)', '#7C3AED')

    arrow(8, 5.8, 3.5, 5.0)
    arrow(8, 5.8, 8, 5.0)
    arrow(8, 5.8, 12.5, 5.0)

    # ML Models row
    box(0.5,3.8, 3.5,1.2, 'Model A\nCrop Health\n(XGBoost Clf)', '#059669')
    box(4.5,3.8, 3.5,1.2, 'Model B\nSoil Quality\n(Random Forest)', '#059669')
    box(8.5,3.8, 3.5,1.2, 'Model C\nWater Avail.\n(GradBoost Reg)', '#059669')

    for x in [2.25, 6.25, 10.25]:
        arrow(x, 3.8, 8, 2.9)

    box(5.5,1.8, 5,1.2, 'Model D — Credit Risk\nHierarchical XGBoost\n(Uses A+B+C predictions)', '#DC2626')

    arrow(8, 1.8, 8, 1.0)
    box(5,0.0, 6,1.0, 'Visual Credit Score\nLoan Decision + Geospatial Evidence', '#B45309', 11)

    fig.savefig(os.path.join(OUT,'System_Architecture','TerraTrust_Pipeline_Overview.png'),
                facecolor=fig.get_facecolor())
    plt.close()
    print("  [OK] Pipeline Overview")

def draw_model_architecture():
    fig, ax = plt.subplots(figsize=(14,8))
    ax.set_xlim(0,14); ax.set_ylim(0,8); ax.axis('off')
    ax.set_facecolor('#1E293B'); fig.patch.set_facecolor('#1E293B')

    def box(x,y,w,h,txt,color,fs=9):
        r = mpatches.FancyBboxPatch((x,y),w,h, boxstyle="round,pad=0.12",
            facecolor=color, edgecolor='white', linewidth=1.2, alpha=0.85)
        ax.add_patch(r)
        ax.text(x+w/2, y+h/2, txt, ha='center', va='center', color='white',
                fontsize=fs, fontweight='bold')

    ax.text(7,7.5,'4-Model Hierarchical ML Architecture', ha='center',
            color='white', fontsize=15, fontweight='bold')

    # Input features
    feats = ['Soil (clay,sand,pH,N)','Climate (rain,temp,\nhumidity)','NDVI Satellite',
             'Groundwater depth','Geolocation\n(lat,lon)','Engineered\n(aridity, fertility)']
    for i,f in enumerate(feats):
        box(0.2+i*2.3, 5.8, 2.1, 1.1, f, '#334155', 8)

    # Models
    models_info = [
        ('Model A\nCrop Health\nXGBoost\nAcc: {:.1%}'.format(M['Model_A_CropHealth']['spatial']['test_accuracy']), '#2563EB'),
        ('Model B\nSoil Quality\nRandom Forest\nAcc: {:.1%}'.format(M['Model_B_Soil']['spatial']['test_accuracy']), '#10B981'),
        ('Model C\nWater Avail.\nGradBoost\nR²: {:.3f}'.format(M['Model_C_Water']['spatial']['test_r2']), '#F59E0B'),
    ]
    for i,(txt,c) in enumerate(models_info):
        box(0.5+i*4.5, 3.5, 3.5, 1.8, txt, c, 9)

    box(4.5, 0.5, 5, 1.8,
        'Model D — Credit Risk\nHierarchical XGBoost\nAcc: {:.1%}\nInputs: A+B+C predictions + context'.format(
            M['Model_D_Credit_Risk']['spatial']['test_accuracy']), '#EF4444', 9)

    fig.savefig(os.path.join(OUT,'System_Architecture','TerraTrust_Model_Architecture.png'),
                facecolor=fig.get_facecolor())
    plt.close()
    print("  [OK] Model Architecture")


# ═══════════════════════════════════════════════════════════════════════════
# 2. TRAINING CURVES
# ═══════════════════════════════════════════════════════════════════════════

def plot_training_curves():
    models = [
        ('Model_A_CropHealth', 'Model A: Crop Health', True),
        ('Model_B_Soil', 'Model B: Soil Quality', True),
        ('Model_C_Water', 'Model C: Water Availability', False),
        ('Model_D_Credit_Risk', 'Model D: Credit Risk', True),
    ]
    for key, title, is_clf in models:
        m = M[key]
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        fig.suptitle(f'TerraTrust — {title} Training Curves', fontsize=14, fontweight='bold')
        metric = 'Accuracy' if is_clf else 'R²'
        s = m['spatial']

        # Panel 1: Train vs Test
        ax = axes[0]
        tr_s = s.get('train_accuracy', s.get('train_r2',0))
        te_s = s.get('test_accuracy', s.get('test_r2',0))
        x = np.arange(1); w = 0.3
        ax.bar(x-w/2, [tr_s], w, color=COLORS[0], label='Train', alpha=0.85)
        ax.bar(x+w/2, [te_s], w, color=COLORS[1], label='Test', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(['Evaluation Split'])
        ax.set_ylabel(metric); ax.set_title(f'Train vs Test {metric}')
        ax.legend(); ax.set_ylim(0, 1.1)
        ax.text(0-w/2, tr_s+0.02, f'{tr_s:.3f}', ha='center', fontsize=10)
        ax.text(0+w/2, te_s+0.02, f'{te_s:.3f}', ha='center', fontsize=10)

        # Panel 2: CV Scores
        ax = axes[1]
        cv_s = s.get('cv_scores',[])
        x_s = range(1, len(cv_s)+1)
        ax.plot(x_s, cv_s, 'o-', color=COLORS[0], label=f'CV (mean={s["cv_mean"]:.3f})', lw=2)
        ax.axhline(s['cv_mean'], color=COLORS[0], ls='--', alpha=0.5)
        ax.set_xlabel('Fold'); ax.set_ylabel(metric)
        ax.set_title('Cross-Validation Scores (GroupKFold)'); ax.legend(fontsize=10)

        # Panel 3: Gap Analysis
        ax = axes[2]
        gap = s['train_test_gap']
        color_g = '#10B981' if gap < 0.08 else '#F59E0B' if gap < 0.15 else '#EF4444'
        bar = ax.bar(['Evaluation Split'], [gap], width=0.4, color=color_g, alpha=0.85)[0]
        ax.axhline(0.08, color='#10B981', ls='--', alpha=0.6, label='Good (<0.08)')
        ax.axhline(0.15, color='#F59E0B', ls='--', alpha=0.6, label='Moderate (<0.15)')
        ax.set_ylabel('Train-Test Gap'); ax.set_title('Overfitting Analysis')
        ax.legend(fontsize=10)
        ax.text(bar.get_x()+bar.get_width()/2, gap+0.005, f'{gap:.4f}', ha='center', fontsize=10)

        plt.tight_layout()
        fname = f'{key}_Training_Curves.png'
        fig.savefig(os.path.join(OUT,'Training_Curves', fname))
        plt.close()
        print(f"  [OK] {fname}")

    # Combined summary dashboard
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TerraTrust — All Models Performance Summary', fontsize=15, fontweight='bold')
    for idx, (key, title, is_clf) in enumerate(models):
        ax = axes[idx//2][idx%2]
        m = M[key]
        s = m['spatial']
        metric = 'Accuracy' if is_clf else 'R²'
        tr_s = s.get('train_accuracy', s.get('train_r2',0))
        te_s = s.get('test_accuracy', s.get('test_r2',0))
        x = np.arange(1); w = 0.3
        ax.bar(x-w/2, [tr_s], w, color=COLORS[0], label='Train', alpha=0.85)
        ax.bar(x+w/2, [te_s], w, color=COLORS[1], label='Test', alpha=0.85)
        ax.set_xticks(x); ax.set_xticklabels(['Evaluation Split'])
        ax.set_ylabel(metric); ax.set_title(title); ax.legend(fontsize=10)
        ax.set_ylim(0, 1.1)
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,'Training_Curves','All_Models_Summary.png'))
    plt.close()
    print("  [OK] All_Models_Summary.png")


# ═══════════════════════════════════════════════════════════════════════════
# 3. VALIDATION MAPS (choropleth-style using scatter)
# ═══════════════════════════════════════════════════════════════════════════

def plot_validation_maps():
    df = pd.read_csv(MASTER)

    # Crop Health Map
    fig, ax = plt.subplots(figsize=(10, 12))
    df['crop_h_label'] = np.where(df['ndvi_annual_mean'] >= 0.4, 'Healthy', 'Stressed')
    taluk_health = df.groupby('taluk').agg(
        lat=('latitude','mean'), lon=('longitude','mean'),
        ndvi=('ndvi_annual_mean','mean')).reset_index()
    sc = ax.scatter(taluk_health['lon'], taluk_health['lat'], c=taluk_health['ndvi'],
                    cmap='RdYlGn', s=80, edgecolors='#333', linewidth=0.5, vmin=0.1, vmax=0.8)
    plt.colorbar(sc, ax=ax, label='Mean NDVI', shrink=0.7)
    ax.set_title('Karnataka — Crop Health by Taluk (NDVI)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    fig.savefig(os.path.join(OUT,'Validation_Maps','Karnataka_CropHealth_Map.png'))
    plt.close()
    print("  [OK] CropHealth Map")

    # Groundwater Map
    fig, ax = plt.subplots(figsize=(10, 12))
    taluk_gw = df.groupby('taluk').agg(
        lat=('latitude','mean'), lon=('longitude','mean'),
        gw=('groundwater_depth_m','mean')).reset_index()
    sc = ax.scatter(taluk_gw['lon'], taluk_gw['lat'], c=taluk_gw['gw'],
                    cmap='RdYlBu', s=80, edgecolors='#333', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='Groundwater Depth (m)', shrink=0.7)
    ax.set_title('Karnataka — Groundwater Depth by Taluk', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    fig.savefig(os.path.join(OUT,'Validation_Maps','Karnataka_GroundwaterDepth_Map.png'))
    plt.close()
    print("  [OK] Groundwater Map")

    # Soil Quality Map
    fig, ax = plt.subplots(figsize=(10, 12))
    taluk_soil = df.groupby('taluk').agg(
        lat=('latitude','mean'), lon=('longitude','mean'),
        oc=('organic_carbon_dg_per_kg','mean')).reset_index()
    sc = ax.scatter(taluk_soil['lon'], taluk_soil['lat'], c=taluk_soil['oc'],
                    cmap='YlOrBr', s=80, edgecolors='#333', linewidth=0.5)
    plt.colorbar(sc, ax=ax, label='Organic Carbon (dg/kg)', shrink=0.7)
    ax.set_title('Karnataka — Soil Quality by Taluk', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    fig.savefig(os.path.join(OUT,'Validation_Maps','Karnataka_SoilQuality_Map.png'))
    plt.close()
    print("  [OK] Soil Quality Map")

    # Credit Risk Map
    fig, ax = plt.subplots(figsize=(10, 12))
    df['simulated_cs'] = df['loan_risk_3class'].map({'High': 400, 'Moderate': 550, 'Low': 750}).fillna(500)
    taluk_credit = df.groupby('taluk').agg(
        lat=('latitude','mean'), lon=('longitude','mean'),
        cs=('simulated_cs','mean')).reset_index()
    sc = ax.scatter(taluk_credit['lon'], taluk_credit['lat'], c=taluk_credit['cs'],
                    cmap='RdYlGn', s=80, edgecolors='#333', linewidth=0.5, vmin=350, vmax=700)
    plt.colorbar(sc, ax=ax, label='Credit Score', shrink=0.7)
    ax.set_title('Karnataka — Visual Credit Score by Taluk', fontsize=14, fontweight='bold')
    ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
    fig.savefig(os.path.join(OUT,'Validation_Maps','Karnataka_Visual_Credit_Score.png'))
    plt.close()
    print("  [OK] Credit Score Map")

    # Comparison: NDVI vs Credit Score
    fig, axes = plt.subplots(1, 2, figsize=(18, 10))
    ax1, ax2 = axes
    sc1 = ax1.scatter(taluk_health['lon'], taluk_health['lat'], c=taluk_health['ndvi'],
                      cmap='RdYlGn', s=70, edgecolors='#333', linewidth=0.5, vmin=0.1, vmax=0.8)
    plt.colorbar(sc1, ax=ax1, label='NDVI', shrink=0.7)
    ax1.set_title('Crop Health (NDVI)', fontweight='bold')
    ax1.set_xlabel('Longitude'); ax1.set_ylabel('Latitude')

    sc2 = ax2.scatter(taluk_credit['lon'], taluk_credit['lat'], c=taluk_credit['cs'],
                      cmap='RdYlGn', s=70, edgecolors='#333', linewidth=0.5, vmin=350, vmax=700)
    plt.colorbar(sc2, ax=ax2, label='Credit Score', shrink=0.7)
    ax2.set_title('Visual Credit Score', fontweight='bold')
    ax2.set_xlabel('Longitude'); ax2.set_ylabel('Latitude')
    fig.suptitle('Karnataka — Crop Health vs Credit Risk Comparison', fontsize=15, fontweight='bold')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,'Validation_Maps','Comparison_KARNATAKA.png'))
    plt.close()
    print("  [OK] Comparison Map")


# ═══════════════════════════════════════════════════════════════════════════
# 4. XAI / SHAP ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def plot_shap_analysis():
    # Feature importance bar charts from model_metrics.json (SHAP-like)
    models = [
        ('Model_A_CropHealth', 'Model A: Crop Health'),
        ('Model_B_Soil', 'Model B: Soil Quality'),
        ('Model_C_Water', 'Model C: Water Availability'),
        ('Model_D_Credit_Risk', 'Model D: Credit Risk'),
    ]

    # Individual SHAP bar plots
    for key, title in models:
        fi = M[key].get('feature_importances', {})
        if not fi: continue
        feats = list(fi.keys())[:12]
        vals = [fi[f] for f in feats]

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(range(len(feats)), vals[::-1], color=COLORS[0], alpha=0.85)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats[::-1])
        ax.set_xlabel('Feature Importance (Gini / Gain)')
        ax.set_title(f'{title} — Feature Importance', fontsize=13, fontweight='bold')
        for bar, v in zip(bars, vals[::-1]):
            ax.text(bar.get_width()+0.003, bar.get_y()+bar.get_height()/2,
                    f'{v:.4f}', va='center', fontsize=8)
        plt.tight_layout()
        fname = f'SHAP_{key}_Bar.png'
        fig.savefig(os.path.join(OUT, 'XAI_SHAP_Analysis', fname))
        plt.close()
        print(f"  [OK] {fname}")

    # Combined feature importance
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('TerraTrust — Feature Importance Analysis (All Models)', fontsize=15, fontweight='bold')
    for idx, (key, title) in enumerate(models):
        ax = axes[idx//2][idx%2]
        fi = M[key].get('feature_importances', {})
        feats = list(fi.keys())[:8]
        vals = [fi[f] for f in feats]
        ax.barh(range(len(feats)), vals[::-1], color=COLORS[idx], alpha=0.85)
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels(feats[::-1], fontsize=8)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Importance')
    plt.tight_layout()
    fig.savefig(os.path.join(OUT,'XAI_SHAP_Analysis','SHAP_All_Models_Feature_Importance.png'))
    plt.close()
    print("  [OK] Combined Feature Importance")

# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("TerraTrust — Visualization Generator")
    print(f"Output: {OUT}")
    print("=" * 60)

    print("\n[1/4] System Architecture...")
    draw_pipeline_overview()
    draw_model_architecture()

    print("\n[2/4] Training Curves...")
    plot_training_curves()

    print("\n[3/4] Validation Maps...")
    plot_validation_maps()

    print("\n[4/4] XAI / SHAP Analysis...")
    plot_shap_analysis()

    print("\n" + "=" * 60)
    print("ALL VISUALIZATIONS GENERATED")
    print(f"Output: {OUT}")
    print("=" * 60)
