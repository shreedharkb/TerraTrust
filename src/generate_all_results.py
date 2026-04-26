"""
TerraTrust — Master Results & Visualization Generator
======================================================
Auto-cleans previous results and generates publication-quality plots in 4 folders:
  1. System_Architecture/   — Pipeline block diagram
  2. Training_Curves/       — Loss & accuracy curves for all models
  3. Comparison_Maps/       — Choropleth maps (NDVI vs Credit Score)
  4. SHAP_Analysis/         — Beeswarm, bar, feature impact plots
"""

import os, sys, shutil, json
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
RESULTS_DIR = os.path.join(BASE_DIR, 'results_and_visualizations')

# AUTO-DELETE previous folder
if os.path.exists(RESULTS_DIR):
    print(f"[*] Cleaning up previous plots... Deleting {RESULTS_DIR}")
    shutil.rmtree(RESULTS_DIR)

ARCH_DIR   = os.path.join(RESULTS_DIR, 'System_Architecture')
CURVES_DIR = os.path.join(RESULTS_DIR, 'Training_Curves')
MAPS_DIR   = os.path.join(RESULTS_DIR, 'Comparison_Maps')
SHAP_DIR   = os.path.join(RESULTS_DIR, 'SHAP_Analysis')

for d in [ARCH_DIR, CURVES_DIR, MAPS_DIR, SHAP_DIR]:
    os.makedirs(d, exist_ok=True)
print(f"[*] Created fresh directory structure in: {RESULTS_DIR}")

# Publication styling
plt.rcParams.update({
    'font.size': 12, 'figure.dpi': 300,
    'axes.spines.top': False, 'axes.spines.right': False,
    'font.family': 'sans-serif'
})

# ── Robust Merge (exact name match — NO uppercasing) ─────────────────────────
def robust_spatial_merge(df, gdf):
    """Merge CSV data with GeoJSON using exact taluk name match."""
    geo_col = 'KGISTalukN'
    for candidate in ['KGISTalukN', 'Taluk_Name', 'TALUK']:
        if candidate in gdf.columns:
            geo_col = candidate
            break
    # Clean both sides
    df = df.copy()
    df['_merge_key'] = df['taluk'].astype(str).str.strip()
    gdf = gdf.copy()
    gdf['_merge_key'] = gdf[geo_col].astype(str).str.strip()
    merged = gdf.merge(df, on='_merge_key', how='left')
    matched = merged[merged['taluk'].notna()].shape[0]
    total = gdf.shape[0]
    print(f"    Spatial merge: {matched}/{total} taluks matched ({matched/total*100:.1f}%)")
    return merged


# ══════════════════════════════════════════════════════════════════════════════
# FOLDER 1: System Architecture
# ══════════════════════════════════════════════════════════════════════════════
def generate_architecture_diagrams():
    print("Generating System Architecture Diagrams...")
    import matplotlib.patheffects as pe
    
    # Enable nicer fonts
    plt.rcParams['font.family'] = 'sans-serif'
    
    # PREMIUM LIGHT THEME
    bg_color = '#F8FAFC'
    fig, ax = plt.subplots(figsize=(16, 9), facecolor=bg_color)
    ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis('off')

    colors = {
        'data': '#3B82F6', 'process': '#8B5CF6', 'model': '#10B981',
        'risk': '#EF4444', 'front': '#F59E0B'
    }

    # Soft UI shadow
    shadow = [pe.withSimplePatchShadow(offset=(2.5,-2.5), shadow_rgbFace='#CBD5E1', alpha=0.6)]

    def premium_box(x, y, w, h, txt, color, title=None):
        # Main white card
        patch = mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.2",
            facecolor='#FFFFFF', edgecolor='#E2E8F0', linewidth=1.5, path_effects=shadow, zorder=2)
        ax.add_patch(patch)
        
        # Color accent line on the left
        accent = mpatches.FancyBboxPatch(
            (x-0.18, y-0.18), 0.1, h+0.36, boxstyle="round,pad=0.0",
            facecolor=color, edgecolor='none', zorder=3)
        ax.add_patch(accent)
        
        # Text
        if title:
            ax.text(x+w/2, y+h-0.2, title, ha='center', va='center',
                    fontsize=11, fontweight='bold', color=color, zorder=4)
            ax.text(x+w/2, y+h/2 - 0.2, txt, ha='center', va='center',
                    fontsize=10, fontweight='medium', color='#334155', zorder=4, linespacing=1.6)
        else:
            ax.text(x+w/2, y+h/2, txt, ha='center', va='center',
                    fontsize=10, fontweight='bold', color='#1E293B', zorder=4, linespacing=1.5)

    def arrow(x1, y1, x2, y2, rad=0.0):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->,head_width=0.4,head_length=0.6",
                                    lw=2.5, color='#94A3B8',
                                    connectionstyle=f"arc3,rad={rad}"), zorder=1)

    # Title
    ax.text(8, 8.5, 'TerraTrust — End-to-End Evaluation Pipeline',
            ha='center', fontsize=22, fontweight='heavy', color='#0F172A')

    # Data Sources
    ax.text(1.5, 7.7, 'SATELLITE & SENSOR DATA', ha='center', fontsize=12, fontweight='bold', color='#64748B')
    premium_box(0.3, 6.6, 2.4, 0.7, 'KGIS Shapefiles\n(Taluk Boundaries)', colors['data'])
    premium_box(0.3, 5.4, 2.4, 0.7, 'ISRIC SoilGrids\n(250m Resolution)', colors['data'])
    premium_box(0.3, 4.2, 2.4, 0.7, 'NASA POWER\n(Climate 2019-23)', colors['data'])
    premium_box(0.3, 3.0, 2.4, 0.7, 'Landsat/Sentinel-2\n(NDVI Indices)', colors['data'])
    premium_box(0.3, 1.8, 2.4, 0.7, 'NASA GRACE-FO\n(Groundwater)', colors['data'])

    # Preprocessing
    premium_box(4.2, 3.8, 2.6, 3.0, '• Spatial Join\n• Missing Data Imputation\n• Physics Feature Eng.\n• Normalization', colors['process'], title="Geospatial Engine")
    
    # Connections
    arrow(2.9, 6.95, 4.1, 5.8, rad=-0.1)
    arrow(2.9, 5.75, 4.1, 5.5, rad=-0.05)
    arrow(2.9, 4.55, 4.1, 5.3, rad=0.0)
    arrow(2.9, 3.35, 4.1, 4.9, rad=0.05)
    arrow(2.9, 2.15, 4.1, 4.5, rad=0.1)

    # ML Models
    ax.text(8.8, 7.7, 'HIERARCHICAL ML ARCHITECTURE', ha='center', fontsize=12, fontweight='bold', color='#64748B')
    premium_box(7.5, 6.1, 2.6, 0.9, 'Model A: Crop Health\n(XGBoost Classifier)', colors['model'])
    premium_box(7.5, 4.6, 2.6, 0.9, 'Model B: Soil Quality\n(Random Forest)', colors['model'])
    premium_box(7.5, 3.1, 2.6, 0.9, 'Model C: Water Depth\n(XGBoost Regressor)', colors['model'])
    
    # Connections
    arrow(7.0, 5.3, 7.4, 6.55, rad=-0.1)
    arrow(7.0, 5.3, 7.4, 5.05, rad=0.0)
    arrow(7.0, 5.3, 7.4, 3.55, rad=0.1)

    # Credit Risk Model
    premium_box(11.2, 4.1, 2.8, 1.8, 'Credit Risk Classifier\n(Hierarchical XGBoost)', colors['risk'], title="Model D")
    
    # Connections
    arrow(10.3, 6.55, 11.1, 5.4, rad=0.1)
    arrow(10.3, 5.05, 11.1, 5.0, rad=0.0)
    arrow(10.3, 3.55, 11.1, 4.6, rad=-0.1)

    # Output
    premium_box(11.5, 0.8, 3.5, 2.0, '• Risk Assessment (0-100)\n• Geospatial Evidence\n• Component Breakdown', colors['front'], title="VISUAL CREDIT SCORE")
    arrow(12.6, 3.9, 13.2, 2.9, rad=-0.2)

    plt.savefig(os.path.join(ARCH_DIR, 'TerraTrust_Pipeline_Overview.png'), bbox_inches='tight', facecolor=bg_color, dpi=300)
    plt.close()

    # Data Flow Diagram (Premium Light)
    fig, ax = plt.subplots(figsize=(14, 5), facecolor=bg_color)
    ax.axis('off')
    stages = ['Raw Data\nCollection', 'Cleaning &\nImputation', 'Feature\nEngineering',
              'Model\nTraining', 'Credit\nScoring', 'Dashboard\nVisualization']
    stage_colors = ['#3B82F6', '#6366F1', '#8B5CF6', '#10B981', '#EF4444', '#F59E0B']
    
    for i, (s, c) in enumerate(zip(stages, stage_colors)):
        x = 1.0 + i * 2.2
        # White box with shadow
        patch = mpatches.FancyBboxPatch((x, 1.8), 1.6, 1.2, boxstyle="round,pad=0.2",
                     facecolor='#FFFFFF', edgecolor='#E2E8F0', linewidth=1.5, path_effects=shadow, zorder=2)
        ax.add_patch(patch)
        # Top color accent bar
        accent = mpatches.FancyBboxPatch((x-0.18, 2.8), 1.96, 0.15, boxstyle="round,pad=0.0",
                     facecolor=c, edgecolor='none', zorder=3)
        ax.add_patch(accent)
        
        ax.text(x+0.8, 2.3, s, ha='center', va='center', fontsize=10, fontweight='bold', color='#1E293B', zorder=4)
        if i < 5:
            ax.annotate('', xy=(x+2.1, 2.4), xytext=(x+1.7, 2.4),
                        arrowprops=dict(arrowstyle='->,head_width=0.4,head_length=0.6', lw=3, color='#94A3B8'), zorder=1)
                        
    ax.set_xlim(0, 15); ax.set_ylim(0, 4.5)
    ax.text(7.5, 3.8, 'TerraTrust — Data Processing Flow', ha='center', fontsize=20, fontweight='heavy', color='#0F172A')
    plt.savefig(os.path.join(ARCH_DIR, 'TerraTrust_DataFlow.png'), bbox_inches='tight', facecolor=bg_color, dpi=300)
    plt.close()
    print("  [OK] System Architecture diagrams saved")


# ══════════════════════════════════════════════════════════════════════════════
# FOLDER 2: Training Curves
# ══════════════════════════════════════════════════════════════════════════════
def generate_training_curves():
    print("Generating Training Curves...")
    metrics_path = os.path.join(BASE_DIR, 'results', 'model_metrics.json')
    if not os.path.exists(metrics_path):
        print("  [Skipped] model_metrics.json not found.")
        return

    with open(metrics_path, 'r') as f:
        metrics = json.load(f)

    # Real Learning Curves for Model D
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Try to plot actual evals_result if available
    m_d = metrics.get('Model_D_Credit_Risk', {}).get('spatial', {})
    evals = m_d.get('evals_result', {})
    
    train_loss = evals.get('validation_0', {}).get('mlogloss', [])
    val_loss = evals.get('validation_1', {}).get('mlogloss', [])
    
    if train_loss and val_loss:
        epochs = np.arange(1, len(train_loss) + 1)
        axes[0].plot(epochs, train_loss, label='Train Loss', color='#2563EB', linewidth=2)
        axes[0].plot(epochs, val_loss, label='Val Loss', color='#EF4444', linewidth=2)
        axes[0].set_title('Model D — Log Loss', fontweight='bold')
        axes[0].set_xlabel('Boosting Rounds'); axes[0].set_ylabel('Log Loss')
        axes[0].legend(); axes[0].grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'Training logs not available.\nPlease retrain with eval_set.',
                     ha='center', va='center')
        axes[0].set_title('Model D — Log Loss (Unavailable)')

    # Confusion matrix for Model D
    cm = metrics.get('Model_D_Credit_Risk', {}).get('spatial', {}).get('confusion_matrix', [[0]])
    cm = np.array(cm)
    if cm.size > 1:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1],
                    xticklabels=['High', 'Low', 'Moderate'],
                    yticklabels=['High', 'Low', 'Moderate'])
        axes[1].set_title('Model D — Confusion Matrix', fontweight='bold')
        axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')

    plt.suptitle('TerraTrust — Model D Diagnostics', fontsize=14, fontweight='bold')
    plt.savefig(os.path.join(CURVES_DIR, 'TerraTrust_ModelD_Diagnostics.png'), bbox_inches='tight')
    plt.close()

    # Model comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    
    model_configs = [
        ('Model_A_CropHealth', 'Model A: Crop Health (XGBoost)', True),
        ('Model_B_Soil', 'Model B: Soil Quality (Random Forest)', True),
        ('Model_C_Water', 'Model C: Water Availability (XGBoost)', False),
        ('Model_D_Credit_Risk', 'Model D: Credit Risk (XGBoost)', True),
    ]
    models_list = []
    for key, title, is_clf in model_configs:
        m = metrics.get(key, {}).get('spatial', {})
        tr = m.get('train_accuracy', m.get('train_r2', 0))
        te = m.get('test_accuracy', m.get('test_r2', 0))
        models_list.append({'Model': title.split(':')[0].strip(), 'Train': tr, 'Test': te})

    comp_df = pd.DataFrame(models_list)
    x = np.arange(len(comp_df))
    width = 0.35
    ax.bar(x - width/2, comp_df['Train'], width, label='Train', color='#2563EB', alpha=0.8)
    ax.bar(x + width/2, comp_df['Test'], width, label='Test', color='#10B981', alpha=0.8)
    for i, row in comp_df.iterrows():
        ax.text(i - width/2, row['Train'] + 0.01, f"{row['Train']:.3f}", ha='center', fontsize=9)
        ax.text(i + width/2, row['Test'] + 0.01, f"{row['Test']:.3f}", ha='center', fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(comp_df['Model'])
    ax.set_ylabel('Score'); ax.set_title('Train vs Test Performance Comparison', fontweight='bold')
    ax.legend(); ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 1.1)
    plt.savefig(os.path.join(CURVES_DIR, 'TerraTrust_Model_Comparison.png'), bbox_inches='tight')
    plt.close()
    print("  [OK] Training curves saved")


# ══════════════════════════════════════════════════════════════════════════════
# FOLDER 3: Comparison Maps
# ══════════════════════════════════════════════════════════════════════════════
def generate_comparison_maps():
    print("Generating Comparison Maps...")
    df_path = os.path.join(DATA_DIR, 'karnataka_master_dataset.csv')
    geo_path = os.path.join(DATA_DIR, 'karnataka_taluks.geojson')
    if not os.path.exists(df_path) or not os.path.exists(geo_path):
        print("  [Skipped] Map data missing.")
        return

    df = pd.read_csv(df_path)
    gdf = gpd.read_file(geo_path)

    risk_map = {'Low': 85, 'Moderate': 60, 'High': 35}
    df['heuristic_credit_score'] = df['loan_risk_3class'].map(risk_map).fillna(50)

    taluk_scores = df.groupby('taluk').agg({
        'heuristic_credit_score': 'mean',
        'ndvi_annual_mean': 'mean',
        'groundwater_depth_m': 'mean'
    }).reset_index()

    merged = robust_spatial_merge(taluk_scores, gdf)

    # Dual Choropleth: NDVI vs Credit Score
    fig, axes = plt.subplots(1, 2, figsize=(18, 8))
    plt.subplots_adjust(wspace=0.05)

    merged.plot(column='ndvi_annual_mean', cmap='YlGn', linewidth=0.3, ax=axes[0],
                edgecolor='#333', legend=True, missing_kwds={'color': '#E2E8F0'},
                legend_kwds={'label': 'Mean NDVI', 'orientation': 'horizontal', 'shrink': 0.6})
    axes[0].set_title('Satellite Vegetation Index (NDVI)\nCrop Health by Taluk', fontsize=13, fontweight='bold')
    axes[0].axis('off')

    merged.plot(column='heuristic_credit_score', cmap='RdYlGn', linewidth=0.3, ax=axes[1],
                edgecolor='#333', legend=True, missing_kwds={'color': '#E2E8F0'},
                legend_kwds={'label': 'Credit Score (0-100)', 'orientation': 'horizontal', 'shrink': 0.6})
    axes[1].set_title('TerraTrust Intelligence\nVisual Credit Score by Taluk', fontsize=13, fontweight='bold')
    axes[1].axis('off')

    plt.suptitle('Research Comparison: Satellite NDVI vs. Native Credit Risk Assessment',
                 fontsize=15, fontweight='bold', y=0.98)
    plt.savefig(os.path.join(MAPS_DIR, 'Comparison_NDVI_vs_Credit.png'), bbox_inches='tight')
    plt.close()

    # Groundwater depth map
    fig, ax = plt.subplots(figsize=(10, 8))
    merged.plot(column='groundwater_depth_m', cmap='Blues_r', linewidth=0.3, ax=ax,
                edgecolor='#333', legend=True, missing_kwds={'color': '#E2E8F0'},
                legend_kwds={'label': 'Groundwater Depth (m)', 'orientation': 'horizontal', 'shrink': 0.6})
    ax.set_title('Karnataka — Groundwater Depth by Taluk (NASA GRACE-FO)', fontsize=13, fontweight='bold')
    ax.axis('off')
    plt.savefig(os.path.join(MAPS_DIR, 'Groundwater_Depth_Map.png'), bbox_inches='tight')
    plt.close()

    # District risk distribution
    if 'district' in df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))
        dist_risk = df.groupby('district')['loan_risk_3class'].value_counts(normalize=True).unstack(fill_value=0)
        for col in ['Low', 'Moderate', 'High']:
            if col not in dist_risk.columns:
                dist_risk[col] = 0
        dist_risk = dist_risk[['Low', 'Moderate', 'High']]
        dist_risk.plot(kind='barh', stacked=True, ax=ax, color=['#10B981', '#F59E0B', '#EF4444'], alpha=0.85)
        ax.set_xlabel('Proportion'); ax.set_title('District-wise Loan Risk Distribution', fontweight='bold')
        ax.legend(title='Risk Category', loc='lower right')
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(MAPS_DIR, 'District_Risk_Distribution.png'), bbox_inches='tight')
        plt.close()

    print("  [OK] Comparison maps saved")


# ══════════════════════════════════════════════════════════════════════════════
# FOLDER 4: SHAP Analysis
# ══════════════════════════════════════════════════════════════════════════════
def generate_shap_plots():
    print("Generating SHAP Explainability Plots...")
    df_path = os.path.join(DATA_DIR, 'karnataka_master_dataset.csv')
    model_path = os.path.join(BASE_DIR, 'models', 'model_d_credit.pkl')

    df = pd.read_csv(df_path)
    model_artifact = joblib.load(model_path)
    model = model_artifact['model']
    scaler = model_artifact['scaler']
    features = model_artifact['features']

    # Compute stacked prediction features from models A, B, C
    for model_name, pred_col in [('model_a_ndvi.pkl', 'pred_crop_health'),
                                  ('model_b_soil.pkl', 'pred_soil_q'),
                                  ('model_c_water.pkl', 'pred_water_depth')]:
        mpath = os.path.join(BASE_DIR, 'models', model_name)
        if os.path.exists(mpath):
            art = joblib.load(mpath)
            sub_feats = [f for f in art['features'] if f in df.columns]
            mask = df[sub_feats].notna().all(axis=1)
            if mask.sum() > 0:
                df.loc[mask, pred_col] = art['model'].predict(
                    art['scaler'].transform(df.loc[mask, sub_feats].values))

    available_features = [f for f in features if f in df.columns]
    X = df[available_features].dropna()
    X_scaled = scaler.transform(X.values)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_scaled)

    # For multiclass, pick class index or use average
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]  # "Low risk" class
    else:
        shap_to_plot = shap_values

    bg_color = '#F8FAFC'
    
    # Beeswarm plot
    plt.figure(figsize=(10, 6), facecolor=bg_color)
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    shap.summary_plot(shap_to_plot, X, show=False, cmap="coolwarm",
                      feature_names=[f.replace('_', ' ').title() for f in features if f in df.columns])
    plt.title("SHAP Summary: Drivers of Credit Risk (Beeswarm)", fontweight='bold', fontsize=14, color='#0F172A', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "SHAP_Summary_Beeswarm.png"), bbox_inches='tight', facecolor=bg_color, dpi=300)
    plt.close()

    # Bar plot
    plt.figure(figsize=(10, 6), facecolor=bg_color)
    ax = plt.gca()
    ax.set_facecolor(bg_color)
    shap.summary_plot(shap_to_plot, X, plot_type="bar", show=False,
                      feature_names=[f.replace('_', ' ').title() for f in features if f in df.columns])
    plt.title("SHAP Feature Importance (Bar Chart)", fontweight='bold', fontsize=14, color='#0F172A', pad=15)
    plt.tight_layout()
    plt.savefig(os.path.join(SHAP_DIR, "SHAP_Summary_Bar.png"), bbox_inches='tight', facecolor=bg_color, dpi=300)
    plt.close()

    print("  [OK] Real SHAP plots generated")



    # Feature correlation heatmap
    try:
        df2 = pd.read_csv(df_path)
        corr_cols = ['ndvi_annual_mean', 'groundwater_depth_m', 'organic_carbon_dg_per_kg',
                     'avg_monthly_rainfall_mm', 'avg_root_zone_wetness', 'thermal_stress',
                     'soil_fertility_index', 'aridity_index']
        corr_cols = [c for c in corr_cols if c in df2.columns]
        fig, ax = plt.subplots(figsize=(10, 8), facecolor=bg_color)
        ax.set_facecolor(bg_color)
        corr = df2[corr_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                    ax=ax, square=True, linewidths=1.5, linecolor='white')
        ax.set_title('Feature Correlation Matrix', fontweight='bold', fontsize=15, color='#0F172A', pad=15)
        plt.tight_layout()
        plt.savefig(os.path.join(SHAP_DIR, 'Feature_Correlation_Matrix.png'), bbox_inches='tight', facecolor=bg_color, dpi=300)
        plt.close()
    except Exception:
        pass

    print("  [OK] SHAP analysis plots saved")


# ══════════════════════════════════════════════════════════════════════════════
# Execute
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("=" * 60)
    print("TerraTrust Master Plot Generator (Auto-Cleaning Enabled)")
    print("=" * 60)

    generate_architecture_diagrams()
    generate_training_curves()
    generate_comparison_maps()
    generate_shap_plots()

    # Count generated files
    total = 0
    for root, dirs, files in os.walk(RESULTS_DIR):
        for f in files:
            if f.endswith('.png'):
                total += 1

    print("\n" + "=" * 60)
    print(f"SUCCESS: {total} plots generated in {RESULTS_DIR}")
    for d in [ARCH_DIR, CURVES_DIR, MAPS_DIR, SHAP_DIR]:
        count = len([f for f in os.listdir(d) if f.endswith('.png')])
        print(f"  {os.path.basename(d)}: {count} plots")
    print("=" * 60)
