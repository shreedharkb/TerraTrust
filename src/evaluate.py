"""
TerraTrust Model Evaluation & Report Generator
=================================================
Generates publication-quality plots and metrics tables
for direct inclusion in LaTeX academic reports.

Usage: python -u src/evaluate.py
Output: report/figures/*.png + report/metrics_summary.txt
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    confusion_matrix, classification_report,
    roc_curve, auc, ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *

# ============================================================
# Configuration
# ============================================================
FIGURES_DIR = os.path.join(PROJECT_ROOT, "report", "figures")
REPORT_DIR  = os.path.join(PROJECT_ROOT, "report")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Plot style
plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.family': 'serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.facecolor': 'white',
    'axes.facecolor': '#fafafa',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

COLORS = {
    'primary':   '#10b981',
    'secondary': '#3b82f6',
    'accent':    '#f59e0b',
    'danger':    '#ef4444',
    'dark':      '#1e293b',
}


# ============================================================
# Data Loading (mirrors models.py exactly)
# ============================================================

def load_evaluation_data():
    """Load the master dataset with NDVI merged, same as training."""
    master_path = os.path.join(PROCESSED_DIR, "karnataka_master_dataset.csv")
    ndvi_path   = os.path.join(SATELLITE_DIR, "ndvi_timeseries.csv")

    if not os.path.exists(master_path):
        print("  ❌ karnataka_master_dataset.csv not found. Run data_pipeline.py first.")
        return None

    master = pd.read_csv(master_path)
    print(f"  Loaded master dataset: {master.shape}")

    if os.path.exists(ndvi_path):
        ndvi_df = pd.read_csv(ndvi_path)
        yearly_ndvi = ndvi_df.groupby('year')['ndvi_mean'].mean().reset_index()
        yearly_ndvi.rename(columns={'ndvi_mean': 'ndvi'}, inplace=True)
        if 'year' in master.columns:
            master = master.merge(yearly_ndvi, on='year', how='left')
            master = master.dropna(subset=['ndvi'])
    else:
        print("  ⚠️  NDVI timeseries not found, some evaluations may be limited.")

    print(f"  Final evaluation dataset: {master.shape}")
    return master


# ============================================================
# 1. NDVI Predictor Evaluation
# ============================================================

def evaluate_ndvi_predictor(df, report_lines):
    print("\n" + "=" * 60)
    print("  EVALUATING: NDVI Predictor (XGBoost Regressor)")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "ndvi_predictor_model.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  Model not found, skipping.")
        return

    artifacts = joblib.load(model_path)
    model   = artifacts['model']
    scaler  = artifacts['scaler']
    features = artifacts['features']

    model_df = df[features + ['ndvi']].dropna()
    X = model_df[features]
    y = model_df['ndvi']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae  = np.mean(np.abs(y_test - y_pred))

    # Cross-validation
    cv_scores = cross_val_score(model, scaler.transform(X), y, cv=5, scoring='neg_mean_squared_error')
    cv_rmse = np.sqrt(-cv_scores)

    report_lines.append("\n" + "=" * 50)
    report_lines.append("MODEL 1: NDVI Predictor (XGBoost Regressor)")
    report_lines.append("=" * 50)
    report_lines.append(f"  Features:       {features}")
    report_lines.append(f"  Train samples:  {len(X_train)}")
    report_lines.append(f"  Test samples:   {len(X_test)}")
    report_lines.append(f"  RMSE:           {rmse:.4f}")
    report_lines.append(f"  MAE:            {mae:.4f}")
    report_lines.append(f"  R² Score:       {r2:.4f}")
    report_lines.append(f"  5-Fold CV RMSE: {cv_rmse.mean():.4f} (±{cv_rmse.std():.4f})")

    # --- Plot 1: Actual vs Predicted ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.6, c=COLORS['primary'], edgecolors='white', s=60)
    lims = [min(y_test.min(), y_pred.min()) - 0.02, max(y_test.max(), y_pred.max()) + 0.02]
    ax.plot(lims, lims, '--', color=COLORS['danger'], lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual NDVI')
    ax.set_ylabel('Predicted NDVI')
    ax.set_title(f'NDVI Predictor — Actual vs Predicted\nRMSE={rmse:.4f}  R²={r2:.4f}')
    ax.legend()

    # --- Plot 2: Feature Importance ---
    ax = axes[1]
    importances = model.feature_importances_
    fi = pd.DataFrame({'feature': features, 'importance': importances}).sort_values('importance')
    ax.barh(fi['feature'], fi['importance'], color=COLORS['primary'], edgecolor='white')
    ax.set_xlabel('Feature Importance')
    ax.set_title('NDVI Predictor — Feature Importance')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "01_ndvi_predictor.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {path}")

    # --- Plot 3: Residuals ---
    fig, ax = plt.subplots(figsize=(8, 4))
    residuals = y_test.values - y_pred
    ax.hist(residuals, bins=30, color=COLORS['secondary'], edgecolor='white', alpha=0.8)
    ax.axvline(0, color=COLORS['danger'], linestyle='--', lw=2)
    ax.set_xlabel('Residual (Actual - Predicted)')
    ax.set_ylabel('Frequency')
    ax.set_title(f'NDVI Predictor — Residual Distribution (Mean={np.mean(residuals):.4f})')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "02_ndvi_residuals.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {path}")


# ============================================================
# 2. Soil Suitability Classifier Evaluation
# ============================================================

def evaluate_soil_classifier(df, report_lines):
    print("\n" + "=" * 60)
    print("  EVALUATING: Soil Suitability Classifier")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "soil_classifier_model.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  Model not found, skipping.")
        return

    artifacts = joblib.load(model_path)
    model   = artifacts['model']
    scaler  = artifacts['scaler']
    le      = artifacts['label_encoder']
    features = artifacts['features']

    # Reconstruct target the same way as training
    if 'soil_suitability_label' in df.columns and not df['soil_suitability_label'].isnull().all():
        target_col = 'soil_suitability_label'
    else:
        target_col = 'derived_soil_label'
        bins = [0, 50, 75, 100]
        labels = ['Unsuitable', 'Marginal', 'Suitable']
        df = df.copy()
        df[target_col] = pd.cut(df['soil_suitability_score'], bins=bins, labels=labels, include_lowest=True)

    model_df = df[features + [target_col]].dropna()
    X = model_df[features]
    y_raw = model_df[target_col]
    y = le.transform(y_raw)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    class_names = le.classes_
    cr = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    report_lines.append("\n" + "=" * 50)
    report_lines.append("MODEL 2: Soil Suitability Classifier (RandomForest)")
    report_lines.append("=" * 50)
    report_lines.append(f"  Features:      {features}")
    report_lines.append(f"  Train samples: {len(X_train)}")
    report_lines.append(f"  Test samples:  {len(X_test)}")
    report_lines.append(f"  Accuracy:      {acc:.4f}")
    report_lines.append(f"\n  Classification Report:\n{cr}")

    # --- Confusion Matrix Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    disp = ConfusionMatrixDisplay(cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap='Greens', colorbar=False)
    axes[0].set_title(f'Soil Classifier — Confusion Matrix\nAccuracy={acc:.4f}')

    # Feature Importance
    fi = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance')
    axes[1].barh(fi['feature'], fi['importance'], color=COLORS['accent'], edgecolor='white')
    axes[1].set_xlabel('Feature Importance')
    axes[1].set_title('Soil Classifier — Feature Importance')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "03_soil_classifier.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {path}")


# ============================================================
# 3. Water Availability Regressor Evaluation
# ============================================================

def evaluate_water_regressor(df, report_lines):
    print("\n" + "=" * 60)
    print("  EVALUATING: Water Availability Regressor")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "water_regressor_model.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  Model not found, skipping.")
        return

    artifacts = joblib.load(model_path)
    model   = artifacts['model']
    scaler  = artifacts['scaler']
    features = artifacts['features']

    model_df = df[features + ['water_availability_score']].dropna()
    X = model_df[features]
    y = model_df['water_availability_score']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)
    mae  = np.mean(np.abs(y_test - y_pred))

    report_lines.append("\n" + "=" * 50)
    report_lines.append("MODEL 3: Water Availability Regressor (XGBoost)")
    report_lines.append("=" * 50)
    report_lines.append(f"  Features:      {features}")
    report_lines.append(f"  Train samples: {len(X_train)}")
    report_lines.append(f"  Test samples:  {len(X_test)}")
    report_lines.append(f"  RMSE:          {rmse:.4f}")
    report_lines.append(f"  MAE:           {mae:.4f}")
    report_lines.append(f"  R² Score:      {r2:.4f}")

    # --- Plots ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    ax.scatter(y_test, y_pred, alpha=0.6, c=COLORS['secondary'], edgecolors='white', s=60)
    lims = [min(y_test.min(), y_pred.min()) - 2, max(y_test.max(), y_pred.max()) + 2]
    ax.plot(lims, lims, '--', color=COLORS['danger'], lw=2, label='Perfect prediction')
    ax.set_xlabel('Actual Water Score')
    ax.set_ylabel('Predicted Water Score')
    ax.set_title(f'Water Regressor — Actual vs Predicted\nRMSE={rmse:.4f}  R²={r2:.4f}')
    ax.legend()

    fi = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance')
    axes[1].barh(fi['feature'], fi['importance'], color=COLORS['secondary'], edgecolor='white')
    axes[1].set_xlabel('Feature Importance')
    axes[1].set_title('Water Regressor — Feature Importance')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "04_water_regressor.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {path}")


# ============================================================
# 4. Credit Risk Classifier Evaluation
# ============================================================

def evaluate_credit_risk_classifier(df, report_lines):
    print("\n" + "=" * 60)
    print("  EVALUATING: Credit Risk Classifier (XGBoost)")
    print("=" * 60)

    model_path = os.path.join(MODELS_DIR, "credit_risk_classifier_model.pkl")
    if not os.path.exists(model_path):
        print("  ⚠️  Model not found, skipping.")
        return

    artifacts = joblib.load(model_path)
    model   = artifacts['model']
    scaler  = artifacts['scaler']
    features = artifacts['features']

    if 'ndvi' not in df.columns and 'predicted_ndvi' in df.columns:
        df = df.rename(columns={'predicted_ndvi': 'ndvi'})

    if 'loan_risk_category' in df.columns:
        df['loan_risk_binary'] = df['loan_risk_category'].isin(['Low Risk', 'Very Low Risk']).astype(int)
    else:
        df['loan_risk_binary'] = (df['credit_score'] >= 650).astype(int)
        
    model_df = df[features + ['loan_risk_binary']].dropna()
    X = model_df[features]
    y = model_df['loan_risk_binary'].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    cr  = classification_report(y_test, y_pred, target_names=['Default', 'Good Repayment'], zero_division=0)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    report_lines.append("\n" + "=" * 50)
    report_lines.append("MODEL 4: Credit Risk Classifier (XGBoost)")
    report_lines.append("=" * 50)
    report_lines.append(f"  Features:      {features}")
    report_lines.append(f"  Train samples: {len(X_train)}")
    report_lines.append(f"  Test samples:  {len(X_test)}")
    report_lines.append(f"  Accuracy:      {acc:.4f}")
    report_lines.append(f"  ROC AUC:       {roc_auc:.4f}")
    report_lines.append(f"\n  Classification Report:\n{cr}")
    report_lines.append(f"  Confusion Matrix:\n{cm}")

    # --- 3-panel figure ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Confusion Matrix
    disp = ConfusionMatrixDisplay(cm, display_labels=['Default', 'Repaid'])
    disp.plot(ax=axes[0], cmap='Blues', colorbar=False)
    axes[0].set_title(f'Credit Risk — Confusion Matrix\nAccuracy={acc:.4f}')

    # ROC Curve
    axes[1].plot(fpr, tpr, color=COLORS['primary'], lw=2.5, label=f'ROC (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], '--', color='gray', lw=1)
    axes[1].fill_between(fpr, tpr, alpha=0.15, color=COLORS['primary'])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title('Credit Risk — ROC Curve')
    axes[1].legend(loc='lower right')

    # Feature Importance
    fi = pd.DataFrame({'feature': features, 'importance': model.feature_importances_}).sort_values('importance')
    axes[2].barh(fi['feature'], fi['importance'], color=COLORS['primary'], edgecolor='white')
    axes[2].set_xlabel('Feature Importance')
    axes[2].set_title('Credit Risk — Feature Importance')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "05_credit_risk_classifier.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {path}")


# ============================================================
# 5. Summary Comparison Table
# ============================================================

def generate_summary_table(report_lines):
    """Generate a LaTeX-ready model comparison table."""
    print("\n" + "=" * 60)
    print("  GENERATING: Model Comparison Summary")
    print("=" * 60)

    metrics_path = os.path.join(MODELS_DIR, "model_metrics.json")
    if not os.path.exists(metrics_path):
        print("  ⚠️  model_metrics.json not found, skipping summary table.")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    report_lines.append("\n" + "=" * 50)
    report_lines.append("CONSOLIDATED MODEL COMPARISON")
    report_lines.append("=" * 50)

    rows = []
    for name, m in metrics.items():
        algo = m.get('model', 'N/A')
        acc  = m.get('accuracy', '-')
        rmse = m.get('rmse', '-')
        r2   = m.get('r2_score', '-')
        rows.append([name, algo, str(acc), str(rmse), str(r2)])

    header = f"{'Model':<30} {'Algorithm':<15} {'Accuracy':<10} {'RMSE':<10} {'R²':<10}"
    report_lines.append(header)
    report_lines.append("-" * len(header))
    for r in rows:
        report_lines.append(f"{r[0]:<30} {r[1]:<15} {r[2]:<10} {r[3]:<10} {r[4]:<10}")

    # --- Bar chart comparing models ---
    fig, ax = plt.subplots(figsize=(10, 5))
    model_names = [r[0].replace('_', '\n') for r in rows]
    # Use accuracy for classifiers, (1 - RMSE) for regressors as a unified "performance" metric
    performances = []
    colors = []
    for r in rows:
        if r[2] != '-' and r[2] != 'None':
            performances.append(float(r[2]) * 100)
            colors.append(COLORS['primary'])
        elif r[3] != '-' and r[3] != 'None':
            performances.append(max(0, 100 - float(r[3]) * 100))
            colors.append(COLORS['secondary'])
        else:
            performances.append(0)
            colors.append('gray')

    bars = ax.bar(model_names, performances, color=colors, edgecolor='white', width=0.6)
    ax.set_ylabel('Performance Score (%)')
    ax.set_title('TerraTrust — Model Performance Comparison')
    ax.set_ylim(0, 105)
    for bar, perf in zip(bars, performances):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f'{perf:.1f}%', ha='center', fontweight='bold', fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "06_model_comparison.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {path}")


# ============================================================
# 6. Data Distribution Plots
# ============================================================

def plot_data_distributions(df):
    """Plot distributions of key features for the dataset overview section."""
    print("\n" + "=" * 60)
    print("  GENERATING: Data Distribution Plots")
    print("=" * 60)

    plot_cols = ['pH', 'clay_pct', 'sand_pct', 'nitrogen_g_per_kg',
                 'avg_monthly_rainfall_mm', 'avg_humidity_pct',
                 'soil_suitability_score', 'water_availability_score']
    available = [c for c in plot_cols if c in df.columns]

    if len(available) == 0:
        print("  ⚠️  No plottable columns found.")
        return

    ncols = 4
    nrows = (len(available) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes = axes.flatten() if nrows > 1 else [axes] if len(available) == 1 else axes.flatten()

    for i, col in enumerate(available):
        data = df[col].dropna()
        if len(data) == 0:
            continue
        axes[i].hist(data, bins=30, color=COLORS['primary'], edgecolor='white', alpha=0.8)
        axes[i].set_title(col.replace('_', ' ').title())
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Count')

    # Hide unused subplots
    for j in range(len(available), len(axes)):
        axes[j].set_visible(False)

    plt.suptitle('TerraTrust — Feature Distributions (Karnataka Dataset)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "07_data_distributions.png")
    plt.savefig(path, bbox_inches='tight')
    plt.close()
    print(f"  💾 Saved: {path}")


# ============================================================
# Main Execution
# ============================================================

def run_evaluation():
    start = time.time()

    print("\n" + "=" * 60)
    print("  TerraTrust — Model Evaluation & Report Generator")
    print("=" * 60)

    df = load_evaluation_data()
    if df is None:
        return

    report_lines = []
    report_lines.append("=" * 60)
    report_lines.append(" TerraTrust — Model Evaluation Report")
    report_lines.append(f" Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f" Dataset: karnataka_master_dataset.csv ({df.shape[0]} rows, {df.shape[1]} columns)")
    report_lines.append("=" * 60)

    # Run all evaluations
    evaluate_ndvi_predictor(df, report_lines)
    evaluate_soil_classifier(df, report_lines)
    evaluate_water_regressor(df, report_lines)
    evaluate_credit_risk_classifier(df, report_lines)
    generate_summary_table(report_lines)
    plot_data_distributions(df)

    # Save text report
    elapsed = time.time() - start
    report_lines.append(f"\n\n⏱️  Evaluation completed in {elapsed:.1f}s")

    report_path = os.path.join(REPORT_DIR, "metrics_summary.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report_lines))
    print(f"\n  💾 Full metrics report saved to: {report_path}")

    print("\n" + "=" * 60)
    print(f"  ✅ EVALUATION COMPLETE — {elapsed:.1f}s")
    print(f"  📁 Figures saved to: {FIGURES_DIR}")
    print(f"  📄 Metrics report:   {report_path}")
    print("=" * 60)

    # Print summary to console
    print("\n" + "\n".join(report_lines))


if __name__ == "__main__":
    run_evaluation()
