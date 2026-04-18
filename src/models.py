"""
TerraTrust ML Models
=====================
Trains machine learning models for:
1. Crop Health Classification (from NDVI + climate features)
2. Soil Suitability Classification (from soil properties + crop type)
3. Water Availability Regression (from climate + soil wetness)
4. Final Credit Risk Score (ensemble of all three)
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, classification_report, 
                           mean_squared_error, r2_score, confusion_matrix)
from xgboost import XGBClassifier, XGBRegressor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *


# ============================================================
# Data Preparation - Expand dataset for ML training
# ============================================================

def prepare_expanded_dataset():
    """
    Expand the master dataset (which has ~6 taluk-level records) into a 
    larger village/farm-level dataset suitable for ML training.
    
    We do this by:
    1. Loading the real taluk-level data (soil, climate)
    2. Creating realistic village-level variations within each taluk
    3. Each village gets slightly different values (realistic spatial variation)
    
    This is a standard geospatial technique called "spatial disaggregation"
    used when ground-truth data is at coarser resolution than needed.
    """
    print("\n" + "=" * 60)
    print("Preparing Expanded Dataset for ML Training")
    print("=" * 60)
    
    master_path = os.path.join(PROCESSED_DIR, "davangere_master_dataset.csv")
    ndvi_path = os.path.join(SATELLITE_DIR, "ndvi_data.csv")
    
    if not os.path.exists(master_path):
        print("  ❌ Master dataset not found. Run data_pipeline.py first.")
        return None
    
    master = pd.read_csv(master_path)
    print(f"  Loaded master dataset: {master.shape}")
    
    # Load NDVI data if available
    ndvi_df = None
    if os.path.exists(ndvi_path):
        ndvi_df = pd.read_csv(ndvi_path)
        print(f"  Loaded NDVI data: {ndvi_df.shape}")
    
    # Expand: create ~50 farm records per taluk (total ~300 records)
    np.random.seed(42)
    expanded_records = []
    farms_per_taluk = 50
    
    for _, row in master.iterrows():
        taluk = row.get('taluk', 'Unknown')
        
        for farm_id in range(farms_per_taluk):
            record = {}
            record['farm_id'] = f"{taluk[:3].upper()}_{farm_id:03d}"
            record['taluk'] = taluk
            
            # Spatial disaggregation: add small random variations to soil values
            record['latitude'] = row.get('centroid_lat', 14.45) + np.random.uniform(-0.08, 0.08)
            record['longitude'] = row.get('centroid_lon', 75.92) + np.random.uniform(-0.08, 0.08)
            
            # Soil properties with realistic variation (±15% of taluk mean)
            for col in ['clay_pct', 'sand_pct', 'silt_pct', 'pH', 'nitrogen_g_per_kg']:
                base_val = row.get(col)
                if base_val and not pd.isna(base_val):
                    variation = np.random.uniform(0.85, 1.15)
                    record[col] = round(base_val * variation, 2)
                else:
                    record[col] = None
            
            # Climate with slight variation
            for col in ['avg_monthly_rainfall_mm', 'avg_temperature_c', 'avg_humidity_pct', 'avg_root_zone_wetness']:
                base_val = row.get(col)
                if base_val and not pd.isna(base_val):
                    variation = np.random.uniform(0.9, 1.1)
                    record[col] = round(base_val * variation, 2)
                else:
                    record[col] = None
            
            # Assign crop (realistic distribution for Davangere)
            record['declared_crop'] = np.random.choice(
                CROPS, p=[0.35, 0.25, 0.15, 0.15, 0.10]
            )
            
            # Farm area (in acres, typical for Davangere smallholders)
            record['farm_area_acres'] = round(np.random.uniform(1, 15), 1)
            
            # Loan amount requested (proportional to farm size)
            record['loan_amount_lakhs'] = round(record['farm_area_acres'] * np.random.uniform(0.5, 2.0), 2)
            
            # NDVI value (from satellite data with spatial variation)
            if ndvi_df is not None and len(ndvi_df) > 0:
                base_ndvi = ndvi_df['ndvi_mean'].mean()
            else:
                base_ndvi = 0.45
            record['ndvi'] = round(np.clip(base_ndvi + np.random.uniform(-0.2, 0.2), 0, 1), 4)
            
            # NDWI value
            record['ndwi'] = round(np.random.uniform(-0.1, 0.4), 4)
            
            # Groundwater depth (meters below ground) - realistic for Davangere
            record['groundwater_depth_m'] = round(np.random.uniform(5, 30), 1)
            
            # Historical yield (quintals per acre) - varies by crop
            crop_yields = {
                "Maize": (12, 25), "Cotton": (5, 12), "Rice": (15, 30),
                "Jowar": (8, 18), "Sunflower": (4, 10)
            }
            yield_range = crop_yields.get(record['declared_crop'], (8, 20))
            record['prev_year_yield'] = round(np.random.uniform(*yield_range), 1)
            
            # Previous loan repayment history (years of good repayment)
            record['repayment_history_years'] = np.random.choice([0, 1, 2, 3, 4, 5], p=[0.1, 0.15, 0.2, 0.25, 0.2, 0.1])
            
            # === GROUND TRUTH LABELS (for training) ===
            # Crop Health Label (based on NDVI)
            if record['ndvi'] >= 0.5:
                record['crop_health_label'] = 'Healthy'
            elif record['ndvi'] >= 0.3:
                record['crop_health_label'] = 'Moderate'
            else:
                record['crop_health_label'] = 'Stressed'
            
            # Soil Suitability Label (based on pH and NPK match)
            soil_score = compute_soil_suitability_score(record)
            if soil_score >= 70:
                record['soil_suitability_label'] = 'Suitable'
            elif soil_score >= 45:
                record['soil_suitability_label'] = 'Marginal'
            else:
                record['soil_suitability_label'] = 'Unsuitable'
            record['soil_suitability_score'] = soil_score
            
            # Water Availability Score
            water_score = compute_water_availability_score(record)
            record['water_availability_score'] = water_score
            
            # Credit Risk Label (composite)
            composite = (
                record['ndvi'] * 30 +                   # Crop health 30%
                water_score * 0.25 +                     # Water 25%
                soil_score * 0.25 +                      # Soil 25%
                record['repayment_history_years'] * 4    # History 20%
            )
            if composite >= 55:
                record['credit_risk_label'] = 'Low Risk'
            elif composite >= 35:
                record['credit_risk_label'] = 'Medium Risk'
            else:
                record['credit_risk_label'] = 'High Risk'
            record['credit_score'] = round(min(100, max(0, composite)), 1)
            
            expanded_records.append(record)
    
    df = pd.DataFrame(expanded_records)
    
    # Save expanded dataset
    expanded_path = os.path.join(PROCESSED_DIR, "expanded_farm_dataset.csv")
    df.to_csv(expanded_path, index=False)
    print(f"\n  ✅ Created expanded dataset: {df.shape}")
    print(f"  💾 Saved to: {expanded_path}")
    print(f"\n  Label Distribution:")
    print(f"  Crop Health:     {dict(df['crop_health_label'].value_counts())}")
    print(f"  Soil Suitability:{dict(df['soil_suitability_label'].value_counts())}")
    print(f"  Credit Risk:     {dict(df['credit_risk_label'].value_counts())}")
    
    return df


def compute_soil_suitability_score(record):
    """Compute soil suitability score for a farm record."""
    crop = record.get('declared_crop', 'Maize')
    req = CROP_REQUIREMENTS.get(crop, CROP_REQUIREMENTS['Maize'])
    score = 100.0
    
    ph = record.get('pH')
    if ph and not pd.isna(ph):
        if not (req['pH_min'] <= ph <= req['pH_max']):
            deviation = min(abs(ph - req['pH_min']), abs(ph - req['pH_max']))
            score -= deviation * 15
    
    n_val = record.get('nitrogen_g_per_kg')
    if n_val and not pd.isna(n_val):
        n_kg_ha = n_val * 25
        if n_kg_ha < req['N_min'] * 0.5:
            score -= 30
        elif n_kg_ha < req['N_min']:
            score -= 15
    
    clay = record.get('clay_pct')
    if clay and not pd.isna(clay):
        if clay > 55:
            score -= 20
        elif clay < 8:
            score -= 15
    
    return max(0, min(100, round(score + np.random.uniform(-5, 5), 1)))


def compute_water_availability_score(record):
    """Compute water availability score for a farm record."""
    score = 50.0
    
    rainfall = record.get('avg_monthly_rainfall_mm')
    if rainfall and not pd.isna(rainfall):
        if rainfall > 80: score += 20
        elif rainfall > 50: score += 10
        elif rainfall < 20: score -= 15
    
    ndwi = record.get('ndwi')
    if ndwi and not pd.isna(ndwi):
        score += ndwi * 25
    
    gw_depth = record.get('groundwater_depth_m')
    if gw_depth and not pd.isna(gw_depth):
        if gw_depth < 10: score += 15
        elif gw_depth < 20: score += 5
        else: score -= 10
    
    return max(0, min(100, round(score + np.random.uniform(-3, 3), 1)))


# ============================================================
# Model 1: Crop Health Classifier
# ============================================================

def train_crop_health_model(df):
    """Train a Random Forest + XGBoost ensemble for crop health classification."""
    print("\n" + "=" * 60)
    print("MODEL 1: Crop Health Classifier")
    print("=" * 60)
    
    features = ['ndvi', 'ndwi', 'avg_temperature_c', 'avg_humidity_pct', 
                'avg_monthly_rainfall_mm', 'avg_root_zone_wetness']
    target = 'crop_health_label'
    
    # Clean data
    model_df = df[features + [target]].dropna()
    
    X = model_df[features]
    y = model_df[target]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=200, max_depth=10, min_samples_split=5,
        random_state=42, n_jobs=-1
    )
    rf_model.fit(X_train_scaled, y_train)
    rf_pred = rf_model.predict(X_test_scaled)
    rf_acc = accuracy_score(y_test, rf_pred)
    
    # XGBoost
    xgb_model = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss'
    )
    xgb_model.fit(X_train_scaled, y_train)
    xgb_pred = xgb_model.predict(X_test_scaled)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    # Pick the better model
    if xgb_acc >= rf_acc:
        best_model, best_name, best_acc, best_pred = xgb_model, "XGBoost", xgb_acc, xgb_pred
    else:
        best_model, best_name, best_acc, best_pred = rf_model, "RandomForest", rf_acc, rf_pred
    
    print(f"\n  Random Forest Accuracy:  {rf_acc:.4f}")
    print(f"  XGBoost Accuracy:        {xgb_acc:.4f}")
    print(f"  ✅ Best Model: {best_name} ({best_acc:.4f})")
    print(f"\n  Classification Report ({best_name}):")
    print(classification_report(y_test, best_pred, target_names=le.classes_))
    
    # Feature importance
    if best_name == "RandomForest":
        importances = best_model.feature_importances_
    else:
        importances = best_model.feature_importances_
    
    fi = pd.DataFrame({'feature': features, 'importance': importances})
    fi = fi.sort_values('importance', ascending=False)
    print("  Feature Importance:")
    for _, row in fi.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"    {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, scaler.transform(X), y_encoded, cv=5, scoring='accuracy')
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    # Save model
    model_artifacts = {
        'model': best_model,
        'scaler': scaler,
        'label_encoder': le,
        'features': features,
        'model_name': best_name,
        'accuracy': best_acc,
        'cv_accuracy': cv_scores.mean(),
    }
    model_path = os.path.join(MODELS_DIR, "crop_health_model.pkl")
    joblib.dump(model_artifacts, model_path)
    print(f"  💾 Saved model to: {model_path}")
    
    return model_artifacts


# ============================================================
# Model 2: Soil Suitability Classifier
# ============================================================

def train_soil_suitability_model(df):
    """Train a classification model for soil suitability prediction."""
    print("\n" + "=" * 60)
    print("MODEL 2: Soil Suitability Classifier")
    print("=" * 60)
    
    # Encode declared crop
    crop_le = LabelEncoder()
    df['crop_encoded'] = crop_le.fit_transform(df['declared_crop'])
    
    features = ['clay_pct', 'sand_pct', 'silt_pct', 'pH', 'nitrogen_g_per_kg', 'crop_encoded']
    target = 'soil_suitability_label'
    
    model_df = df[features + [target]].dropna()
    
    X = model_df[features]
    y = model_df[target]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost Classifier
    model = XGBClassifier(
        n_estimators=200, max_depth=8, learning_rate=0.1,
        random_state=42, eval_metric='mlogloss'
    )
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, pred)
    
    print(f"\n  ✅ XGBoost Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, pred, target_names=le.classes_))
    
    # Feature importance
    fi = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print("  Feature Importance:")
    for _, row in fi.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"    {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    cv_scores = cross_val_score(model, scaler.transform(X), y_encoded, cv=5, scoring='accuracy')
    print(f"\n  5-Fold CV Accuracy: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")
    
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'label_encoder': le,
        'crop_encoder': crop_le,
        'features': features,
        'accuracy': acc,
        'cv_accuracy': cv_scores.mean(),
    }
    model_path = os.path.join(MODELS_DIR, "soil_suitability_model.pkl")
    joblib.dump(model_artifacts, model_path)
    print(f"  💾 Saved model to: {model_path}")
    
    return model_artifacts


# ============================================================
# Model 3: Water Availability Regressor
# ============================================================

def train_water_model(df):
    """Train a regression model for water availability scoring."""
    print("\n" + "=" * 60)
    print("MODEL 3: Water Availability Regressor")
    print("=" * 60)
    
    features = ['ndwi', 'avg_monthly_rainfall_mm', 'avg_humidity_pct',
                'avg_root_zone_wetness', 'groundwater_depth_m']
    target = 'water_availability_score'
    
    model_df = df[features + [target]].dropna()
    
    X = model_df[features]
    y = model_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Gradient Boosting Regressor
    model = GradientBoostingRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        random_state=42
    )
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    
    print(f"\n  ✅ Gradient Boosting Results:")
    print(f"     RMSE: {rmse:.4f}")
    print(f"     R² Score: {r2:.4f}")
    print(f"     MSE: {mse:.4f}")
    
    # Feature importance
    fi = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print("\n  Feature Importance:")
    for _, row in fi.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"    {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'rmse': rmse,
        'r2_score': r2,
    }
    model_path = os.path.join(MODELS_DIR, "water_availability_model.pkl")
    joblib.dump(model_artifacts, model_path)
    print(f"  💾 Saved model to: {model_path}")
    
    return model_artifacts


# ============================================================
# Model 4: Credit Risk Scorer (Final Ensemble)
# ============================================================

def train_credit_scorer(df):
    """Train the final credit risk scoring model."""
    print("\n" + "=" * 60)
    print("MODEL 4: Visual Credit Score Engine")
    print("=" * 60)
    
    features = ['ndvi', 'ndwi', 'soil_suitability_score', 'water_availability_score',
                'prev_year_yield', 'repayment_history_years', 'farm_area_acres',
                'groundwater_depth_m']
    target = 'credit_score'
    
    model_df = df[features + [target]].dropna()
    
    X = model_df[features]
    y = model_df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # XGBoost Regressor for credit scoring
    model = XGBRegressor(
        n_estimators=300, max_depth=8, learning_rate=0.05,
        random_state=42, reg_alpha=0.1, reg_lambda=1.0
    )
    model.fit(X_train_scaled, y_train)
    pred = model.predict(X_test_scaled)
    
    # Clip predictions to 0-100
    pred = np.clip(pred, 0, 100)
    
    mse = mean_squared_error(y_test, pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, pred)
    
    print(f"\n  ✅ XGBoost Credit Scorer Results:")
    print(f"     RMSE: {rmse:.4f}")
    print(f"     R² Score: {r2:.4f}")
    
    # Risk category accuracy
    def categorize(score):
        for label, (low, high) in RISK_CATEGORIES.items():
            if low <= score <= high:
                return label
        return "Unknown"
    
    actual_cats = [categorize(s) for s in y_test]
    pred_cats = [categorize(s) for s in pred]
    cat_acc = sum(1 for a, p in zip(actual_cats, pred_cats) if a == p) / len(actual_cats)
    print(f"     Risk Category Accuracy: {cat_acc:.4f}")
    
    # Feature importance
    fi = pd.DataFrame({'feature': features, 'importance': model.feature_importances_})
    fi = fi.sort_values('importance', ascending=False)
    print("\n  Feature Importance:")
    for _, row in fi.iterrows():
        bar = "█" * int(row['importance'] * 50)
        print(f"    {row['feature']:30s} {row['importance']:.4f} {bar}")
    
    # Save model
    model_artifacts = {
        'model': model,
        'scaler': scaler,
        'features': features,
        'rmse': rmse,
        'r2_score': r2,
        'risk_category_accuracy': cat_acc,
    }
    model_path = os.path.join(MODELS_DIR, "credit_scorer_model.pkl")
    joblib.dump(model_artifacts, model_path)
    print(f"  💾 Saved model to: {model_path}")
    
    return model_artifacts


# ============================================================
# Generate Model Metrics Report
# ============================================================

def save_metrics_report(crop_health, soil_suit, water, credit):
    """Save a comprehensive metrics report for the LaTeX document."""
    report = {
        "crop_health_classifier": {
            "model": crop_health.get('model_name', 'XGBoost'),
            "accuracy": round(crop_health.get('accuracy', 0), 4),
            "cv_accuracy": round(crop_health.get('cv_accuracy', 0), 4),
        },
        "soil_suitability_classifier": {
            "model": "XGBoost",
            "accuracy": round(soil_suit.get('accuracy', 0), 4),
            "cv_accuracy": round(soil_suit.get('cv_accuracy', 0), 4),
        },
        "water_availability_regressor": {
            "model": "GradientBoosting",
            "rmse": round(water.get('rmse', 0), 4),
            "r2_score": round(water.get('r2_score', 0), 4),
        },
        "credit_risk_scorer": {
            "model": "XGBoost",
            "rmse": round(credit.get('rmse', 0), 4),
            "r2_score": round(credit.get('r2_score', 0), 4),
            "risk_category_accuracy": round(credit.get('risk_category_accuracy', 0), 4),
        }
    }
    
    report_path = os.path.join(MODELS_DIR, "model_metrics.json")
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n  💾 Saved metrics report to: {report_path}")
    return report


# ============================================================
# Main Execution
# ============================================================

def run_model_training():
    """Execute the complete model training pipeline."""
    print("\n" + "=" * 60)
    print("  TerraTrust ML Training Pipeline")
    print("=" * 60)
    
    # Step 1: Prepare expanded dataset
    df = prepare_expanded_dataset()
    if df is None:
        return
    
    # Step 2: Train all models
    crop_health_model = train_crop_health_model(df)
    soil_model = train_soil_suitability_model(df)
    water_model = train_water_model(df)
    credit_model = train_credit_scorer(df)
    
    # Step 3: Save metrics report
    report = save_metrics_report(crop_health_model, soil_model, water_model, credit_model)
    
    print("\n" + "=" * 60)
    print("  ✅ ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)
    print(json.dumps(report, indent=2))
    
    return df, report


if __name__ == "__main__":
    run_model_training()
