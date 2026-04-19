"""
TerraTrust ML Models - Step 5 & 6
=================================
Trains all 4 required ML models using purely physical features.
Exports comprehensively detailed json metrics (Train/Test R2, Acc, F1, 5-Fold CV, Confusion Matrices, Importances).
"""

import os
import sys
import json
import time
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    mean_squared_error, r2_score, accuracy_score,
    precision_score, recall_score, f1_score, confusion_matrix
)
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBRegressor, XGBClassifier

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
MASTER_CSV = os.path.join(DATA_DIR, 'karnataka_master_dataset.csv')
METRICS_JSON = os.path.join(MODELS_DIR, 'model_metrics.json')

os.makedirs(MODELS_DIR, exist_ok=True)

XGB_KWARGS = {'tree_method': 'hist', 'device': 'cuda'}

def load_dataset():
    if not os.path.exists(MASTER_CSV):
        raise FileNotFoundError(f"Master dataset not found at {MASTER_CSV}.")
    df = pd.read_csv(MASTER_CSV)
    print(f"Loaded Master Dataset: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def eval_regressor(model, X_train, X_test, y_train, y_test, feat_names, cv=5):
    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    r2_train = r2_score(y_train, y_pred_train)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    
    r2_test = r2_score(y_test, y_pred_test)
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
    
    X_all = np.vstack((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))
    cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='r2')
    
    importances = {}
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        importances = {feat_names[i]: float(fi[i]) for i in range(len(feat_names))}
        
    return {
        "training_time_seconds": round(t_train, 2),
        "train_r2": round(float(r2_train), 4),
        "test_r2": round(float(r2_test), 4),
        "train_rmse": round(float(rmse_train), 4),
        "test_rmse": round(float(rmse_test), 4),
        "cv_5fold_mean": round(float(np.mean(cv_scores)), 4),
        "cv_5fold_std": round(float(np.std(cv_scores)), 4),
        "feature_importances": importances
    }

def eval_classifier(model, X_train, X_test, y_train, y_test, feat_names, cv=5, is_binary=True):
    t0 = time.time()
    model.fit(X_train, y_train)
    t_train = time.time() - t0
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    acc_train = accuracy_score(y_train, y_pred_train)
    acc_test = accuracy_score(y_test, y_pred_test)
    
    avg = 'binary' if is_binary else 'weighted'
    try:
        prec = precision_score(y_test, y_pred_test, average=avg, zero_division=0)
        rec = recall_score(y_test, y_pred_test, average=avg, zero_division=0)
        f1 = f1_score(y_test, y_pred_test, average=avg, zero_division=0)
    except:
        prec, rec, f1 = 0.0, 0.0, 0.0

    cm = confusion_matrix(y_test, y_pred_test)
    
    X_all = np.vstack((X_train, X_test))
    y_all = np.concatenate((y_train, y_test))
    cv_scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='accuracy')

    importances = {}
    if hasattr(model, 'feature_importances_'):
        fi = model.feature_importances_
        importances = {feat_names[i]: float(fi[i]) for i in range(len(feat_names))}
        
    return {
        "training_time_seconds": round(t_train, 2),
        "train_accuracy": round(float(acc_train), 4),
        "test_accuracy": round(float(acc_test), 4),
        "test_precision": round(float(prec), 4),
        "test_recall": round(float(rec), 4),
        "test_f1_score": round(float(f1), 4),
        "cv_5fold_mean": round(float(np.mean(cv_scores)), 4),
        "cv_5fold_std": round(float(np.std(cv_scores)), 4),
        "confusion_matrix": cm.tolist(),
        "feature_importances": importances
    }

def main():
    print("=" * 60)
    print("TerraTrust ML Pipeline - Step 5 & 6 Strict Compliance")
    print("=" * 60)
    df = load_dataset().dropna()

    metrics_export = {}

    print("\n[1/4] Training Model A: NDVI Predictor (XGBoost GPU)")
    feat_a = ['clay_pct', 'sand_pct', 'pH', 'avg_monthly_rainfall_mm', 'groundwater_depth_m']
    target_a = 'ndvi_annual_mean'
    
    if target_a not in df.columns and 'ndvi' in df.columns:
        target_a = 'ndvi'

    df_a = df[feat_a + [target_a]].dropna()
    X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(df_a[feat_a].values, df_a[target_a].values, test_size=0.2, random_state=42)
    scaler_a = StandardScaler()
    model_a = XGBRegressor(**XGB_KWARGS, n_estimators=200, random_state=42)
    metrics_export['Model_A_NDVI'] = eval_regressor(model_a, scaler_a.fit_transform(X_train_a), scaler_a.transform(X_test_a), y_train_a, y_test_a, feat_a)
    joblib.dump({'model': model_a, 'scaler': scaler_a}, os.path.join(MODELS_DIR, 'model_a_ndvi.pkl'))


    print("\n[2/4] Training Model B: Soil Suitability Classifier (RandomForest)")
    feat_b = ['pH', 'nitrogen_g_per_kg', 'clay_pct', 'sand_pct', 'silt_pct']
    # Adjust bins to capture the dataset variance (85, 100) instead of a single class
    df['derived_soil_label'] = pd.cut(df['soil_suitability_score'], bins=[-1, 80, 95, 101], labels=['Unsuitable', 'Marginal', 'Suitable'])
    df_b = df[feat_b + ['derived_soil_label']].dropna()
    le_b = LabelEncoder()
    y_b = le_b.fit_transform(df_b['derived_soil_label'])

    X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(df_b[feat_b].values, y_b, test_size=0.2, random_state=42, stratify=y_b)
    scaler_b = StandardScaler()
    model_b = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    metrics_export['Model_B_Soil'] = eval_classifier(model_b, scaler_b.fit_transform(X_train_b), scaler_b.transform(X_test_b), y_train_b, y_test_b, feat_b, is_binary=False)
    joblib.dump({'model': model_b, 'scaler': scaler_b, 'encoder': le_b}, os.path.join(MODELS_DIR, 'model_b_soil.pkl'))


    print("\n[3/4] Training Model C: Water Availability Regressor (XGBoost GPU)")
    feat_c = ['avg_monthly_rainfall_mm', 'groundwater_depth_m']
    target_c = 'water_availability_score'
    df_c = df[feat_c + [target_c]].dropna()

    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(df_c[feat_c].values, df_c[target_c].values, test_size=0.2, random_state=42)
    scaler_c = StandardScaler()
    model_c = XGBRegressor(**XGB_KWARGS, n_estimators=200, random_state=42)
    metrics_export['Model_C_Water'] = eval_regressor(model_c, scaler_c.fit_transform(X_train_c), scaler_c.transform(X_test_c), y_train_c, y_test_c, feat_c)
    joblib.dump({'model': model_c, 'scaler': scaler_c}, os.path.join(MODELS_DIR, 'model_c_water.pkl'))


    print("\n[4/4] Training Model D: Credit Risk Classifier (XGBoost GPU)")
    df['loan_risk_binary'] = (df['credit_score'] >= 650).astype(int)
    feat_d = ['soil_suitability_score', 'water_availability_score', target_a]
    df_d = df[feat_d + ['loan_risk_binary']].dropna()
    y_d = df_d['loan_risk_binary'].values
    scale_pos_weight = max(1, (y_d == 0).sum() / max((y_d == 1).sum(), 1))

    X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(df_d[feat_d].values, y_d, test_size=0.2, random_state=42, stratify=y_d)
    scaler_d = StandardScaler()
    model_d = XGBClassifier(**XGB_KWARGS, n_estimators=200, max_depth=5, scale_pos_weight=scale_pos_weight, random_state=42)
    metrics_export['Model_D_Credit_Risk'] = eval_classifier(model_d, scaler_d.fit_transform(X_train_d), scaler_d.transform(X_test_d), y_train_d, y_test_d, feat_d, is_binary=True)
    
    metrics_export['Model_D_Credit_Risk']['class_names'] = {0: 'Bad/High Risk', 1: 'Good/Low Risk'}
    joblib.dump({'model': model_d, 'scaler': scaler_d}, os.path.join(MODELS_DIR, 'model_d_credit.pkl'))

    with open(METRICS_JSON, 'w') as f:
        json.dump(metrics_export, f, indent=4)
        
    print(f"\n? All errors corrected. Metrics exported complying to Step 6: {METRICS_JSON}")

if __name__ == "__main__":
    main()
