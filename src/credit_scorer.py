"""
TerraTrust Credit Scorer
=========================
Deterministc Heuristic Scoring Engine based on Satellite ML.
"""

import os
import sys
import json
import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import *


class VisualCreditScorer:
    """
    Generates a Visual Credit Score using a four-model ML pipeline.
    The final credit decision is made by the XGBoost Credit Risk Classifier.
    """
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load the ML models."""
        # NDVI Predictor
        path_ndvi = os.path.join(MODELS_DIR, "model_a_ndvi.pkl")
        if os.path.exists(path_ndvi):
            self.models['ndvi_predictor'] = joblib.load(path_ndvi)
            print(f"  [+] Loaded NDVI Predictor model")
        else:
            print(f"  [!] NDVI Predictor model not found at {path_ndvi}")

        # Soil Classifier
        path_soil = os.path.join(MODELS_DIR, "model_b_soil.pkl")
        if os.path.exists(path_soil):
            self.models['soil_classifier'] = joblib.load(path_soil)
            print(f"  [+] Loaded Soil Classifier model")
        else:
            print(f"  [!] Soil Classifier model not found at {path_soil}")

        # Water Regressor
        path_water = os.path.join(MODELS_DIR, "model_c_water.pkl")
        if os.path.exists(path_water):
            self.models['water_regressor'] = joblib.load(path_water)
            print(f"  [+] Loaded Water Regressor model")
        # Credit Risk Classifier (final decision engine)
        path_credit = os.path.join(MODELS_DIR, "model_d_credit.pkl")
        if os.path.exists(path_credit):
            self.models['credit_risk_classifier'] = joblib.load(path_credit)
            print(f"  [+] Loaded Credit Risk Classifier model")
        else:
            print(f"  [!] Credit Risk Classifier model not found at {path_credit}")
    
    def predict_ndvi(self, farm_data):
        """Predict NDVI from Ground Features using XGBoost."""
        if 'ndvi_predictor' not in self.models:
            raise ValueError("NDVI Predictor model not loaded.")
            
        artifacts = self.models['ndvi_predictor']
        model = artifacts['model']
        scaler = artifacts['scaler']
        features = artifacts['features']
        
        X = np.array([[farm_data.get(f, 0) for f in features]])
        try:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            return float(np.clip(pred, 0, 1))
        except Exception as e:
            raise ValueError(f"Error predicting NDVI: {e}")
            
    def compute_soil_score(self, farm_data):
        """Use ML Classifier to determine soil suitability score."""
        if 'soil_classifier' not in self.models:
            raise ValueError("Soil Classifier model not loaded.")
            
        artifacts = self.models['soil_classifier']
        model = artifacts['model']
        scaler = artifacts['scaler']
        features = artifacts['features']
        
        X = np.array([[farm_data.get(f, 0) for f in features]])
        try:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            return float(pred)
        except Exception as e:
            raise ValueError(f"Error computing soil score: {e}")

    def compute_water_score(self, farm_data):
        """Use ML Regressor to determine water availability score."""
        if 'water_regressor' not in self.models:
            raise ValueError("Water Regressor model not loaded.")
            
        artifacts = self.models['water_regressor']
        model = artifacts['model']
        scaler = artifacts['scaler']
        features = artifacts['features']
        
        X = np.array([[farm_data.get(f, 0) for f in features]])
        try:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            return float(pred)
        except Exception as e:
            raise ValueError(f"Error computing water score: {e}")
        
    def predict_credit_risk(self, farm_data, pred_crop_health, pred_soil_q, pred_water_depth):
        """
        Use the XGBoost Credit Risk Classifier to predict repayment probability.
        Returns (repay_probability [0-1], predicted_label).
        """
        if 'credit_risk_classifier' not in self.models:
            raise ValueError("Credit Risk Classifier model not loaded.")

        artifacts = self.models['credit_risk_classifier']
        model    = artifacts['model']
        scaler   = artifacts['scaler']
        features = artifacts['features']

        # Map feature names to values
        feature_vals = {
            'pred_crop_health': pred_crop_health,
            'pred_soil_q': pred_soil_q,
            'pred_water_depth': pred_water_depth,
            'aridity_index': farm_data.get('aridity_index', 0),
            'sand_clay_ratio': farm_data.get('sand_clay_ratio', 0),
            'thermal_stress': farm_data.get('thermal_stress', 0),
            'avg_monthly_rainfall_mm': farm_data.get('avg_monthly_rainfall_mm', 0),
            'avg_root_zone_wetness': farm_data.get('avg_root_zone_wetness', 0),
            'soil_fertility_index': farm_data.get('soil_fertility_index', 0),
            'water_table_pressure': farm_data.get('water_table_pressure', 0)
        }
        X = np.array([[feature_vals.get(f, 0) for f in features]])
        try:
            X_scaled = scaler.transform(X)
            label_idx = model.predict(X_scaled)[0]
            label = artifacts['encoder'].inverse_transform([label_idx])[0]
            
            # Phase 3 Fix: Use genuine predict_proba instead of hardcoded numbers
            probas = model.predict_proba(X_scaled)[0]
            classes = list(artifacts['encoder'].classes_)
            
            # Get the genuine probability that this is 'Low' risk (meaning High repayment probability)
            low_idx = classes.index('Low')
            prob_low = float(probas[low_idx])
                
            return prob_low, label
        except Exception as e:
            raise ValueError(f"Error predicting credit risk: {e}")

    def generate_credit_score(self, farm_data):
        """Generate the Visual Credit Score using hierarchical ML + geospatial blending."""

        predicted_crop_health = self.predict_ndvi(farm_data)
        soil_score = self.compute_soil_score(farm_data)
        water_score = self.compute_water_score(farm_data)

        repay_prob, repay_label = self.predict_credit_risk(
            farm_data, predicted_crop_health, soil_score, water_score
        )

        ndvi_raw = farm_data.get('ndvi_annual_mean', farm_data.get('ndvi_point_adj', 0.4))
        try:
            ndvi_raw = float(ndvi_raw) if not pd.isna(ndvi_raw) else 0.4
        except Exception:
            ndvi_raw = 0.4

        rainfall = float(farm_data.get('avg_monthly_rainfall_mm', 80))
        final_score = round(repay_prob * 100, 1)
        final_score = max(5.0, min(98.0, final_score))

        if final_score >= 70:
            risk_category = 'Low'
            recommendation = "APPROVE — Strong satellite-verified agricultural capacity."
        elif final_score >= 48:
            risk_category = 'Moderate'
            recommendation = "REVIEW — Marginal geospatial indicators; field verification advised."
        else:
            risk_category = 'High'
            recommendation = "REJECT — Poor crop health and water availability signals."

        result = {
            'point_id_yr': farm_data.get('point_id_yr', 'UNKNOWN'),
            'year': farm_data.get('year', 2023),
            'taluk': farm_data.get('taluk', 'Unknown'),
            'district': farm_data.get('district', 'Unknown'),
            'heuristic_credit_score': final_score,
            'risk_category': risk_category,
            'recommendation': recommendation,
            'repayment_probability': round(repay_prob, 4),
            'components': {
                'pred_crop_health': predicted_crop_health,
                'pred_water_depth': water_score,
                'pred_soil_q': soil_score,
            },
            'evidence': {
                'satellite_ndvi': round(ndvi_raw, 3),
                'soil_ph': farm_data.get('pH', None),
                'nitrogen_gkg': farm_data.get('nitrogen_g_per_kg', None),
                'rainfall_mm': round(rainfall, 1),
                'root_zone_wetness': farm_data.get('avg_root_zone_wetness', None),
                'coordinates': {
                    'lat': farm_data.get('latitude', None),
                    'lon': farm_data.get('longitude', None)
                }
            }
        }

        return result
    
    def score_all_farms(self, farms_df):
        """Score all coordinates in the dataset and return results."""
        print("\n" + "=" * 60)
        print("  Scoring All Data Points - Heuristic Credit Engine")
        print("=" * 60)
        
        results = []
        for _, row in farms_df.iterrows():
            farm_data = row.to_dict()
            score_result = self.generate_credit_score(farm_data)
            results.append({
                'point_id_yr': score_result['point_id_yr'],
                'year': score_result['year'],
                'taluk': score_result['taluk'],
                'heuristic_credit_score': score_result['heuristic_credit_score'],
                'risk_category': score_result['risk_category'],
                'recommendation': score_result['recommendation'],
                'predicted_ndvi': score_result['components']['pred_crop_health'],
                'latitude': score_result['evidence']['coordinates']['lat'],
                'longitude': score_result['evidence']['coordinates']['lon'],
            })
        
        results_df = pd.DataFrame(results)

        results_path = os.path.join(PROCESSED_DIR, "heuristic_credit_scores.csv")
        results_df.to_csv(results_path, index=False)
        print(f"  Scored {len(results_df)} temporal coordinates")
        print(f"  Saved to: {results_path}")
        
        return results_df


def run_credit_scoring():
    """Run the credit scoring pipeline."""
    df_path = os.path.join(PROCESSED_DIR, "karnataka_master_dataset.csv")
    if not os.path.exists(df_path):
        print("❌ Dataset not found. Run data_pipeline.py first.")
        return
    
    df = pd.read_csv(df_path)
    
    scorer = VisualCreditScorer()
    results = scorer.score_all_farms(df)
    
    # Save a sample detailed report
    if len(df) > 0:
        sample_farm = df.iloc[0].to_dict()
        detailed = scorer.generate_credit_score(sample_farm)
        
        report_path = os.path.join(PROCESSED_DIR, "sample_heuristic_report.json")
        with open(report_path, 'w') as f:
            json.dump(detailed, f, indent=2, default=str)
        print(f"\n  💾 Sample heuristic report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    run_credit_scoring()
