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
        path_ndvi = os.path.join(MODELS_DIR, "ndvi_predictor_model.pkl")
        if os.path.exists(path_ndvi):
            self.models['ndvi_predictor'] = joblib.load(path_ndvi)
            print(f"  ✅ Loaded NDVI Predictor model")
        else:
            print(f"  ⚠️ NDVI Predictor model not found at {path_ndvi}")

        # Soil Classifier
        path_soil = os.path.join(MODELS_DIR, "soil_classifier_model.pkl")
        if os.path.exists(path_soil):
            self.models['soil_classifier'] = joblib.load(path_soil)
            print(f"  ✅ Loaded Soil Classifier model")
        else:
            print(f"  ⚠️ Soil Classifier model not found at {path_soil}")

        # Water Regressor
        path_water = os.path.join(MODELS_DIR, "water_regressor_model.pkl")
        if os.path.exists(path_water):
            self.models['water_regressor'] = joblib.load(path_water)
            print(f"  ✅ Loaded Water Regressor model")
        # Credit Risk Classifier (final decision engine)
        path_credit = os.path.join(MODELS_DIR, "credit_risk_classifier_model.pkl")
        if os.path.exists(path_credit):
            self.models['credit_risk_classifier'] = joblib.load(path_credit)
            print(f"  ✅ Loaded Credit Risk Classifier model")
        else:
            print(f"  ⚠️ Credit Risk Classifier model not found at {path_credit}")
    
    def predict_ndvi(self, farm_data):
        """Predict NDVI from Ground Features using XGBoost."""
        if 'ndvi_predictor' not in self.models:
            return farm_data.get('ndvi', 0.45)
            
        artifacts = self.models['ndvi_predictor']
        model = artifacts['model']
        scaler = artifacts['scaler']
        features = artifacts['features']
        
        X = np.array([[farm_data.get(f, 0) for f in features]])
        try:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            return float(np.clip(pred, 0, 1))
        except:
            return farm_data.get('ndvi', 0.45)
            
    def compute_soil_score(self, farm_data):
        """Use ML Classifier to determine soil suitability score."""
        if 'soil_classifier' not in self.models:
            # Fallback if model missing
            return 50.0
            
        artifacts = self.models['soil_classifier']
        model = artifacts['model']
        scaler = artifacts['scaler']
        le = artifacts['label_encoder']
        features = artifacts['features']
        
        X = np.array([[farm_data.get(f, 0) for f in features]])
        try:
            X_scaled = scaler.transform(X)
            pred_idx = model.predict(X_scaled)[0]
            label = le.inverse_transform([pred_idx])[0]
            
            # Map labels to numeric scores
            score_map = {'Suitable': 95.0, 'Marginal': 60.0, 'Unsuitable': 30.0}
            return float(score_map.get(label, 50.0))
        except:
            return 50.0

    def compute_water_score(self, farm_data):
        """Use ML Regressor to determine water availability score."""
        if 'water_regressor' not in self.models:
            return 50.0
            
        artifacts = self.models['water_regressor']
        model = artifacts['model']
        scaler = artifacts['scaler']
        features = artifacts['features']
        
        X = np.array([[farm_data.get(f, 0) for f in features]])
        try:
            X_scaled = scaler.transform(X)
            pred = model.predict(X_scaled)[0]
            return float(np.clip(pred, 0, 100))
        except:
            return 50.0
        
    def predict_credit_risk(self, ndvi_score, soil_score, water_score, farm_size):
        """
        Use the XGBoost Credit Risk Classifier to predict repayment probability.
        Returns (repay_probability [0-1], predicted_label [0/1]).
        """
        if 'credit_risk_classifier' not in self.models:
            # Fallback: weighted average if model not trained yet
            raw = (ndvi_score * 0.40) + (water_score * 0.30) + (soil_score * 0.30)
            prob = round(max(0, min(100, raw)), 1) / 100.0
            return prob, int(prob >= 0.5)

        artifacts = self.models['credit_risk_classifier']
        model    = artifacts['model']
        scaler   = artifacts['scaler']
        features = artifacts['features']

        # Map feature names to values
        feature_vals = {
            'ndvi':                    ndvi_score / 100.0,   # model trained on raw ndvi (0-1)
            'soil_suitability_score':  soil_score,
            'water_availability_score': water_score,
            'farm_size':               farm_size,
        }
        X = np.array([[feature_vals.get(f, 0) for f in features]])
        try:
            X_scaled = scaler.transform(X)
            prob  = float(model.predict_proba(X_scaled)[0][1])   # P(good repayment)
            label = int(model.predict(X_scaled)[0])
            return prob, label
        except Exception as e:
            raw = (ndvi_score * 0.40 + water_score * 0.30 + soil_score * 0.30) / 100.0
            return raw, int(raw >= 0.5)

    def generate_credit_score(self, farm_data):
        """
        Generate the complete Visual Credit Score with supporting evidence.
        Final score = repayment probability (from Credit Risk Classifier) × 100.
        Component models provide explainability.
        """
        
        # Get intermediate ML predictions
        predicted_ndvi = self.predict_ndvi(farm_data)
        soil_score     = self.compute_soil_score(farm_data)
        water_score    = self.compute_water_score(farm_data)
        farm_size      = float(farm_data.get('farm_size', 3.0) or 3.0)

        ndvi_score = predicted_ndvi * 100

        # Final score from Credit Risk Classifier (repayment probability × 100)
        repay_prob, repay_label = self.predict_credit_risk(
            ndvi_score, soil_score, water_score, farm_size
        )
        final_score = round(repay_prob * 100, 1)
        
        # Risk category
        risk_category = "Unknown"
        for label, (low, high) in RISK_CATEGORIES.items():
            if low <= final_score <= high:
                risk_category = label
                break
        
        # Generate recommendation
        if final_score >= 70:
            recommendation = "APPROVE - Strong physical agricultural capacity."
        elif final_score >= 45:
            recommendation = "REVIEW - Marginal physical indicators."
        else:
            recommendation = "REJECT - Poor physical crop capacity."
        
        result = {
            'point_id_yr': farm_data.get('point_id_yr', 'UNKNOWN'),
            'year':        farm_data.get('year', 2023),
            'taluk':       farm_data.get('taluk', 'Unknown'),
            'district':    farm_data.get('district', 'Unknown'),

            'heuristic_credit_score': final_score,
            'risk_category':          risk_category,
            'recommendation':         recommendation,
            'repayment_probability':  round(repay_prob, 4),

            'components': {
                'predicted_ndvi_score': round(ndvi_score, 1),
                'water_score':          round(water_score, 1),
                'soil_score':           round(soil_score, 1),
                'farm_size':            farm_size,
            },
            
            'evidence': {
                'ml_predicted_ndvi': round(predicted_ndvi, 3),
                'actual_satellite_ndvi': round(farm_data.get('ndvi', 0), 3) if not pd.isna(farm_data.get('ndvi')) else None,
                'soil_ph': farm_data.get('pH', None),
                'nitrogen': farm_data.get('nitrogen_g_per_kg', None),
                'rainfall_mm': farm_data.get('avg_monthly_rainfall_mm', None),
                'groundwater_depth': farm_data.get('groundwater_depth_m', None),
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
                'predicted_ndvi': score_result['evidence']['ml_predicted_ndvi'],
                'latitude': score_result['evidence']['coordinates']['lat'],
                'longitude': score_result['evidence']['coordinates']['lon'],
            })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(PROCESSED_DIR, "heuristic_credit_scores.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n  ✅ Scored {len(results_df)} temporal coordinates")
        print(f"  💾 Saved to: {results_path}")
        
        return results_df


def run_credit_scoring():
    """Run the credit scoring pipeline."""
    df_path = os.path.join(PROCESSED_DIR, "davangere_master_dataset.csv")
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
