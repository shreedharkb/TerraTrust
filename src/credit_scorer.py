"""
TerraTrust Credit Scorer
=========================
The Visual Credit Score engine that combines all ML model outputs
into a final 0-100 credit risk score with supporting evidence.
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
    Generates a Visual Credit Score for a farmer/farm plot.
    
    The score is a weighted combination of:
    - Crop Health Score (30%) - from NDVI-based ML model
    - Water Availability Score (25%) - from climate + groundwater ML model
    - Soil Suitability Score (25%) - from soil + crop match ML model
    - Historical Trend Score (20%) - from NDVI time-series stability
    """
    
    def __init__(self):
        self.models = {}
        self.load_models()
    
    def load_models(self):
        """Load all trained ML models."""
        model_files = {
            'crop_health': 'crop_health_model.pkl',
            'soil_suitability': 'soil_suitability_model.pkl',
            'water_availability': 'water_availability_model.pkl',
            'credit_scorer': 'credit_scorer_model.pkl',
        }
        
        for key, filename in model_files.items():
            path = os.path.join(MODELS_DIR, filename)
            if os.path.exists(path):
                self.models[key] = joblib.load(path)
                print(f"  ✅ Loaded {key} model")
            else:
                print(f"  ⚠️ {key} model not found at {path}")
    
    def predict_crop_health(self, farm_data):
        """Predict crop health using the trained classifier."""
        if 'crop_health' not in self.models:
            return {'label': 'Unknown', 'score': 50, 'confidence': 0}
        
        artifacts = self.models['crop_health']
        model = artifacts['model']
        scaler = artifacts['scaler']
        le = artifacts['label_encoder']
        features = artifacts['features']
        
        X = np.array([[farm_data.get(f, 0) for f in features]])
        X_scaled = scaler.transform(X)
        
        pred = model.predict(X_scaled)[0]
        label = le.inverse_transform([pred])[0]
        
        # Get probability if available
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(max(proba))
        else:
            confidence = 0.8
        
        # Convert label to score
        score_map = {'Healthy': 85, 'Moderate': 55, 'Stressed': 25}
        score = score_map.get(label, 50)
        
        return {'label': label, 'score': score, 'confidence': round(confidence, 3)}
    
    def predict_soil_suitability(self, farm_data):
        """Predict soil suitability using the trained classifier."""
        if 'soil_suitability' not in self.models:
            return {'label': 'Unknown', 'score': 50, 'confidence': 0}
        
        artifacts = self.models['soil_suitability']
        model = artifacts['model']
        scaler = artifacts['scaler']
        le = artifacts['label_encoder']
        crop_le = artifacts['crop_encoder']
        features = artifacts['features']
        
        # Encode crop
        crop = farm_data.get('declared_crop', 'Maize')
        try:
            crop_encoded = crop_le.transform([crop])[0]
        except ValueError:
            crop_encoded = 0
        
        farm_data_copy = farm_data.copy()
        farm_data_copy['crop_encoded'] = crop_encoded
        
        X = np.array([[farm_data_copy.get(f, 0) for f in features]])
        X_scaled = scaler.transform(X)
        
        pred = model.predict(X_scaled)[0]
        label = le.inverse_transform([pred])[0]
        
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X_scaled)[0]
            confidence = float(max(proba))
        else:
            confidence = 0.8
        
        score_map = {'Suitable': 90, 'Marginal': 55, 'Unsuitable': 20}
        score = score_map.get(label, 50)
        
        return {'label': label, 'score': score, 'confidence': round(confidence, 3)}
    
    def predict_water_availability(self, farm_data):
        """Predict water availability using the trained regressor."""
        if 'water_availability' not in self.models:
            return {'score': 50, 'label': 'Unknown'}
        
        artifacts = self.models['water_availability']
        model = artifacts['model']
        scaler = artifacts['scaler']
        features = artifacts['features']
        
        X = np.array([[farm_data.get(f, 0) for f in features]])
        X_scaled = scaler.transform(X)
        
        score = float(model.predict(X_scaled)[0])
        score = max(0, min(100, score))
        
        if score >= 70:
            label = 'Good'
        elif score >= 45:
            label = 'Moderate'
        else:
            label = 'Poor'
        
        return {'score': round(score, 1), 'label': label}
    
    def compute_historical_trend_score(self, farm_data):
        """Compute historical trend score from NDVI time series."""
        # Load time series if available
        ts_path = os.path.join(SATELLITE_DIR, "ndvi_timeseries.csv")
        if os.path.exists(ts_path):
            ts = pd.read_csv(ts_path)
            ndvi_values = ts['ndvi_mean'].values
            
            if len(ndvi_values) > 2:
                # Trend slope
                x = np.arange(len(ndvi_values))
                slope = np.polyfit(x, ndvi_values, 1)[0]
                
                # Stability (low std = good)
                stability = 1 - min(1, np.std(ndvi_values) / 0.3)
                
                # Score: positive trend + high stability = good
                trend_score = 50 + slope * 500 + stability * 30
                trend_score = max(0, min(100, trend_score))
                
                direction = 'Improving' if slope > 0.001 else ('Declining' if slope < -0.001 else 'Stable')
                
                return {
                    'score': round(trend_score, 1),
                    'trend_direction': direction,
                    'slope': round(slope, 6),
                    'stability': round(stability, 3)
                }
        
        return {'score': 50, 'trend_direction': 'Insufficient Data', 'slope': 0, 'stability': 0.5}
    
    def generate_credit_score(self, farm_data):
        """
        Generate the complete Visual Credit Score with supporting evidence.
        
        Returns a comprehensive dict containing:
        - Final credit score (0-100)
        - Risk category and recommendation
        - Individual component scores
        - Supporting evidence
        """
        # Get individual component scores
        crop_health = self.predict_crop_health(farm_data)
        soil_suit = self.predict_soil_suitability(farm_data)
        water = self.predict_water_availability(farm_data)
        trend = self.compute_historical_trend_score(farm_data)
        
        # Weighted combination fallback removed! Now pulling true ML predictions.
        # Ensure we use the actual ensembled ML regressor trained in models.py
        if 'credit_scorer' in self.models:
            artifacts = self.models['credit_scorer']
            model = artifacts['model']
            scaler = artifacts['scaler']
            features = artifacts['features']
            
            X = np.array([[farm_data.get(f, 0) for f in features]])
            X_scaled = scaler.transform(X)
            final_score = float(model.predict(X_scaled)[0])
        else:
            # Fallback if no model is loaded
            weights = CREDIT_SCORE_WEIGHTS
            final_score = (
                crop_health['score'] * weights.get('crop_health', 0.3) +
                water['score'] * weights.get('water_availability', 0.25) +
                soil_suit['score'] * weights.get('soil_suitability', 0.25) +
                trend['score'] * weights.get('historical_trend', 0.2)
            )
            
        final_score = round(max(0, min(100, final_score)), 1)
        
        # Risk category
        risk_category = "Unknown"
        for label, (low, high) in RISK_CATEGORIES.items():
            if low <= final_score <= high:
                risk_category = label
                break
        
        # Generate recommendation
        if final_score >= 75:
            recommendation = "APPROVE - Strong agricultural indicators support loan viability."
        elif final_score >= 50:
            recommendation = "REVIEW - Moderate risk. Additional field verification recommended."
        else:
            recommendation = "REJECT - High risk indicators. Suggest crop/soil advisory before reapplication."
        
        result = {
            'farm_id': farm_data.get('farm_id', 'UNKNOWN'),
            'taluk': farm_data.get('taluk', 'Unknown'),
            'declared_crop': farm_data.get('declared_crop', 'Unknown'),
            'loan_amount_lakhs': farm_data.get('loan_amount_lakhs', 0),
            
            'credit_score': final_score,
            'risk_category': risk_category,
            'recommendation': recommendation,
            
            'components': {
                'crop_health': crop_health,
                'soil_suitability': soil_suit,
                'water_availability': water,
                'historical_trend': trend,
            },
            
            'ml_model_used': 'credit_scorer' in self.models,
            
            'evidence': {
                'ndvi_value': farm_data.get('ndvi', None),
                'ndwi_value': farm_data.get('ndwi', None),
                'soil_ph': farm_data.get('pH', None),
                'nitrogen': farm_data.get('nitrogen_g_per_kg', None),
                'rainfall': farm_data.get('avg_monthly_rainfall_mm', None),
                'groundwater_depth': farm_data.get('groundwater_depth_m', None),
                'historical_yield': farm_data.get('historical_yield', None),
                'repayment_history_years': farm_data.get('repayment_history_years', None),
                'coordinates': {
                    'lat': farm_data.get('latitude', None),
                    'lon': farm_data.get('longitude', None)
                }
            }
        }
        
        return result
    
    def score_all_farms(self, farms_df):
        """Score all farms in the dataset and return results."""
        print("\n" + "=" * 60)
        print("  Scoring All Farms - Visual Credit Score Engine")
        print("=" * 60)
        
        results = []
        for _, row in farms_df.iterrows():
            farm_data = row.to_dict()
            score_result = self.generate_credit_score(farm_data)
            results.append({
                'farm_id': score_result['farm_id'],
                'taluk': score_result['taluk'],
                'declared_crop': score_result['declared_crop'],
                'credit_score': score_result['credit_score'],
                'risk_category': score_result['risk_category'],
                'crop_health_score': score_result['components']['crop_health']['score'],
                'crop_health_label': score_result['components']['crop_health']['label'],
                'soil_score': score_result['components']['soil_suitability']['score'],
                'soil_label': score_result['components']['soil_suitability']['label'],
                'water_score': score_result['components']['water_availability']['score'],
                'water_label': score_result['components']['water_availability']['label'],
                'trend_score': score_result['components']['historical_trend']['score'],
                'recommendation': score_result['recommendation'],
                'latitude': farm_data.get('latitude'),
                'longitude': farm_data.get('longitude'),
            })
        
        results_df = pd.DataFrame(results)
        
        # Save results
        results_path = os.path.join(PROCESSED_DIR, "credit_scores.csv")
        results_df.to_csv(results_path, index=False)
        print(f"\n  ✅ Scored {len(results_df)} farms")
        print(f"  💾 Saved to: {results_path}")
        
        # Summary
        print(f"\n  Risk Distribution:")
        for cat, count in results_df['risk_category'].value_counts().items():
            pct = count / len(results_df) * 100
            print(f"    {cat}: {count} ({pct:.1f}%)")
        
        print(f"\n  Average Credit Score: {results_df['credit_score'].mean():.1f}")
        print(f"  Score Range: {results_df['credit_score'].min():.1f} - {results_df['credit_score'].max():.1f}")
        
        return results_df


def run_credit_scoring():
    """Run the credit scoring pipeline."""
    # Load authentic dataset
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
        
        report_path = os.path.join(PROCESSED_DIR, "sample_credit_report.json")
        with open(report_path, 'w') as f:
            json.dump(detailed, f, indent=2, default=str)
        print(f"\n  💾 Sample detailed report saved to: {report_path}")
    
    return results


if __name__ == "__main__":
    run_credit_scoring()
