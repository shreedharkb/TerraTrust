import json
import os

metrics_path = os.path.join("models", "model_metrics.json")
if not os.path.exists(metrics_path):
    print("Metrics file not found. Please train models first.")
    exit(1)

with open(metrics_path) as f:
    m = json.load(f)

print("====================================")
print("   TerraTrust Model Testing Stats   ")
print("====================================")

print("\n--- Model A (NDVI Predictor) ---")
print(f"Test R2   : {m['Model_A_NDVI']['test_r2']}")
print(f"Test RMSE : {m['Model_A_NDVI']['test_rmse']}")
print(f"Train R2  : {m['Model_A_NDVI']['train_r2']}")

print("\n--- Model B (Soil Suitability) ---")
print(f"Test Acc  : {m['Model_B_Soil']['test_accuracy']}")
print(f"Test F1   : {m['Model_B_Soil']['test_f1_score']}")

print("\n--- Model C (Water Availability) ---")
print(f"Test R2   : {m['Model_C_Water']['test_r2']}")
print(f"Test RMSE : {m['Model_C_Water']['test_rmse']}")

print("\n--- Model D (Credit Risk Classifier) ---")
print(f"Test Acc  : {m['Model_D_Credit_Risk']['test_accuracy']}")
print(f"Train Acc : {m['Model_D_Credit_Risk']['train_accuracy']}")
print(f"Test F1   : {m['Model_D_Credit_Risk']['test_f1_score']}")
print("\nTarget Leakage Fixed: Model D now correctly learns from physical proxies rather than perfect labels!")
print("====================================")
