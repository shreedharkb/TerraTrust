"""
TerraTrust - Main Pipeline Orchestrator
=========================================
Runs the complete end-to-end pipeline:
1. Data collection from KGIS + APIs (All Karnataka)
2. Satellite imagery analysis
3. ML model training (GPU-accelerated)
4. Credit score generation

Usage: python -u main.py
"""

import os
import sys
import time
import io

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.data_pipeline import run_full_data_pipeline
from src.satellite_pipeline import run_satellite_pipeline
from src.models import run_model_training
from src.credit_scorer import run_credit_scoring
from src.config import *


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("  TerraTrust - Intelligent Rural Banking Loan System")
    print("  Target: All Karnataka Districts")
    print("  All data sources are GENUINE and verifiable")
    print("=" * 60)
    
    # -- Phase 1: Data Pipeline --
    print("\n\n" + "#" * 60)
    print("  PHASE 1: DATA COLLECTION & PREPROCESSING")
    print("#" * 60)
    
    p1 = time.time()
    master, district_gdf, taluks_gdf = run_full_data_pipeline()
    print(f"\n  ⏱️  Phase 1 total: {time.time()-p1:.1f}s")
    
    # -- Phase 2: Satellite Pipeline --
    print("\n\n" + "#" * 60)
    print("  PHASE 2: SATELLITE IMAGERY ANALYSIS")
    print("#" * 60)
    
    p2 = time.time()
    bbox = list(district_gdf.geometry.total_bounds)
    ndvi_df, sat_summary = run_satellite_pipeline(bbox)
    print(f"\n  ⏱️  Phase 2 total: {time.time()-p2:.1f}s")
    
    # -- Phase 3: ML Model Training --
    print("\n\n" + "#" * 60)
    print("  PHASE 3: ML MODEL TRAINING (RTX 3050 GPU)")
    print("#" * 60)
    
    p3 = time.time()
    expanded_df, metrics = run_model_training()
    print(f"\n  ⏱️  Phase 3 total: {time.time()-p3:.1f}s")
    
    # -- Phase 4: Credit Scoring --
    print("\n\n" + "#" * 60)
    print("  PHASE 4: VISUAL CREDIT SCORE GENERATION")
    print("#" * 60)
    
    p4 = time.time()
    credit_results = run_credit_scoring()
    print(f"\n  ⏱️  Phase 4 total: {time.time()-p4:.1f}s")
    
    # -- Summary --
    elapsed = time.time() - start_time
    
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  ⏱️  Total execution time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"\n  Generated files:")
    
    for root, dirs, files in os.walk(DATA_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, PROJECT_ROOT)
            print(f"     {rel} ({size:,} bytes)")
    
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, PROJECT_ROOT)
            print(f"     {rel} ({size:,} bytes)")
    
    print(f"\n  Next step: cd app && npm run dev")


if __name__ == "__main__":
    main()
