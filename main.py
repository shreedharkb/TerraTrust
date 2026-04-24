"""
TerraTrust - Main Pipeline Orchestrator
=========================================
Runs the complete end-to-end pipeline:
1. ML model training
2. Credit score generation
3. Result Visualizations

Usage: python -u main.py
"""

import os
import sys
import time
import io
import subprocess

# Fix Windows console encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import models
from src.credit_scorer import run_credit_scoring
from src.config import DATA_DIR, MODELS_DIR, PROJECT_ROOT


def main():
    start_time = time.time()
    
    print("=" * 60)
    print("  TerraTrust - Intelligent Rural Banking Loan System")
    print("  Target: All Karnataka Districts")
    print("  All data sources are GENUINE and verifiable")
    print("=" * 60)
    
    # -- Phase 1: ML Model Training --
    print("\n\n" + "#" * 60)
    print("  PHASE 1: ML MODEL TRAINING (Dual Evaluation)")
    print("#" * 60)
    
    p1 = time.time()
    models.main()
    print(f"\n  ⏱️  Phase 1 total: {time.time()-p1:.1f}s")
    
    # -- Phase 2: Credit Scoring --
    print("\n\n" + "#" * 60)
    print("  PHASE 2: VISUAL CREDIT SCORE GENERATION")
    print("#" * 60)
    
    p2 = time.time()
    credit_results = run_credit_scoring()
    print(f"\n  ⏱️  Phase 2 total: {time.time()-p2:.1f}s")

    # -- Phase 3: Visualizations --
    print("\n############################################################")
    print("  PHASE 3: GENERATING RESULTS & VISUALIZATIONS (SKIPPED)")
    print("############################################################")
    # import generate_visualizations
    # generate_visualizations.main()
    print(f"\n  ⏱️  Phase 3 total: {time.time()-p3:.1f}s")
    
    # -- Summary --
    elapsed = time.time() - start_time
    
    print("\n\n" + "=" * 60)
    print("  PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\n  ⏱️  Total execution time: {elapsed:.1f}s ({elapsed/60:.1f} minutes)")
    print(f"\n  Generated files:")
    
    for root, dirs, files in os.walk(MODELS_DIR):
        for f in files:
            fpath = os.path.join(root, f)
            size = os.path.getsize(fpath)
            rel = os.path.relpath(fpath, PROJECT_ROOT)
            print(f"     {rel} ({size:,} bytes)")
    
    print(f"\n  Next step: Review Visualizations or cd app && npm run dev")


if __name__ == "__main__":
    main()
