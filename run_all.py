"""
run_all.py

Runs the full pipeline end-to-end:
  1. Simulate 20 skin-derived organoid/co-culture inflammation experiments,
     across 4 differentiation protocol variants -> data/experiments.csv
  2. Differentiation protocol optimization: compute quality composite per
     protocol, identify best performer -> output/protocol_quality_summary.csv
  3. Train + LOOCV-evaluate single-modality and multi-modal models,
     predicting treatment response -> output/model_results.json

Run with:  python run_all.py
"""

import simulate_experiments
import protocol_optimization
import train_model


def main():
    print("--- Simulating 20 organoid/co-culture inflammation experiments ---")
    simulate_experiments.main()

    print("\n--- Differentiation protocol optimization ---")
    protocol_optimization.main()

    print("\n--- Training multi-modal treatment-response model (LOOCV) ---")
    train_model.main()


if __name__ == "__main__":
    main()
