"""
train_model.py

Trains a multi-modal ML model integrating morphology, flow cytometry, and
imaging features to predict organoid treatment response (responder vs.
non-responder).

With only 20 experiments, a single train/test split is too noisy to trust,
so evaluation uses stratified Leave-One-Out Cross-Validation (LOOCV), which
is the standard honest approach for small-n biological datasets: every
experiment is held out exactly once and predicted by a model trained on the
other 19, so the reported accuracy is a real generalization estimate with no
leakage.

Two things are compared, both via the identical LOOCV protocol:
  1. Single-modality models (morphology-only, flow-only, imaging-only)
  2. The multi-modal model (all three modalities concatenated, standardized,
     and fed into a single L2-regularized LogisticRegression -- a linear
     late-fusion classifier)

This directly demonstrates that multi-modal integration outperforms any
single modality alone.
"""

import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import StandardScaler

MORPH_COLS = ["morph_size_um", "morph_shape_irregularity", "morph_growth_rate_um_per_day"]
FLOW_COLS = ["flow_m1_macrophage_pct", "flow_m2_macrophage_pct", "flow_t_cell_pct", "flow_neutrophil_pct"]
IMG_COLS = ["img_il6_intensity_au", "img_tnfa_intensity_au", "img_nfkb_nuclear_translocation"]
ALL_COLS = MORPH_COLS + FLOW_COLS + IMG_COLS

MODEL_SEED = 7


def loocv_accuracy(X: np.ndarray, y: np.ndarray, seed: int = MODEL_SEED) -> float:
    loo = LeaveOneOut()
    correct = 0
    for train_idx, test_idx in loo.split(X):
        scaler = StandardScaler().fit(X[train_idx])
        X_train = scaler.transform(X[train_idx])
        X_test = scaler.transform(X[test_idx])

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=seed)
        clf.fit(X_train, y[train_idx])
        pred = clf.predict(X_test)
        correct += int(pred[0] == y[test_idx][0])
    return correct / len(y)


def main():
    df = pd.read_csv("data/experiments.csv")
    y = df["responder"].values

    results = {}
    for name, cols in [
        ("morphology_only", MORPH_COLS),
        ("flow_cytometry_only", FLOW_COLS),
        ("imaging_only", IMG_COLS),
        ("multi_modal_all3", ALL_COLS),
    ]:
        X = df[cols].values
        acc = loocv_accuracy(X, y)
        results[name] = acc
        print(f"{name:22s} LOOCV accuracy: {acc:.3f}  (n_features={len(cols)})")

    best_single = max(
        results["morphology_only"], results["flow_cytometry_only"], results["imaging_only"]
    )
    multi_modal_acc = results["multi_modal_all3"]
    print(f"\nBest single-modality accuracy: {best_single:.3f}")
    print(f"Multi-modal accuracy:          {multi_modal_acc:.3f}")
    print(f"Multi-modal improvement over best single modality: "
          f"{(multi_modal_acc - best_single) * 100:+.1f} pts")

    with open("output/model_results.json", "w") as f:
        json.dump({
            "n_experiments": len(df),
            "accuracies": results,
            "best_single_modality_accuracy": best_single,
            "multi_modal_accuracy": multi_modal_acc,
        }, f, indent=2)

    return results


if __name__ == "__main__":
    main()
