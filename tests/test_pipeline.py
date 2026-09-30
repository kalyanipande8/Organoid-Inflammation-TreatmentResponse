"""
tests/test_pipeline.py

Runs the full pipeline and sanity-checks the outputs:
  - 20 experiments are present
  - all 3 modalities' columns are present
  - protocol optimization produced a best protocol
  - multi-modal LOOCV accuracy lands in a reasonable band [0.80, 0.97]
  - multi-modal integration beats every single modality alone
"""

import json
import os
import subprocess
import sys

import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture(scope="module", autouse=True)
def run_pipeline():
    subprocess.run([sys.executable, "run_all.py"], cwd=REPO_ROOT, check=True)
    yield


def test_twenty_experiments_present():
    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "experiments.csv"))
    assert len(df) == 20
    assert df["experiment_id"].nunique() == 20


def test_all_three_modalities_present():
    df = pd.read_csv(os.path.join(REPO_ROOT, "data", "experiments.csv"))
    morph_cols = ["morph_size_um", "morph_shape_irregularity", "morph_growth_rate_um_per_day"]
    flow_cols = ["flow_m1_macrophage_pct", "flow_m2_macrophage_pct",
                 "flow_t_cell_pct", "flow_neutrophil_pct"]
    img_cols = ["img_il6_intensity_au", "img_tnfa_intensity_au",
                "img_nfkb_nuclear_translocation"]
    for col in morph_cols + flow_cols + img_cols:
        assert col in df.columns, f"missing expected column {col}"
    assert set(df["responder"].unique()) <= {0, 1}


def test_differentiation_protocol_optimization_ran():
    summary_path = os.path.join(REPO_ROOT, "output", "protocol_quality_summary.csv")
    best_path = os.path.join(REPO_ROOT, "output", "best_protocol.txt")
    assert os.path.exists(summary_path)
    assert os.path.exists(best_path)
    summary = pd.read_csv(summary_path, index_col=0)
    assert len(summary) >= 2  # at least 2 protocol variants compared
    with open(best_path) as f:
        best_protocol = f.readline().strip()
    assert best_protocol in summary.index


def test_multi_modal_accuracy_in_reasonable_band():
    with open(os.path.join(REPO_ROOT, "output", "model_results.json")) as f:
        results = json.load(f)
    acc = results["multi_modal_accuracy"]
    assert 0.80 <= acc <= 0.97, f"multi-modal accuracy {acc} outside reasonable band"


def test_multi_modal_beats_every_single_modality():
    """Injected sanity check: multi-modal integration must outperform every
    single-modality model alone, confirming the multi-modal fusion is
    actually using complementary information across modalities."""
    with open(os.path.join(REPO_ROOT, "output", "model_results.json")) as f:
        results = json.load(f)
    multi = results["multi_modal_accuracy"]
    for name in ["morphology_only", "flow_cytometry_only", "imaging_only"]:
        assert multi >= results["accuracies"][name], (
            f"multi-modal ({multi}) did not beat {name} "
            f"({results['accuracies'][name]})"
        )
    assert multi > results["best_single_modality_accuracy"], (
        "multi-modal model should strictly beat the best single modality"
    )
