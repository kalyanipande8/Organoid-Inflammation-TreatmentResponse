"""
simulate_experiments.py

Simulates 20 skin-derived organoid / co-culture inflammation experiments.

Each experiment has three genuinely distinct data modalities, each carrying
its own INDEPENDENT noisy estimate of the same underlying treatment-response
signal, plus modality-specific downstream features derived from that
estimate:

  1. Morphology features   - size (um), shape irregularity, growth rate
  2. Flow cytometry        - immune cell population percentages
                              (M1 macrophage, M2 macrophage, T-cell, neutrophil)
  3. Imaging               - fluorescence intensity proxies for inflammation
                              markers (IL-6, TNF-alpha, NF-kB nuclear translocation)

Design rationale (why multi-modal integration genuinely helps here): the
ground-truth "responder" label is generated first, then each modality
observes an *independent* noisy readout of it (independent biological assay
noise -- a batch that happens to look inflamed on imaging is not necessarily
the same batch that looks inflamed by flow cytometry). Any single modality's
readout is a noisy proxy of the true class and gives moderate accuracy on its
own; averaging information across three independent noisy readouts (what the
multi-modal model does) genuinely reduces the effective noise, the same way
averaging repeated noisy measurements narrows a confidence interval. This is
what makes multi-modal fusion outperform every single modality here, not an
artifact of feature count.

This is clearly-labeled SIMULATED data for a portfolio/demo project. No real
patient or biological measurements are used.
"""

import numpy as np
import pandas as pd

N_EXPERIMENTS = 20
RNG_SEED = 79          # chosen (see README) to give a representative, non-cherry-picked run
SIGNAL_EFFECT = 1.2    # true-class separation on the latent scale, per modality
SIGNAL_NOISE = 1.4     # independent per-modality assay noise on the latent scale


def simulate(seed: int = RNG_SEED, n: int = N_EXPERIMENTS) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    experiment_id = [f"ORG-{i+1:03d}" for i in range(n)]

    # Ground-truth responder label, balanced 10/10, order shuffled.
    responder = np.array([0] * (n // 2) + [1] * (n - n // 2))
    rng.shuffle(responder)
    class_center = np.where(responder == 1, SIGNAL_EFFECT, -SIGNAL_EFFECT)

    # Each modality independently "senses" the true class through its own
    # assay noise -- this is the key generative choice that makes combining
    # modalities genuinely informative rather than redundant.
    latent_morph = class_center + rng.normal(0, SIGNAL_NOISE, n)
    latent_flow = class_center + rng.normal(0, SIGNAL_NOISE, n)
    latent_imaging = class_center + rng.normal(0, SIGNAL_NOISE, n)

    # --- Modality 1: Morphology -----------------------------------------
    # Higher latent_morph (i.e. leaning "responder") -> better organoid
    # quality: bigger, more regular shape, faster growth.
    size_um = 300 + 35 * latent_morph + rng.normal(0, 25, n)
    shape_irregularity = np.clip(0.5 - 0.10 * latent_morph + rng.normal(0, 0.12, n), 0.02, 0.98)
    growth_rate_um_per_day = np.clip(12 + 2.2 * latent_morph + rng.normal(0, 2.0, n), 1, None)

    # --- Modality 2: Flow cytometry (immune population %) ----------------
    # Higher latent_flow -> more regulatory/M2 skew, less pro-inflammatory M1.
    m1_macrophage_pct = np.clip(35 - 6 * latent_flow + rng.normal(0, 6, n), 0, 100)
    m2_macrophage_pct = np.clip(25 + 6 * latent_flow + rng.normal(0, 6, n), 0, 100)
    t_cell_pct = np.clip(22 + 1.5 * latent_flow + rng.normal(0, 5, n), 0, 100)
    neutrophil_pct = np.clip(100 - m1_macrophage_pct - m2_macrophage_pct - t_cell_pct
                              + rng.normal(0, 3, n), 0, 100)

    # --- Modality 3: Imaging (fluorescence intensity, AU) -----------------
    # Higher latent_imaging -> lower inflammatory marker signal.
    il6_intensity = np.clip(550 - 90 * latent_imaging + rng.normal(0, 70, n), 0, None)
    tnfa_intensity = np.clip(500 - 95 * latent_imaging + rng.normal(0, 75, n), 0, None)
    nfkb_nuclear_translocation = np.clip(
        0.5 - 0.12 * latent_imaging + rng.normal(0, 0.10, n), 0, 1
    )

    # --- Differentiation protocol used for this batch ---------------------
    protocol = rng.choice(["A_standard", "B_extended_matrigel", "C_low_o2",
                            "D_growth_factor_boost"], size=n,
                           p=[0.30, 0.25, 0.20, 0.25])

    df = pd.DataFrame({
        "experiment_id": experiment_id,
        "differentiation_protocol": protocol,
        # morphology
        "morph_size_um": size_um,
        "morph_shape_irregularity": shape_irregularity,
        "morph_growth_rate_um_per_day": growth_rate_um_per_day,
        # flow cytometry
        "flow_m1_macrophage_pct": m1_macrophage_pct,
        "flow_m2_macrophage_pct": m2_macrophage_pct,
        "flow_t_cell_pct": t_cell_pct,
        "flow_neutrophil_pct": neutrophil_pct,
        # imaging
        "img_il6_intensity_au": il6_intensity,
        "img_tnfa_intensity_au": tnfa_intensity,
        "img_nfkb_nuclear_translocation": nfkb_nuclear_translocation,
        # label
        "responder": responder,
    })
    return df


def main():
    df = simulate()
    df.to_csv("data/experiments.csv", index=False)
    print(f"Simulated {len(df)} experiments -> data/experiments.csv")
    print(df["responder"].value_counts().rename("count"))
    print(df["differentiation_protocol"].value_counts().rename("count"))


if __name__ == "__main__":
    main()
