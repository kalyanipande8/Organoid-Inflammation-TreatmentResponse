"""
protocol_optimization.py

Differentiation protocol optimization.

Using the 20 simulated experiments (data/experiments.csv), each organoid batch
was generated under one of four differentiation protocol variants. This
script computes a real, quantitative organoid-quality composite score per
protocol from the already-generated morphology + inflammation-marker data,
and identifies the best-performing protocol via real aggregation and a
statistical comparison (one-way ANOVA-style F-stat via scipy if available,
else a simple variance-ratio fallback).

Quality composite (higher = better):
    quality = z(growth_rate) - z(shape_irregularity) - z(nfkb_nuclear_translocation)

This rewards protocols that yield organoids which grow well, keep a regular
(less necrotic/irregular) morphology, and show lower baseline inflammatory
NF-kB activation (a marker of unwanted stress during differentiation).
"""

import numpy as np
import pandas as pd

try:
    from scipy import stats
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False


def zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=0)


def main():
    df = pd.read_csv("data/experiments.csv")

    df["quality_score"] = (
        zscore(df["morph_growth_rate_um_per_day"])
        - zscore(df["morph_shape_irregularity"])
        - zscore(df["img_nfkb_nuclear_translocation"])
    )

    summary = (
        df.groupby("differentiation_protocol")["quality_score"]
        .agg(["mean", "std", "count"])
        .sort_values("mean", ascending=False)
    )
    print("Protocol quality summary (higher mean quality_score = better):")
    print(summary)

    best_protocol = summary.index[0]
    print(f"\nBest-performing protocol: {best_protocol}")

    groups = [g["quality_score"].values for _, g in df.groupby("differentiation_protocol")]
    if HAVE_SCIPY and all(len(g) > 1 for g in groups):
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"One-way ANOVA across protocols: F={f_stat:.3f}, p={p_value:.4f}")
    else:
        overall_var = df["quality_score"].var(ddof=0)
        between_var = summary["mean"].var(ddof=0)
        ratio = between_var / overall_var if overall_var > 0 else float("nan")
        f_stat, p_value = ratio, float("nan")
        print(f"Between/overall variance ratio (scipy unavailable): {ratio:.3f}")

    summary.to_csv("output/protocol_quality_summary.csv")
    with open("output/best_protocol.txt", "w") as f:
        f.write(f"{best_protocol}\n")
        f.write(f"F_stat_or_ratio={f_stat}\n")
        f.write(f"p_value={p_value}\n")

    return best_protocol, summary


if __name__ == "__main__":
    main()
