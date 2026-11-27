#!/usr/bin/env python3
"""
plot_results.py -- Generate publication-quality RMSE vs. Coverage plots
Reads benchmark_clean.csv and benchmark_augmented.csv, then produces:
  - Voltage Magnitude RMSE (|V|) vs PMU Coverage (%)
  - Voltage Angle RMSE (theta) vs PMU Coverage (%)
  - Physics Residual (L_phys) vs PMU Coverage (%)
"""

import os
import pandas as pd
import matplotlib.pyplot as plt


def plot_benchmarks(clean_csv="results_clean/benchmark_clean.csv", aug_csv="results_augmented/benchmark_augmented.csv", out_dir="plots"):
    os.makedirs(out_dir, exist_ok=True)
    dfs = []
    if os.path.exists(clean_csv):
        df_clean = pd.read_csv(clean_csv)
        dfs.append(df_clean)
    if os.path.exists(aug_csv):
        df_aug = pd.read_csv(aug_csv)
        dfs.append(df_aug)

    if not dfs:
        print("No benchmark CSVs found to plot!")
        return

    df = pd.concat(dfs, ignore_index=True)
    # Parse dropout level to numeric coverage: e.g. "10%" -> 90% coverage
    df["Dropout_Pct"] = df["Dropout_Level"].str.replace("%", "").astype(float)
    df["PMU_Coverage_Pct"] = 100.0 - df["Dropout_Pct"]

    conditions = df["Condition"].unique()

    for cond in conditions:
        sub = df[df["Condition"] == cond]
        fig, axes = plt.subplots(1, 2, figsize=(13, 5), dpi=300)

        # 1. Voltage Magnitude RMSE (|V|)
        ax = axes[0]
        for lam, group in sub.groupby("Lambda"):
            group = group.sort_values("PMU_Coverage_Pct")
            label = f"Baseline (λ=0)" if lam == 0.0 else f"PI-MLP (λ={lam})"
            style = "o--" if lam == 0.0 else "s-"
            lw = 2.2 if lam in [0.0, 0.1, 0.01] else 1.2
            ax.plot(group["PMU_Coverage_Pct"], group["RMSE_V_pu"] * 1000, style, label=label, linewidth=lw)

        ax.set_title(f"Voltage Magnitude Error ({cond} Training)", fontsize=12, fontweight="bold")
        ax.set_xlabel("PMU Measurement Coverage (%)", fontsize=11)
        ax.set_ylabel("RMSE |V| (× 10⁻³ pu)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9)

        # 2. Voltage Angle RMSE (theta)
        ax = axes[1]
        for lam, group in sub.groupby("Lambda"):
            group = group.sort_values("PMU_Coverage_Pct")
            label = f"Baseline (λ=0)" if lam == 0.0 else f"PI-MLP (λ={lam})"
            style = "o--" if lam == 0.0 else "s-"
            lw = 2.2 if lam in [0.0, 0.1, 0.01] else 1.2
            ax.plot(group["PMU_Coverage_Pct"], group["RMSE_Theta_deg"], style, label=label, linewidth=lw)

        ax.set_title(f"Voltage Phase Angle Error ({cond} Training)", fontsize=12, fontweight="bold")
        ax.set_xlabel("PMU Measurement Coverage (%)", fontsize=11)
        ax.set_ylabel("RMSE θ (degrees)", fontsize=11)
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend(fontsize=9)

        plt.tight_layout()
        plot_path = os.path.join(out_dir, f"rmse_vs_coverage_{cond.lower()}.png")
        plt.savefig(plot_path)
        plt.close()
        print(f"Saved plot: {plot_path}")

    print(f"\nAll plots successfully saved to: {out_dir}/")


if __name__ == "__main__":
    plot_benchmarks()
