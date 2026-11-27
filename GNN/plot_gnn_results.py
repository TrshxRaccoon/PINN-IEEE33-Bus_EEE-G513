#!/usr/bin/env python3
"""
plot_gnn_results.py -- Publication-Ready Figures for GNN State Estimation

Generates 300 DPI publication figures:
  1. rmse_vs_coverage_clean.png: |V| and theta RMSE vs PMU Coverage (Clean Training)
  2. rmse_vs_coverage_augmented.png: |V| and theta RMSE vs PMU Coverage (Augmented Training)
  3. lambda_sensitivity_sweep_clean.png: Error metrics and L_phys vs Lambda (Clean)
  4. lambda_sensitivity_sweep_augmented.png: Error metrics and L_phys vs Lambda (Augmented)
  5. gnn_vs_mlp_comparison.png: Direct side-by-side comparison between GNN and MLP

Usage:
  python plot_gnn_results.py
  python plot_gnn_results.py --clean_gnn results_clean/benchmark_clean.csv --aug_gnn results_augmented/benchmark_augmented.csv
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

here = os.path.dirname(os.path.abspath(__file__))


def load_benchmark(csv_path: str) -> pd.DataFrame:
    """Loads CSV if exists, returns None otherwise."""
    if not os.path.exists(csv_path):
        return None
    try:
        df = pd.read_csv(csv_path)
        return prepare_dataframe(df)
    except Exception as e:
        print(f"Warning: could not read {csv_path}: {e}")
        return None


def prepare_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes column names across heterogeneous GNN and MLP benchmark CSV formats.
    Handles capitalization differences (Lambda vs lambda, RMSE_V_pu vs vm_rmse, etc.).
    """
    if df is None or df.empty:
        return None
    df = df.copy()

    # Normalize column names: stripped and lowercase
    col_map = {c: c.strip().lower() for c in df.columns}
    df = df.rename(columns=col_map)

    # 1. Standardize lambda
    if "lambda" not in df.columns:
        for alt in ["lam", "penalty"]:
            if alt in df.columns:
                df["lambda"] = df[alt]
                break

    # 2. Standardize vm_rmse
    if "vm_rmse" not in df.columns:
        for alt in ["rmse_v_pu", "rmse_v", "vm_err"]:
            if alt in df.columns:
                df["vm_rmse"] = df[alt]
                break

    # 3. Standardize va_rmse_deg
    if "va_rmse_deg" not in df.columns:
        for alt in ["rmse_theta_deg", "rmse_va_deg", "rmse_theta", "va_err"]:
            if alt in df.columns:
                df["va_rmse_deg"] = df[alt]
                break

    # 4. Standardize l_phys
    if "l_phys" not in df.columns:
        for alt in ["phys_loss", "phys_res", "residual"]:
            if alt in df.columns:
                df["l_phys"] = df[alt]
                break

    # 5. Standardize coverage_num (e.g. 50, 60, 70, 80, 90)
    if "coverage_num" not in df.columns:
        if "pmu_coverage" in df.columns:
            if df["pmu_coverage"].dtype == object:
                df["coverage_num"] = (
                    df["pmu_coverage"].astype(str).str.rstrip("%").astype(float).astype(int)
                )
            else:
                df["coverage_num"] = df["pmu_coverage"].astype(int)
        elif "dropout_level" in df.columns or "pmu_dropout" in df.columns:
            drop_col = "dropout_level" if "dropout_level" in df.columns else "pmu_dropout"
            if df[drop_col].dtype == object:
                drop_val = (
                    df[drop_col].astype(str).str.rstrip("%").astype(float).astype(int)
                )
            else:
                drop_val = df[drop_col].astype(int)
            df["coverage_num"] = 100 - drop_val
            df["pmu_coverage"] = df["coverage_num"].astype(str) + "%"

    # 6. Ensure numeric types
    if "lambda" in df.columns:
        df["lambda"] = pd.to_numeric(df["lambda"], errors="coerce")

    for col in ["vm_rmse", "va_rmse_deg", "l_phys", "coverage_num"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def plot_rmse_vs_coverage(
    df_gnn: pd.DataFrame,
    df_mlp: pd.DataFrame,
    condition: str,
    out_dir: str,
):
    """
    Plots |V| and theta RMSE vs PMU Coverage (50% to 90%) for GNN and MLP.
    """
    if df_gnn is None or df_gnn.empty:
        print(f"Skipping {condition} coverage plot: GNN benchmark not available.")
        return

    # Separate GNN Baseline (lambda=0) and Best Physics model
    baseline_gnn = df_gnn[df_gnn["lambda"] == 0.0].sort_values("coverage_num")

    # Find the best physics lambda for GNN (minimum score at 50% coverage)
    phys_gnn_50 = df_gnn[(df_gnn["lambda"] > 0.0) & (df_gnn["coverage_num"] == 50)]
    if not phys_gnn_50.empty:
        best_lam = phys_gnn_50.loc[
            (phys_gnn_50["vm_rmse"] / 0.01 + phys_gnn_50["va_rmse_deg"] / 0.1).idxmin()
        ]["lambda"]
        best_phys_gnn = df_gnn[df_gnn["lambda"] == best_lam].sort_values("coverage_num")
        phys_gnn_label = f"Physics GCN (λ={best_lam})"
    else:
        best_phys_gnn = None
        phys_gnn_label = "Physics GCN"

    # Separate MLP data if available
    baseline_mlp = None
    best_phys_mlp = None
    phys_mlp_label = "Physics MLP"
    if df_mlp is not None and not df_mlp.empty and "lambda" in df_mlp.columns:
        baseline_mlp = df_mlp[df_mlp["lambda"] == 0.0].sort_values("coverage_num")
        phys_mlp_50 = df_mlp[(df_mlp["lambda"] > 0.0) & (df_mlp["coverage_num"] == 50)]
        if not phys_mlp_50.empty:
            mlp_best_lam = phys_mlp_50.loc[
                (phys_mlp_50["vm_rmse"] / 0.01 + phys_mlp_50["va_rmse_deg"] / 0.1).idxmin()
            ]["lambda"]
            best_phys_mlp = df_mlp[df_mlp["lambda"] == mlp_best_lam].sort_values("coverage_num")
            phys_mlp_label = f"Physics MLP (λ={mlp_best_lam})"

    # Create 2-panel figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel 1: Voltage Magnitude RMSE
    if not baseline_gnn.empty:
        ax1.plot(
            baseline_gnn["coverage_num"],
            baseline_gnn["vm_rmse"],
            "b--o",
            label="Baseline GCN (λ=0.0)",
            linewidth=2,
            markersize=6,
        )
    if best_phys_gnn is not None and not best_phys_gnn.empty:
        ax1.plot(
            best_phys_gnn["coverage_num"],
            best_phys_gnn["vm_rmse"],
            "b-s",
            label=phys_gnn_label,
            linewidth=2.5,
            markersize=7,
        )

    if baseline_mlp is not None and not baseline_mlp.empty and "vm_rmse" in baseline_mlp.columns:
        ax1.plot(
            baseline_mlp["coverage_num"],
            baseline_mlp["vm_rmse"],
            "r--^",
            label="Baseline MLP (λ=0.0)",
            linewidth=1.6,
            markersize=5,
            alpha=0.75,
        )
    if best_phys_mlp is not None and not best_phys_mlp.empty and "vm_rmse" in best_phys_mlp.columns:
        ax1.plot(
            best_phys_mlp["coverage_num"],
            best_phys_mlp["vm_rmse"],
            "r-d",
            label=phys_mlp_label,
            linewidth=1.8,
            markersize=5,
            alpha=0.75,
        )

    ax1.set_xlabel("PMU Coverage Level (%)", fontsize=12)
    ax1.set_ylabel("Voltage Magnitude RMSE (pu)", fontsize=12)
    ax1.set_title(
        f"Voltage Magnitude Error vs. PMU Coverage ({condition})",
        fontsize=13,
        fontweight="bold",
    )
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="upper right", fontsize=10)

    # Panel 2: Voltage Phase Angle RMSE
    if not baseline_gnn.empty:
        ax2.plot(
            baseline_gnn["coverage_num"],
            baseline_gnn["va_rmse_deg"],
            "b--o",
            label="Baseline GCN (λ=0.0)",
            linewidth=2,
            markersize=6,
        )
    if best_phys_gnn is not None and not best_phys_gnn.empty:
        ax2.plot(
            best_phys_gnn["coverage_num"],
            best_phys_gnn["va_rmse_deg"],
            "b-s",
            label=phys_gnn_label,
            linewidth=2.5,
            markersize=7,
        )

    if baseline_mlp is not None and not baseline_mlp.empty and "va_rmse_deg" in baseline_mlp.columns:
        ax2.plot(
            baseline_mlp["coverage_num"],
            baseline_mlp["va_rmse_deg"],
            "r--^",
            label="Baseline MLP (λ=0.0)",
            linewidth=1.6,
            markersize=5,
            alpha=0.75,
        )
    if best_phys_mlp is not None and not best_phys_mlp.empty and "va_rmse_deg" in best_phys_mlp.columns:
        ax2.plot(
            best_phys_mlp["coverage_num"],
            best_phys_mlp["va_rmse_deg"],
            "r-d",
            label=phys_mlp_label,
            linewidth=1.8,
            markersize=5,
            alpha=0.75,
        )

    ax2.set_xlabel("PMU Coverage Level (%)", fontsize=12)
    ax2.set_ylabel("Voltage Phase Angle RMSE (deg)", fontsize=12)
    ax2.set_title(
        f"Voltage Angle Error vs. PMU Coverage ({condition})",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="upper right", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"rmse_vs_coverage_{condition.lower()}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved publication plot: {out_path}")


def plot_lambda_sweep(df_gnn: pd.DataFrame, condition: str, out_dir: str):
    """
    Plots estimation errors and AC power-flow residual across the full lambda spectrum (0 to 1000).
    """
    if df_gnn is None or df_gnn.empty:
        return

    # Focus on the most challenging 50% coverage / 50% dropout test
    df_50 = df_gnn[df_gnn["coverage_num"] == 50].copy()
    if df_50.empty:
        # Fallback to pmu_dropout string matching
        if "pmu_dropout" in df_gnn.columns:
            df_50 = df_gnn[df_gnn["pmu_dropout"].astype(str).str.contains("50")].copy()
    if df_50.empty:
        return

    # Aggregate by lambda in case of multiple seeds
    df_50 = (
        df_50.groupby("lambda")[["vm_rmse", "va_rmse_deg", "l_phys"]]
        .mean()
        .reset_index()
        .sort_values("lambda")
    )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Panel 1: RMSE vs Lambda
    ax1.plot(
        df_50["lambda"],
        df_50["vm_rmse"],
        "b-s",
        label="|V| RMSE (pu)",
        linewidth=2,
        markersize=6,
    )
    ax1.set_xscale("symlog", linthresh=1e-4)
    ax1.set_xlabel("Physics Loss Penalty λ", fontsize=12)
    ax1.set_ylabel("Voltage Magnitude RMSE (pu)", color="b", fontsize=12)
    ax1.tick_params(axis="y", labelcolor="b")
    ax1.grid(True, linestyle="--", alpha=0.6)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(
        df_50["lambda"],
        df_50["va_rmse_deg"],
        "g-o",
        label="θ RMSE (deg)",
        linewidth=2,
        markersize=6,
    )
    ax1_twin.set_ylabel("Voltage Phase Angle RMSE (deg)", color="g", fontsize=12)
    ax1_twin.tick_params(axis="y", labelcolor="g")
    ax1.set_title(
        f"Estimation Errors vs. Physics Penalty λ (50% Dropout, {condition})",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 2: Physics Residual L_phys vs Lambda
    ax2.plot(
        df_50["lambda"],
        df_50["l_phys"],
        "m-^",
        label="L_phys Residual",
        linewidth=2.2,
        markersize=7,
    )
    ax2.set_xscale("symlog", linthresh=1e-4)
    ax2.set_yscale("log")
    ax2.set_xlabel("Physics Loss Penalty λ", fontsize=12)
    ax2.set_ylabel("Power Flow Residual L_phys (pu)", fontsize=12)
    ax2.set_title(
        f"AC Power-Flow Residual vs. λ (50% Dropout, {condition})",
        fontsize=13,
        fontweight="bold",
    )
    ax2.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"lambda_sensitivity_sweep_{condition.lower()}.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved lambda sensitivity plot: {out_path}")


def plot_side_by_side_bar_summary(
    df_clean_gnn: pd.DataFrame,
    df_clean_mlp: pd.DataFrame,
    df_aug_gnn: pd.DataFrame,
    df_aug_mlp: pd.DataFrame,
    out_dir: str,
):
    """
    Generates a high-level bar chart comparing GNN vs. MLP under severe 50% dropout.
    """
    labels = []
    v_errors = []
    va_errors = []

    def extract_metrics(df, name):
        if df is None or df.empty:
            return None
        sub = df[df["coverage_num"] == 50]
        if sub.empty:
            return None
        # Baseline (lam = 0)
        base = sub[sub["lambda"] == 0.0]
        # Best Physics (lam > 0)
        phys = sub[sub["lambda"] > 0.0]
        best_p = (
            phys.loc[(phys["vm_rmse"] / 0.01 + phys["va_rmse_deg"] / 0.1).idxmin()]
            if not phys.empty
            else None
        )

        res = {}
        if not base.empty:
            res["Baseline"] = (base.iloc[0]["vm_rmse"], base.iloc[0]["va_rmse_deg"])
        if best_p is not None:
            res[f"Physics (λ={best_p['lambda']})"] = (
                best_p["vm_rmse"],
                best_p["va_rmse_deg"],
            )
        return res

    clean_gnn_m = extract_metrics(df_clean_gnn, "Clean GNN")
    clean_mlp_m = extract_metrics(df_clean_mlp, "Clean MLP")
    aug_gnn_m = extract_metrics(df_aug_gnn, "Aug GNN")
    aug_mlp_m = extract_metrics(df_aug_mlp, "Aug MLP")

    entries = []
    if clean_mlp_m and "Baseline" in clean_mlp_m:
        entries.append(("Clean MLP Base", clean_mlp_m["Baseline"]))
    if clean_mlp_m:
        for k, v in clean_mlp_m.items():
            if k != "Baseline":
                entries.append((f"Clean MLP Phys", v))
    if clean_gnn_m and "Baseline" in clean_gnn_m:
        entries.append(("Clean GCN Base", clean_gnn_m["Baseline"]))
    if clean_gnn_m:
        for k, v in clean_gnn_m.items():
            if k != "Baseline":
                entries.append((f"Clean GCN Phys", v))

    if aug_mlp_m and "Baseline" in aug_mlp_m:
        entries.append(("Aug MLP Base", aug_mlp_m["Baseline"]))
    if aug_mlp_m:
        for k, v in aug_mlp_m.items():
            if k != "Baseline":
                entries.append((f"Aug MLP Phys", v))
    if aug_gnn_m and "Baseline" in aug_gnn_m:
        entries.append(("Aug GCN Base", aug_gnn_m["Baseline"]))
    if aug_gnn_m:
        for k, v in aug_gnn_m.items():
            if k != "Baseline":
                entries.append((f"Aug GCN Phys", v))

    if not entries:
        return

    names = [e[0] for e in entries]
    v_vals = [e[1][0] * 1000.0 for e in entries]  # in 1e-3 pu
    va_vals = [e[1][1] for e in entries]  # in deg

    x = np.arange(len(names))
    width = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.bar(x, v_vals, width, color=["#e74c3c" if "MLP" in n else "#3498db" for n in names])
    ax1.set_ylabel("Voltage Magnitude RMSE (×10⁻³ pu)", fontsize=11)
    ax1.set_title("50% PMU Dropout: Voltage Magnitude Error", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")

    ax2.bar(x, va_vals, width, color=["#e74c3c" if "MLP" in n else "#3498db" for n in names])
    ax2.set_ylabel("Voltage Angle RMSE (deg)", fontsize=11)
    ax2.set_title("50% PMU Dropout: Voltage Angle Error", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=35, ha="right", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.5, axis="y")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "gnn_vs_mlp_summary.png")
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"✓ Saved side-by-side summary bar chart: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate publication plots for GNN state estimation"
    )
    parser.add_argument(
        "--clean_gnn",
        type=str,
        default=os.path.join(here, "results_clean", "benchmark_clean.csv"),
    )
    parser.add_argument(
        "--aug_gnn",
        type=str,
        default=os.path.join(here, "results_augmented", "benchmark_augmented.csv"),
    )
    parser.add_argument(
        "--clean_mlp",
        type=str,
        default=os.path.normpath(
            os.path.join(here, "..", "MLP", "results_clean", "benchmark_clean.csv")
        ),
    )
    parser.add_argument(
        "--aug_mlp",
        type=str,
        default=os.path.normpath(
            os.path.join(here, "..", "MLP", "results_augmented", "benchmark_augmented.csv")
        ),
    )
    parser.add_argument("--out", type=str, default=os.path.join(here, "plots"))
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    print("\n================================================================================")
    print(" GENERATING PUBLICATION PLOTS FOR GNN STATE ESTIMATION")
    print(f" Target Output Directory: {args.out}")
    print("================================================================================")

    # Load and standardize all available datasets
    df_clean_gnn = load_benchmark(args.clean_gnn)
    df_clean_mlp = load_benchmark(args.clean_mlp)
    df_aug_gnn = load_benchmark(args.aug_gnn)
    df_aug_mlp = load_benchmark(args.aug_mlp)

    # 1. Clean condition coverage plot & lambda sweep
    if df_clean_gnn is not None:
        plot_rmse_vs_coverage(df_clean_gnn, df_clean_mlp, "Clean", args.out)
        plot_lambda_sweep(df_clean_gnn, "Clean", args.out)

    # 2. Augmented condition coverage plot & lambda sweep
    if df_aug_gnn is not None:
        plot_rmse_vs_coverage(df_aug_gnn, df_aug_mlp, "Augmented", args.out)
        plot_lambda_sweep(df_aug_gnn, "Augmented", args.out)

    # 3. Overall GNN vs MLP comparison bar chart
    plot_side_by_side_bar_summary(
        df_clean_gnn, df_clean_mlp, df_aug_gnn, df_aug_mlp, args.out
    )

    print("\n✓ Plotting script completed successfully.\n")


if __name__ == "__main__":
    main()
