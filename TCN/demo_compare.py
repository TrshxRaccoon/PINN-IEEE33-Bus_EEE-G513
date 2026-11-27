#!/usr/bin/env python3
"""
demo_compare.py -- Interactive Demonstration and Side-by-Side Model Comparison Tool
===================================================================================
Compares the pure Data-Driven Baseline TCN (λ=0) against the Physics-Informed TCN (λ>0)
on any sequential sliding window from the IEEE 33-bus test dataset under PMU telemetry dropouts.

Key Capabilities:
  1. Sequential Sliding Window Inference: Extracts a historical window of length `seq_len`
     leading up to the target snapshot `seq_idx + seq_len - 1`.
  2. PMU Telemetry Health Report: Audits which of the 16 installed PMUs are actively
     transmitting telemetry vs. experiencing packet dropouts at the target time step.
  3. Comprehensive Bus-by-Bus Terminal Table: Compares ground truth voltage magnitudes and
     phase angles against predictions from both models across all 33 buses.
  4. Overall Summary Metrics Card: Reports non-slack |V| RMSE, θ RMSE, and AC power-flow
     physics residuals (L_phys), highlighting percentage improvements.
  5. 2-Panel Visual Inspection Plot: When `--plot` is enabled, exports a high-resolution
     visualization displaying voltage profiles, dropped PMU locations, and bus error bars.
"""

import argparse
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

# Enforce double precision globally
torch.set_default_dtype(torch.float64)

# Ensure local TCN directory is in sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from tcn_model import StateEstimatorTCN, to_physical, power_residual


# ==============================================================================
# 1. DATA CONTRACT SCALER (Fit strictly on train.npz)
# ==============================================================================
class DataContractScaler:
    """
    Standardizer fit STRICTLY on train.npz only.
    Standardizes:
      - pmu_vm_full, pmu_va_full (16 PMUs)
      - p_pseudo, q_pseudo (33 buses)
      - vm_true, va_true (33 buses)
    Handles zero-variance slack bus by setting std=1.0 safeguard.
    """
    def __init__(self, train_path: str):
        if not os.path.exists(train_path):
            raise FileNotFoundError(f"train.npz not found at: {train_path}")

        d = np.load(train_path)
        vm_full = d["pmu_vm_full"].astype(np.float64)
        va_full = d["pmu_va_full"].astype(np.float64)
        self.vm_mean = vm_full.mean(axis=0)
        self.vm_std = vm_full.std(axis=0)
        self.vm_std[self.vm_std < 1e-8] = 1.0

        self.va_mean = va_full.mean(axis=0)
        self.va_std = va_full.std(axis=0)
        self.va_std[self.va_std < 1e-8] = 1.0

        p_ps = d["p_pseudo"].astype(np.float64)
        q_ps = d["q_pseudo"].astype(np.float64)
        self.p_mean = p_ps.mean(axis=0)
        self.p_std = p_ps.std(axis=0)
        self.p_std[self.p_std < 1e-8] = 1.0

        self.q_mean = q_ps.mean(axis=0)
        self.q_std = q_ps.std(axis=0)
        self.q_std[self.q_std < 1e-8] = 1.0

        vm_true = d["vm_true"].astype(np.float64)
        va_true = d["va_true"].astype(np.float64)
        self.target_vm_mean = vm_true.mean(axis=0)
        self.target_vm_std = vm_true.std(axis=0)
        self.target_vm_std[self.target_vm_std < 1e-8] = 1.0

        self.target_va_mean = va_true.mean(axis=0)
        self.target_va_std = va_true.std(axis=0)
        self.target_va_std[self.target_va_std < 1e-8] = 1.0


# ==============================================================================
# 2. MODEL LOADER WITH ROBUST ARCHITECTURE DETECTION
# ==============================================================================
def load_tcn_model(checkpoint_path: str, device: torch.device) -> StateEstimatorTCN:
    """Load StateEstimatorTCN from checkpoint with auto-detection of config."""
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt

    # Default architectural hyperparameters for TCN
    hidden_channels = [64, 64, 128, 128]
    kernel_size = 3
    act = "tanh"

    # Check if config was saved in checkpoint dict
    if isinstance(ckpt, dict):
        if "hidden_channels" in ckpt:
            hidden_channels = ckpt["hidden_channels"]
        elif "hidden" in ckpt:
            hidden_channels = ckpt["hidden"]
        if "kernel_size" in ckpt:
            kernel_size = ckpt["kernel_size"]
        if "act" in ckpt:
            act = ckpt["act"]

    # Deduce hidden_channels from state_dict if possible
    # Keys like 'tcn.network.0.net.0.weight' have shape (out_ch, in_ch, k)
    detected_channels = []
    layer_idx = 0
    while True:
        key = f"tcn.network.{layer_idx}.net.0.weight"
        if key in state_dict:
            detected_channels.append(state_dict[key].shape[0])
            layer_idx += 1
        else:
            break
    if detected_channels:
        hidden_channels = detected_channels

    model = StateEstimatorTCN(
        in_channels=114,
        out_channels=66,
        hidden_channels=hidden_channels,
        kernel_size=kernel_size,
        dropout=0.0,
        act=act,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    return model


def auto_detect_checkpoint(
    condition: str,
    model_type: str,
    user_path: Optional[str] = None
) -> str:
    """Auto-detect path to Baseline or PINN checkpoint."""
    if user_path is not None and os.path.exists(user_path):
        return user_path

    cond_lower = condition.lower()
    res_dirs = [
        f"results_{cond_lower}",
        "results",
        ".",
        os.path.join("..", f"results_{cond_lower}"),
        os.path.join("..", "results")
    ]

    candidates = []
    if model_type == "baseline":
        for d in res_dirs:
            candidates.extend([
                os.path.join(d, f"model_{condition}_lam0.0_seed42.pt"),
                os.path.join(d, f"model_{condition}_lam0.0.pt"),
                os.path.join(d, f"baseline_{condition}.pt"),
                os.path.join(d, f"best_model_{condition}_seed42.pt"),
            ])
    else:  # pinn
        for d in res_dirs:
            candidates.extend([
                os.path.join(d, f"best_model_{condition}_seed42.pt"),
                os.path.join(d, f"best_model_{condition}.pt"),
                os.path.join(d, f"model_{condition}_lam0.1_seed42.pt"),
                os.path.join(d, f"model_{condition}_lam1.0_seed42.pt"),
                os.path.join(d, f"model_{condition}_lam0.01_seed42.pt"),
            ])

    for c in candidates:
        if os.path.exists(c):
            return c

    # Fallback: if user specified a path, return it even if not currently existing
    return user_path if user_path else candidates[0]


# ==============================================================================
# 3. INTERACTIVE COMPARISON EXECUTION
# ==============================================================================
def run_demo():
    parser = argparse.ArgumentParser(
        description="Interactive Comparison: Baseline TCN vs. Physics-Informed TCN on IEEE 33-Bus"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ieee33",
        help="Dataset directory containing .npz files (default: data/ieee33)"
    )
    parser.add_argument(
        "--condition",
        type=str,
        default="Clean",
        choices=["Clean", "Augmented"],
        help="Training condition of models to evaluate (default: Clean)"
    )
    parser.add_argument(
        "--drop",
        type=int,
        default=50,
        choices=[10, 20, 30, 40, 50],
        help="PMU dropout level in percent (default: 50)"
    )
    parser.add_argument(
        "--seq_idx",
        type=int,
        default=0,
        help="Starting sequence index for the sliding window (default: 0)"
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=12,
        help="Sliding window sequence length T (default: 12)"
    )
    parser.add_argument(
        "--baseline",
        type=str,
        default=None,
        help="Path to Baseline model (.pt). If omitted, auto-detected."
    )
    parser.add_argument(
        "--pinn",
        type=str,
        default=None,
        help="Path to Physics-Informed model (.pt). If omitted, auto-detected."
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate and save a 2-panel visual comparison plot (snapshot_comparison.png)"
    )
    args = parser.parse_args()

    device = torch.device("cpu")

    # Locate data files with fallback
    data_dir = args.data_dir
    if not os.path.exists(data_dir):
        fallback = os.path.join("..", data_dir)
        if os.path.exists(fallback):
            data_dir = fallback

    train_path = os.path.join(data_dir, "train.npz")
    test_path = os.path.join(data_dir, f"test_drop{args.drop}.npz")
    meta_path = os.path.join(data_dir, "meta.npz")

    if not os.path.exists(test_path):
        print(f"[Error] Test file not found: {test_path}")
        print("Please ensure sequential dataset is generated via `make_sequential_dataset.py`.")
        sys.exit(1)

    # Auto-detect checkpoint paths if not provided
    baseline_path = auto_detect_checkpoint(args.condition, "baseline", args.baseline)
    pinn_path = auto_detect_checkpoint(args.condition, "pinn", args.pinn)

    target_snapshot = args.seq_idx + args.seq_len - 1

    print("\n" + "=" * 102)
    print("  IEEE 33-BUS TEMPORAL CONVOLUTIONAL NETWORK (TCN) STATE ESTIMATION DEMO")
    print(f"  Condition: {args.condition} | Test Dropout: {args.drop}% | Window [{args.seq_idx}..{target_snapshot}] (T={args.seq_len})")
    print(f"  Baseline Model (λ=0): {baseline_path}")
    print(f"  PI-TCN Model   (λ>0): {pinn_path}")
    print("=" * 102)

    # 1. Fit scaler strictly on train.npz
    scaler = DataContractScaler(train_path)

    # 2. Load Models
    try:
        baseline_model = load_tcn_model(baseline_path, device)
        print(f" [ModelLoaded] Successfully loaded Baseline model from: {baseline_path}")
    except Exception as e:
        print(f"[Error] Failed to load Baseline model from '{baseline_path}': {e}")
        sys.exit(1)

    try:
        pinn_model = load_tcn_model(pinn_path, device)
        print(f" [ModelLoaded] Successfully loaded Physics-Informed model from: {pinn_path}")
    except Exception as e:
        print(f"[Error] Failed to load Physics-Informed model from '{pinn_path}': {e}")
        sys.exit(1)

    # 3. Load Test Data & Network Metadata
    d_test = np.load(test_path)
    meta = np.load(meta_path)

    n_samples = len(d_test["vm_true"])
    if target_snapshot >= n_samples:
        print(f"[Error] Window end {target_snapshot} exceeds dataset length {n_samples}.")
        sys.exit(1)

    Y = torch.as_tensor(meta["Ybus_real"] + 1j * meta["Ybus_imag"], dtype=torch.complex128, device=device)
    pmu_buses = meta["pmu_idx"].tolist()
    slack_idx = int(meta["slack"])
    keep_buses = torch.as_tensor([i for i in range(33) if i != slack_idx], dtype=torch.long, device=device)

    # 4. Construct Sequential Sliding Window Tensor (T, 114)
    # Block order per step: [PMU |V| * mask (16), PMU theta * mask (16), P_ps (33), Q_ps (33), mask (16)] = 114
    window_x_list = []
    for t in range(args.seq_idx, args.seq_idx + args.seq_len):
        pmu_vm_raw = d_test["pmu_vm_full"][t].astype(np.float64)
        pmu_va_raw = d_test["pmu_va_full"][t].astype(np.float64)
        mask_t = d_test["mask"][t].astype(np.float64)
        p_ps_t = d_test["p_pseudo"][t].astype(np.float64)
        q_ps_t = d_test["q_pseudo"][t].astype(np.float64)

        # Standardize FIRST, then multiply by mask
        norm_vm = ((pmu_vm_raw - scaler.vm_mean) / scaler.vm_std) * mask_t
        norm_va = ((pmu_va_raw - scaler.va_mean) / scaler.va_std) * mask_t
        norm_p = (p_ps_t - scaler.p_mean) / scaler.p_std
        norm_q = (q_ps_t - scaler.q_mean) / scaler.q_std

        x_step = np.concatenate([norm_vm, norm_va, norm_p, norm_q, mask_t], axis=-1)
        window_x_list.append(x_step)

    # Shape: (1, 114, T)
    x_seq = np.stack(window_x_list, axis=0).T  # (114, T)
    x_tensor = torch.as_tensor(x_seq, dtype=torch.float64, device=device).unsqueeze(0)  # (1, 114, T)

    # 5. Run Model Inference
    with torch.no_grad():
        out_base = baseline_model(x_tensor)
        out_pinn = pinn_model(x_tensor)

        vm_base, va_base = to_physical(
            out_base,
            scaler.target_vm_mean,
            scaler.target_vm_std,
            scaler.target_va_mean,
            scaler.target_va_std,
            slack_idx=slack_idx
        )
        vm_pinn, va_pinn = to_physical(
            out_pinn,
            scaler.target_vm_mean,
            scaler.target_vm_std,
            scaler.target_va_mean,
            scaler.target_va_std,
            slack_idx=slack_idx
        )

        # Ground truth and pseudo measurements at the target time step
        p_target = torch.as_tensor(d_test["p_pseudo"][target_snapshot], dtype=torch.float64, device=device).unsqueeze(0)
        q_target = torch.as_tensor(d_test["q_pseudo"][target_snapshot], dtype=torch.float64, device=device).unsqueeze(0)

        res_base = power_residual(vm_base, va_base, Y, p_target, q_target, keep_buses).item()
        res_pinn = power_residual(vm_pinn, va_pinn, Y, p_target, q_target, keep_buses).item()

    # Convert to numpy for terminal tables and plotting
    vm_base_np = vm_base.squeeze(0).cpu().numpy()
    va_base_deg = (va_base.squeeze(0).cpu().numpy()) * (180.0 / np.pi)
    vm_pinn_np = vm_pinn.squeeze(0).cpu().numpy()
    va_pinn_deg = (va_pinn.squeeze(0).cpu().numpy()) * (180.0 / np.pi)

    vm_true = d_test["vm_true"][target_snapshot].astype(np.float64)
    va_true_deg = d_test["va_true"][target_snapshot].astype(np.float64) * (180.0 / np.pi)
    mask_target = d_test["mask"][target_snapshot].astype(np.float64)

    # 6. PMU Telemetry Health Report
    active_pmus = [pmu_buses[i] for i in range(len(pmu_buses)) if mask_target[i] == 1.0]
    dropped_pmus = [pmu_buses[i] for i in range(len(pmu_buses)) if mask_target[i] == 0.0]

    print(f"\nTelemetry Health Audit for Target Snapshot #{target_snapshot}:")
    print(f"  • Total Installed PMUs: 16 candidate buses")
    print(f"  • ACTIVE PMUs ({len(active_pmus):02d}/16) : {active_pmus}")
    print(f"  • DROPPED PMUs ({len(dropped_pmus):02d}/16): {dropped_pmus} (Zeroed telemetry)")

    # 7. Terminal Bus-by-Bus Comparison Table
    print("\n" + "-" * 105)
    print(f"{'Bus':<4} | {'Sensor Status':<16} | {'True |V|':<9} | {'Base |V| (Err)':<17} | {'PI-TCN |V| (Err)':<17} | {'True θ':<8} | {'Base θ':<8} | {'PI-TCN θ':<8}")
    print("-" * 105)

    for b in range(33):
        if b == slack_idx:
            status = "Slack (Substation)"
        elif b in dropped_pmus:
            status = "PMU [DROPPED]"
        elif b in active_pmus:
            status = "PMU [ACTIVE]"
        else:
            status = "Unmetered Bus"

        err_base = abs(vm_base_np[b] - vm_true[b]) * 1000.0  # milli-pu
        err_pinn = abs(vm_pinn_np[b] - vm_true[b]) * 1000.0  # milli-pu
        better_sym = "✓" if err_pinn < err_base else " "

        print(
            f"{b:02d}   | {status:<16} | {vm_true[b]:.4f}    | "
            f"{vm_base_np[b]:.4f} ({err_base:4.2f}m)  | {vm_pinn_np[b]:.4f} ({err_pinn:4.2f}m) {better_sym:<1} | "
            f"{va_true_deg[b]:+6.2f}° | {va_base_deg[b]:+6.2f}° | {va_pinn_deg[b]:+6.2f}°"
        )
    print("-" * 105)
    print("*(Errors shown in milli-pu: 'm' = × 10⁻³ pu. '✓' indicates PI-TCN achieved lower state error)*")

    # 8. Overall Summary Metrics Card (non-slack buses 1..32)
    non_slack = [i for i in range(33) if i != slack_idx]
    v_rmse_base = float(np.sqrt(np.mean((vm_base_np[non_slack] - vm_true[non_slack]) ** 2)))
    v_rmse_pinn = float(np.sqrt(np.mean((vm_pinn_np[non_slack] - vm_true[non_slack]) ** 2)))
    th_rmse_base = float(np.sqrt(np.mean((va_base_deg[non_slack] - va_true_deg[non_slack]) ** 2)))
    th_rmse_pinn = float(np.sqrt(np.mean((va_pinn_deg[non_slack] - va_true_deg[non_slack]) ** 2)))

    v_gain = ((v_rmse_base - v_rmse_pinn) / v_rmse_base) * 100.0 if v_rmse_base > 0 else 0.0
    th_gain = ((th_rmse_base - th_rmse_pinn) / th_rmse_base) * 100.0 if th_rmse_base > 0 else 0.0
    res_gain = ((res_base - res_pinn) / res_base) * 100.0 if res_base > 0 else 0.0

    print("\n" + "=" * 65)
    print(" SNAPSHOT EVALUATION SUMMARY CARD")
    print("=" * 65)
    print(f" Metric                      | Baseline TCN   | PI-TCN (Physics)")
    print("-" * 65)
    print(f" Voltage RMSE (|V|)          | {v_rmse_base:.6f} pu | {v_rmse_pinn:.6f} pu ({v_gain:+.2f}%)")
    print(f" Phase Angle RMSE (θ)        | {th_rmse_base:.4f}°     | {th_rmse_pinn:.4f}°     ({th_gain:+.2f}%)")
    print(f" AC Power Residual (L_phys)  | {res_base:.4e}   | {res_pinn:.4e}   ({res_gain:+.2f}%)")
    print("=" * 65)

    # 9. Optional Visual Plot Export
    if args.plot:
        out_plot_name = "snapshot_comparison.png"
        buses = np.arange(33)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8.5), dpi=300)

        # Panel 1: Voltage Magnitude Profiles
        ax1.plot(buses, vm_true, "k-o", label="Ground Truth |V|", linewidth=2.0, markersize=5)
        ax1.plot(buses, vm_base_np, "r--s", label="Baseline TCN (λ=0)", alpha=0.85, linewidth=1.7, markersize=5)
        ax1.plot(buses, vm_pinn_np, "b-.^", label="PI-TCN (Physics-Regularized)", linewidth=2.0, markersize=5)

        # Highlight dropped PMU locations
        if dropped_pmus:
            dropped_vm = [vm_true[b] for b in dropped_pmus]
            ax1.scatter(
                dropped_pmus,
                dropped_vm,
                s=130,
                facecolors="none",
                edgecolors="#FF007F",
                linewidths=2.4,
                label=f"Dropped PMUs ({len(dropped_pmus)})",
                zorder=6
            )

        ax1.set_title(
            f"IEEE 33-Bus Voltage Magnitude Profile (|V|) [Dropout = {args.drop}%, Window T={args.seq_len}]",
            fontsize=12,
            fontweight="bold"
        )
        ax1.set_xlabel("Bus Index", fontsize=11)
        ax1.set_ylabel("Voltage Magnitude (pu)", fontsize=11)
        ax1.grid(True, linestyle="--", alpha=0.6)
        ax1.legend(loc="lower left", framealpha=0.9)

        # Panel 2: Absolute Estimation Error by Bus
        err_base_bar = np.abs(vm_base_np - vm_true) * 1000.0  # milli-pu
        err_pinn_bar = np.abs(vm_pinn_np - vm_true) * 1000.0  # milli-pu
        bar_width = 0.35

        ax2.bar(buses - bar_width / 2, err_base_bar, width=bar_width, color="#E57373", label="Baseline Error (×10⁻³ pu)")
        ax2.bar(buses + bar_width / 2, err_pinn_bar, width=bar_width, color="#4FC3F7", label="PI-TCN Error (×10⁻³ pu)")

        ax2.set_title("Absolute Voltage Estimation Error by Bus", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Bus Index", fontsize=11)
        ax2.set_ylabel("Error (× 10⁻³ pu)", fontsize=11)
        ax2.set_xticks(buses)
        ax2.set_xticklabels([str(b) for b in buses], fontsize=8)
        ax2.grid(True, linestyle="--", alpha=0.6)
        ax2.legend(loc="upper right", framealpha=0.9)

        plt.tight_layout()
        plt.savefig(out_plot_name, dpi=300)
        plt.close()
        print(f"\n[PlotSaved] Visual comparison chart successfully saved to: {os.path.abspath(out_plot_name)}")


if __name__ == "__main__":
    run_demo()
