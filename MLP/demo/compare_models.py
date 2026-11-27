#!/usr/bin/env python3
"""
demo/compare_models.py -- Interactive Comparison Tool for State Estimator Models
Compares the pure Data-Driven Baseline MLP against the Physics-Informed MLP (PINN)
on any test snapshot and dropout level.
"""

import argparse
import os
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add parent directory to path so we can import from train_mlp
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from train_mlp import StateEstimatorMLP, DataContractScaler, to_physical, power_residual

torch.set_default_dtype(torch.float64)


def load_model(checkpoint_path: str, device: torch.device):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    model = StateEstimatorMLP(in_features=114, out_features=66, hidden_dims=[256, 256], act="tanh").to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def run_comparison():
    parser = argparse.ArgumentParser(description="Compare Baseline vs. Physics-Informed MLP on IEEE 33-Bus")
    parser.add_argument("--data_dir", type=str, default="data/ieee33", help="Dataset directory")
    parser.add_argument("--condition", type=str, default="Clean", choices=["Clean", "Augmented"], help="Training condition")
    parser.add_argument("--drop", type=int, default=50, choices=[10, 20, 30, 40, 50], help="PMU dropout level (percent)")
    parser.add_argument("--snapshot", type=int, default=0, help="Snapshot index to inspect (0 to 999)")
    parser.add_argument("--baseline", type=str, default=None, help="Path to Baseline model (.pt)")
    parser.add_argument("--pinn", type=str, default=None, help="Path to Physics-Informed model (.pt)")
    parser.add_argument("--plot", action="store_true", help="Generate and save a visual comparison plot")
    args = parser.parse_args()

    device = torch.device("cpu")
    cond_lower = args.condition.lower()
    res_dir = f"results_{cond_lower}"

    # Auto-detect model paths if not provided
    if args.baseline is None:
        args.baseline = os.path.join(res_dir, f"model_{args.condition}_lam0.0_seed42.pt")
        if not os.path.exists(args.baseline):
            # Fallback to general seed42 model if lam0.0 not found
            args.baseline = os.path.join(res_dir, f"best_model_{args.condition}_seed42.pt")

    if args.pinn is None:
        args.pinn = os.path.join(res_dir, f"best_model_{args.condition}_seed42.pt")

    print("\n" + "=" * 95)
    print(f" IEEE 33-BUS STATE ESTIMATOR DEMO: BASELINE vs. PHYSICS-INFORMED")
    print(f" Condition: {args.condition} | Test Dropout: {args.drop}% | Snapshot Index: #{args.snapshot}")
    print(f" Baseline Model: {args.baseline}")
    print(f" Physics Model:  {args.pinn}")
    print("=" * 95)

    # 1. Load Scaler and Models
    scaler = DataContractScaler(os.path.join(args.data_dir, "train.npz"))
    baseline_model = load_model(args.baseline, device)
    pinn_model = load_model(args.pinn, device)

    # 2. Load Test Snapshot
    test_path = os.path.join(args.data_dir, f"test_drop{args.drop}.npz")
    meta_path = os.path.join(args.data_dir, "meta.npz")
    d_test = np.load(test_path)
    meta = np.load(meta_path)

    Y = torch.as_tensor(meta["Ybus_real"] + 1j * meta["Ybus_imag"], dtype=torch.complex128)
    pmu_buses = meta["pmu_idx"].tolist()
    keep_buses = torch.as_tensor([i for i in range(33) if i != int(meta["slack"])], dtype=torch.long)

    idx = args.snapshot
    pmu_vm_full = d_test["pmu_vm_full"][idx]
    pmu_va_full = d_test["pmu_va_full"][idx]
    mask = d_test["mask"][idx]
    p_ps = d_test["p_pseudo"][idx]
    q_ps = d_test["q_pseudo"][idx]
    vm_true = d_test["vm_true"][idx]
    va_true = d_test["va_true"][idx]

    # Preprocess inputs: standardize first, then mask
    norm_vm = ((pmu_vm_full - scaler.vm_mean) / scaler.vm_std) * mask
    norm_va = ((pmu_va_full - scaler.va_mean) / scaler.va_std) * mask
    norm_p = (p_ps - scaler.p_mean) / scaler.p_std
    norm_q = (q_ps - scaler.q_mean) / scaler.q_std

    x = np.concatenate([norm_vm, norm_va, norm_p, norm_q, mask], axis=-1)
    x_tensor = torch.from_numpy(x).unsqueeze(0).to(device)

    # 3. Predict States
    with torch.no_grad():
        out_base = baseline_model(x_tensor)
        out_pinn = pinn_model(x_tensor)

        vm_base, va_base = to_physical(out_base, scaler, device)
        vm_pinn, va_pinn = to_physical(out_pinn, scaler, device)

        p_tensor = torch.from_numpy(p_ps).unsqueeze(0).to(torch.float64)
        q_tensor = torch.from_numpy(q_ps).unsqueeze(0).to(torch.float64)
        res_base = power_residual(vm_base, va_base, Y, p_tensor, q_tensor, keep_buses).item()
        res_pinn = power_residual(vm_pinn, va_pinn, Y, p_tensor, q_tensor, keep_buses).item()

    vm_base = vm_base.squeeze(0).numpy()
    va_base_deg = (va_base.squeeze(0) * (180.0 / np.pi)).numpy()
    vm_pinn = vm_pinn.squeeze(0).numpy()
    va_pinn_deg = (va_pinn.squeeze(0) * (180.0 / np.pi)).numpy()
    va_true_deg = va_true * (180.0 / np.pi)

    # Telemetry Status
    active_pmus = [pmu_buses[i] for i in range(len(pmu_buses)) if mask[i] == 1.0]
    dropped_pmus = [pmu_buses[i] for i in range(len(pmu_buses)) if mask[i] == 0.0]
    print(f"\nTelemetry Overview for Snapshot #{idx}:")
    print(f"  • Installed PMU Count: 16 buses")
    print(f"  • Active PMUs ({len(active_pmus)}):   {active_pmus}")
    print(f"  • DROPPED PMUs ({len(dropped_pmus)}):  {dropped_pmus} (Missing Data)")

    # 4. Detailed Bus Comparison Table
    print(f"\n{'Bus':<5} | {'Sensor Status':<15} | {'True |V|':<10} | {'Base |V| (Err)':<17} | {'PINN |V| (Err)':<17} | {'True θ':<8} | {'Base θ':<8} | {'PINN θ':<8}")
    print("-" * 98)
    for b in range(33):
        if b == 0:
            status = "Slack (Substation)"
        elif b in dropped_pmus:
            status = "PMU [DROPPED]"
        elif b in active_pmus:
            status = "PMU [ACTIVE]"
        else:
            status = "No Sensor (Unmetered)"

        err_b = abs(vm_base[b] - vm_true[b]) * 1000
        err_p = abs(vm_pinn[b] - vm_true[b]) * 1000
        better = "✓" if err_p < err_b else ""

        print(
            f"{b:02d}    | {status:<15} | {vm_true[b]:.4f}     | "
            f"{vm_base[b]:.4f} ({err_b:4.2f}m)  | {vm_pinn[b]:.4f} ({err_p:4.2f}m) {better:<1} | "
            f"{va_true_deg[b]:+6.2f}° | {va_base_deg[b]:+6.2f}° | {va_pinn_deg[b]:+6.2f}°"
        )
    print("-" * 98)
    print("*(Errors shown in milli-pu: 'm' = x 10^-3 pu. '✓' indicates PINN estimated closer to ground truth)*")

    # 5. Snapshot Summary Card
    v_rmse_base = np.sqrt(np.mean((vm_base[1:] - vm_true[1:]) ** 2))
    v_rmse_pinn = np.sqrt(np.mean((vm_pinn[1:] - vm_true[1:]) ** 2))
    th_rmse_base = np.sqrt(np.mean((va_base_deg[1:] - va_true_deg[1:]) ** 2))
    th_rmse_pinn = np.sqrt(np.mean((va_pinn_deg[1:] - va_true_deg[1:]) ** 2))
    v_improvement = ((v_rmse_base - v_rmse_pinn) / v_rmse_base) * 100.0
    th_improvement = ((th_rmse_base - th_rmse_pinn) / th_rmse_base) * 100.0

    print("\n" + "=" * 55)
    print(" SNAPSHOT EVALUATION SUMMARY CARD")
    print("=" * 55)
    print(f" Metric                     | Baseline    | PINN (Physics)")
    print("-" * 55)
    print(f" Voltage RMSE (|V|)         | {v_rmse_base:.6f} pu | {v_rmse_pinn:.6f} pu ({v_improvement:+.1f}%)")
    print(f" Phase Angle RMSE (θ)       | {th_rmse_base:.4f}°     | {th_rmse_pinn:.4f}°     ({th_improvement:+.1f}%)")
    print(f" AC Power Residual (L_phys) | {res_base:.4e}  | {res_pinn:.4e}")
    print("=" * 55)

    # 6. Optional Visual Plot
    if args.plot:
        os.makedirs("demo", exist_ok=True)
        plot_path = f"demo/snapshot_{args.snapshot}_drop{args.drop}_comparison.png"
        buses = np.arange(33)
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), dpi=300)

        # Voltage Profile
        ax1.plot(buses, vm_true, "k-o", label="True Voltage", linewidth=2.0)
        ax1.plot(buses, vm_base, "r--s", label="Baseline MLP", alpha=0.8)
        ax1.plot(buses, vm_pinn, "b-.^", label="Physics-Informed MLP", linewidth=1.8)
        # Highlight dropped PMU buses
        ax1.scatter(dropped_pmus, [vm_true[b] for b in dropped_pmus], s=120, facecolors="none", edgecolors="magenta", linewidths=2.5, label="Dropped PMUs", zorder=5)
        ax1.set_title(f"IEEE 33-Bus Voltage Magnitude Profile (|V|) [Dropout = {args.drop}%]", fontsize=12, fontweight="bold")
        ax1.set_xlabel("Bus Index", fontsize=11)
        ax1.set_ylabel("Voltage Magnitude (pu)", fontsize=11)
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(fontsize=10)

        # Absolute Error Profile
        err_base_plot = np.abs(vm_base - vm_true) * 1000
        err_pinn_plot = np.abs(vm_pinn - vm_true) * 1000
        ax2.bar(buses - 0.15, err_base_plot, width=0.3, color="salmon", label="Baseline Error (milli-pu)")
        ax2.bar(buses + 0.15, err_pinn_plot, width=0.3, color="cornflowerblue", label="PINN Error (milli-pu)")
        ax2.set_title("Absolute Voltage Estimation Error by Bus", fontsize=12, fontweight="bold")
        ax2.set_xlabel("Bus Index", fontsize=11)
        ax2.set_ylabel("Error (× 10⁻³ pu)", fontsize=11)
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(fontsize=10)

        plt.tight_layout()
        plt.savefig(plot_path)
        plt.close()
        print(f"\nVisual comparison plot saved to: {plot_path}")


if __name__ == "__main__":
    run_comparison()
