#!/usr/bin/env python3
"""
compare_models.py -- Interactive Single-Snapshot Comparison Tool for GNN State Estimation

Allows inspecting and comparing model predictions on any individual snapshot from test sets:
  - PMU sensor telemetry status (active vs dropped)
  - Ground truth vs GNN (and optional MLP) voltage magnitudes |V| and angles theta across all 33 buses
  - Bus-by-bus prediction error table
  - Summary error metrics: |V| RMSE, theta RMSE, max bus error, physics residual L_phys
  - Optional visual plot export (--plot)

Usage:
  python compare_models.py --condition Augmented --drop 50 --snapshot 0
  python compare_models.py --condition Clean --drop 50 --snapshot 0 --plot
  python compare_models.py --gnn_ckpt results_augmented/best_model_Augmented_seed42.pt --drop 50 --snapshot 0
"""

import argparse
import os
import sys

import numpy as np
import torch

here = os.path.dirname(os.path.abspath(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

from graph_data import DataContractScaler, IEEE33GraphDataset, resolve_meta_path
from models import EdgeAttributedGCN
from physics import compute_voltage_errors, load_ybus, power_residual, to_physical

# Enforce double precision
torch.set_default_dtype(torch.float64)


def run_comparison(
    condition: str = "Augmented",
    drop: int = 50,
    snapshot_idx: int = 0,
    data_dir: str = "../MLP/data/ieee33",
    gnn_ckpt: str = None,
    mlp_ckpt: str = None,
    save_plot: bool = False,
    device_str: str = "cpu",
):
    device = torch.device(device_str)
    meta_path = resolve_meta_path(os.path.join(data_dir, "meta.npz"))
    train_path = os.path.join(data_dir, "train.npz")
    test_path = os.path.join(data_dir, f"test_drop{drop}.npz")

    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test file not found: {test_path}")

    # 1. Resolve Checkpoint paths if not explicitly given
    if gnn_ckpt is None:
        candidates = [
            os.path.join(here, f"results_{condition.lower()}", f"best_model_{condition}_seed42.pt"),
            os.path.join(here, f"results_{condition.lower()}", f"model_{condition}_lam0.1_seed42.pt"),
            os.path.join(here, f"results_{condition.lower()}", f"model_{condition}_lam1.0_seed42.pt"),
            os.path.join(here, f"results_{condition.lower()}", f"model_{condition}_lam0.0_seed42.pt"),
        ]
        for c in candidates:
            if os.path.exists(c):
                gnn_ckpt = c
                break

    if mlp_ckpt is None:
        mlp_candidates = [
            os.path.normpath(os.path.join(here, "..", "MLP", f"results_{condition.lower()}", f"best_model_{condition}_seed42.pt")),
            os.path.normpath(os.path.join(here, "..", "MLP", f"results_{condition.lower()}", f"model_{condition}_lam0.1_seed42.pt")),
        ]
        for c in mlp_candidates:
            if os.path.exists(c):
                mlp_ckpt = c
                break

    print("\n" + "=" * 90)
    print(f" IEEE 33-BUS STATE ESTIMATION DEMO: SNAPSHOT #{snapshot_idx} ({drop}% PMU DROPOUT)")
    print(f" Condition: {condition} | GNN Model: {gnn_ckpt or 'NOT FOUND'}")
    print(f" MLP Baseline Model: {mlp_ckpt or 'None (skipped)'}")
    print("=" * 90)

    # 2. Load Scaler and Dataset
    scaler = DataContractScaler(train_path=train_path, meta_path=meta_path)
    dataset = IEEE33GraphDataset(test_path, scaler, meta_path=meta_path, augment=False)
    data_raw = np.load(test_path)

    sample = dataset[snapshot_idx]
    meta = np.load(meta_path)
    pmu_idx = meta["pmu_idx"].tolist()
    slack = int(meta["slack"].item()) if hasattr(meta["slack"], "item") else int(meta["slack"])
    n_bus = int(meta["n_bus"].item()) if hasattr(meta["n_bus"], "item") else int(meta["n_bus"])
    keep_buses = torch.tensor([i for i in range(n_bus) if i != slack], dtype=torch.long, device=device)
    Y = load_ybus(meta_path, device=device)

    # Telemetry inspection
    mask_sample = sample.mask.numpy()
    active_pmus = [b for b, m in zip(pmu_idx, mask_sample) if m == 1.0]
    dropped_pmus = [b for b, m in zip(pmu_idx, mask_sample) if m == 0.0]

    print("\n--- PMU Telemetry Status ---")
    print(f"Installed PMUs:     {len(pmu_idx)} candidate buses ({pmu_idx})")
    print(f"Active Channels:    {len(active_pmus)} buses: {active_pmus}")
    print(f"Dropped Channels:   {len(dropped_pmus)} buses: {dropped_pmus}")
    print(f"PMU Observability:  {len(active_pmus) / len(pmu_idx):.1%} of installed PMUs reporting")

    # 3. GNN Inference
    vm_true = sample.vm_true.unsqueeze(0).to(device)
    va_true = sample.va_true.unsqueeze(0).to(device)
    p_ps = sample.p_pseudo.unsqueeze(0).to(device)
    q_ps = sample.q_pseudo.unsqueeze(0).to(device)

    gnn_vm_pred, gnn_va_pred = None, None
    if gnn_ckpt and os.path.exists(gnn_ckpt):
        ckpt = torch.load(gnn_ckpt, map_location=device)
        model = EdgeAttributedGCN(
            in_channels=6,
            edge_channels=4,
            out_channels=2,
            hidden_dims=ckpt.get("hidden_dims", [128, 128, 128]),
            act=ckpt.get("act", "tanh"),
            num_nodes=33,
        ).to(device)
        model.load_state_dict(ckpt.get("model_state_dict", ckpt))
        model.eval()

        with torch.no_grad():
            sample_dev = sample.clone().to(device)
            out_gnn = model(sample_dev, reshape_batch=True)
            gnn_vm_pred, gnn_va_pred = to_physical(out_gnn, scaler, device)

        gnn_v_err, gnn_va_err = compute_voltage_errors(
            gnn_vm_pred, gnn_va_pred, vm_true, va_true, keep_buses
        )
        gnn_v_rmse = float(torch.sqrt((gnn_v_err ** 2).mean()).item())
        gnn_va_rmse_deg = float(torch.sqrt((gnn_va_err ** 2).mean()).item())
        gnn_phys = float(power_residual(gnn_vm_pred, gnn_va_pred, Y, p_ps, q_ps, keep_buses).item())

    # 4. Optional MLP Inference (if available)
    mlp_vm_pred, mlp_va_pred = None, None
    mlp_v_rmse, mlp_va_rmse_deg, mlp_phys = None, None, None
    if mlp_ckpt and os.path.exists(mlp_ckpt):
        try:
            mlp_data = torch.load(mlp_ckpt, map_location=device)
            # Reconstruct MLP input (114 features)
            norm_vm = ((data_raw["pmu_vm_full"][snapshot_idx] - scaler.vm_mean) / scaler.vm_std) * mask_sample
            norm_va = ((data_raw["pmu_va_full"][snapshot_idx] - scaler.va_mean) / scaler.va_std) * mask_sample
            norm_p = (data_raw["p_pseudo"][snapshot_idx] - scaler.p_mean) / scaler.p_std
            norm_q = (data_raw["q_pseudo"][snapshot_idx] - scaler.q_mean) / scaler.q_std
            x_mlp = np.concatenate([norm_vm, norm_va, norm_p, norm_q, mask_sample], axis=-1)
            x_mlp_t = torch.from_numpy(x_mlp).unsqueeze(0).to(device)

            # Build simple sequential MLP matching checkpoint
            state = mlp_data.get("model_state_dict", mlp_data)
            layers = []
            keys = [k for k in state.keys() if "weight" in k]
            dims = [state[k].shape[1] for k in keys] + [state[keys[-1]].shape[0]]
            for i in range(len(dims) - 2):
                layers.append(torch.nn.Linear(dims[i], dims[i+1], dtype=torch.float64))
                layers.append(torch.nn.Tanh())
            layers.append(torch.nn.Linear(dims[-2], dims[-1], dtype=torch.float64))
            mlp_net = torch.nn.Sequential(*layers).to(device)
            mlp_net.load_state_dict(state)
            mlp_net.eval()

            with torch.no_grad():
                out_mlp = mlp_net(x_mlp_t)
                # De-standardize 66 features to physical
                t_vm_m = torch.tensor(scaler.target_vm_mean, device=device)
                t_vm_s = torch.tensor(scaler.target_vm_std, device=device)
                t_va_m = torch.tensor(scaler.target_va_mean, device=device)
                t_va_s = torch.tensor(scaler.target_va_std, device=device)
                mlp_vm_pred = out_mlp[:, :33] * t_vm_s + t_vm_m
                mlp_va_pred = out_mlp[:, 33:] * t_va_s + t_va_m
                mlp_vm_pred[:, 0] = 1.0
                mlp_va_pred[:, 0] = 0.0

            mlp_v_e, mlp_va_e = compute_voltage_errors(
                mlp_vm_pred, mlp_va_pred, vm_true, va_true, keep_buses
            )
            mlp_v_rmse = float(torch.sqrt((mlp_v_e ** 2).mean()).item())
            mlp_va_rmse_deg = float(torch.sqrt((mlp_va_e ** 2).mean()).item())
            mlp_phys = float(power_residual(mlp_vm_pred, mlp_va_pred, Y, p_ps, q_ps, keep_buses).item())
        except Exception as e:
            print(f"(Note: Could not run MLP comparison: {e})")

    # 5. Bus-by-Bus Prediction Table
    print("\n--- Bus-by-Bus Prediction & Estimation Error Table ---")
    header = (
        f"{'Bus':<4} | {'PMU':<6} | {'True |V|':<9} | {'GNN |V|':<9} | {'|V| Err':<9} | "
        f"{'True th(°)':<10} | {'GNN th(°)':<10} | {'th Err(°)':<10}"
    )
    if mlp_vm_pred is not None:
        header += f" | {'MLP |V|':<9} | {'MLP th(°)'}"
    print(header)
    print("-" * len(header))

    vt_np = vm_true[0].cpu().numpy()
    vat_np = (va_true[0].cpu().numpy() * 180.0 / np.pi)

    gnn_v_np = gnn_vm_pred[0].cpu().numpy() if gnn_vm_pred is not None else np.zeros(33)
    gnn_va_np = (gnn_va_pred[0].cpu().numpy() * 180.0 / np.pi) if gnn_va_pred is not None else np.zeros(33)

    mlp_v_np = mlp_vm_pred[0].cpu().numpy() if mlp_vm_pred is not None else None
    mlp_va_np = (mlp_va_pred[0].cpu().numpy() * 180.0 / np.pi) if mlp_va_pred is not None else None

    for i in range(33):
        pmu_tag = "ACTIVE" if i in active_pmus else ("DROP" if i in dropped_pmus else "--")
        v_err = abs(gnn_v_np[i] - vt_np[i])
        # Circular angle difference in degrees
        diff_rad = np.arctan2(np.sin(np.deg2rad(gnn_va_np[i] - vat_np[i])), np.cos(np.deg2rad(gnn_va_np[i] - vat_np[i])))
        va_err = abs(np.rad2deg(diff_rad))

        row = (
            f"{i:<4d} | {pmu_tag:<6} | {vt_np[i]:<9.5f} | {gnn_v_np[i]:<9.5f} | {v_err:<9.5f} | "
            f"{vat_np[i]:<10.4f} | {gnn_va_np[i]:<10.4f} | {va_err:<10.4f}"
        )
        if mlp_v_np is not None:
            row += f" | {mlp_v_np[i]:<9.5f} | {mlp_va_np[i]:<10.4f}"
        print(row)

    print("-" * len(header))

    # 6. Summary Comparison Metrics
    print("\n--- Summary Performance on Snapshot ---")
    if gnn_vm_pred is not None:
        print(f"GNN Model: |V| RMSE = {gnn_v_rmse:.6f} pu | theta RMSE = {gnn_va_rmse_deg:.4f}° | L_phys = {gnn_phys:.3e} pu")
    if mlp_v_rmse is not None:
        print(f"MLP Model: |V| RMSE = {mlp_v_rmse:.6f} pu | theta RMSE = {mlp_va_rmse_deg:.4f}° | L_phys = {mlp_phys:.3e} pu")

    # 7. Optional Plot
    if save_plot:
        try:
            import matplotlib.pyplot as plt

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
            buses = np.arange(33)

            # Subplot 1: Voltage Magnitude
            ax1.plot(buses, vt_np, "k-o", label="Ground Truth", linewidth=2.0, markersize=5)
            if gnn_vm_pred is not None:
                ax1.plot(buses, gnn_v_np, "b--s", label="Edge-Attributed GCN", linewidth=1.8, markersize=5)
            if mlp_v_np is not None:
                ax1.plot(buses, mlp_v_np, "r:^", label="MLP Baseline", linewidth=1.5, markersize=5)

            # Highlight PMU buses
            for b in active_pmus:
                ax1.axvline(x=b, color="green", alpha=0.15, linestyle="--")
            for b in dropped_pmus:
                ax1.axvline(x=b, color="orange", alpha=0.25, linestyle=":")

            ax1.set_ylabel("Voltage Magnitude |V| (pu)", fontsize=12)
            ax1.set_title(
                f"IEEE 33-Bus State Estimation: Snapshot #{snapshot_idx} ({drop}% PMU Dropout, {len(active_pmus)} Active PMUs)",
                fontsize=13,
                fontweight="bold",
            )
            ax1.grid(True, linestyle="--", alpha=0.6)
            ax1.legend(loc="best", fontsize=11)

            # Subplot 2: Voltage Angle
            ax2.plot(buses, vat_np, "k-o", label="Ground Truth", linewidth=2.0, markersize=5)
            if gnn_va_pred is not None:
                ax2.plot(buses, gnn_va_np, "b--s", label="Edge-Attributed GCN", linewidth=1.8, markersize=5)
            if mlp_va_np is not None:
                ax2.plot(buses, mlp_va_np, "r:^", label="MLP Baseline", linewidth=1.5, markersize=5)

            ax2.set_xlabel("Bus Index (0 = Substation / Slack)", fontsize=12)
            ax2.set_ylabel("Voltage Angle θ (deg)", fontsize=12)
            ax2.grid(True, linestyle="--", alpha=0.6)
            ax2.legend(loc="best", fontsize=11)

            plt.tight_layout()
            out_img = os.path.join(here, f"snapshot_{snapshot_idx}_drop{drop}_{condition.lower()}_comparison.png")
            plt.savefig(out_img, dpi=300)
            plt.close()
            print(f"\n✓ Saved comparison visualization graph: {out_img}")
        except Exception as e:
            print(f"(Note: Could not generate comparison plot: {e})")

    print("=" * 90 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive Single-Snapshot State Estimation Comparison")
    parser.add_argument("--condition", type=str, default="Augmented", choices=["Clean", "Augmented"])
    parser.add_argument("--drop", type=int, default=50, choices=[10, 20, 30, 40, 50])
    parser.add_argument("--snapshot", type=int, default=0, help="Snapshot index in test file")
    parser.add_argument("--data", type=str, default="../MLP/data/ieee33")
    parser.add_argument("--gnn_ckpt", type=str, default=None, help="Explicit path to GNN checkpoint")
    parser.add_argument("--mlp_ckpt", type=str, default=None, help="Explicit path to MLP checkpoint")
    parser.add_argument("--plot", action="store_true", help="Generate and save visual comparison graph")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    run_comparison(
        condition=args.condition,
        drop=args.drop,
        snapshot_idx=args.snapshot,
        data_dir=args.data,
        gnn_ckpt=args.gnn_ckpt,
        mlp_ckpt=args.mlp_ckpt,
        save_plot=args.plot,
        device_str=args.device,
    )
