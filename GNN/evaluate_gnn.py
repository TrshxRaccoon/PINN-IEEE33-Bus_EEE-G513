#!/usr/bin/env python3
"""
evaluate_gnn.py -- Standalone Model Evaluation Script for Edge-Attributed GCN

Evaluates any trained GNN checkpoint on specified test datasets, reporting:
  - Voltage Magnitude |V| RMSE (per-unit)
  - Voltage Phase Angle theta RMSE (degrees)
  - AC Power Flow Physics Residual L_phys (per-unit)
  - Max bus error and individual bus error percentiles

Usage:
  python evaluate_gnn.py --checkpoint results_augmented/best_model_Augmented_seed42.pt --data ../MLP/data/ieee33
  python evaluate_gnn.py --checkpoint results_clean/model_Clean_lam0.1_seed42.pt --drop 50
"""

import argparse
import os
import sys
from typing import Dict, List, Any

import numpy as np
import torch
from torch_geometric.loader import DataLoader

here = os.path.dirname(os.path.abspath(__file__))
if here not in sys.path:
    sys.path.insert(0, here)

from graph_data import DataContractScaler, IEEE33GraphDataset, resolve_meta_path
from models import EdgeAttributedGCN
from physics import compute_voltage_errors, load_ybus, power_residual, to_physical

# Enforce double precision
torch.set_default_dtype(torch.float64)


def evaluate_checkpoint(
    ckpt_path: str,
    data_dir: str = "../MLP/data/ieee33",
    drops: List[int] = None,
    device_str: str = "cpu",
    batch_size: int = 128,
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found at: {ckpt_path}")

    device = torch.device(device_str)
    print(f"\n================================================================================")
    print(f" EVALUATING GNN CHECKPOINT: {ckpt_path}")
    print(f" Target Data: {data_dir} | Device: {device}")
    print(f"================================================================================")

    # 1. Load Checkpoint
    ckpt = torch.load(ckpt_path, map_location=device)
    hidden_dims = ckpt.get("hidden_dims", [128, 128, 128])
    act = ckpt.get("act", "tanh")
    condition = ckpt.get("condition", "Unknown")
    lam = ckpt.get("lambda", ckpt.get("best_lambda", "Unknown"))
    state_dict = ckpt.get("model_state_dict", ckpt)

    print(f"Model Condition: {condition} | Lambda: {lam}")
    print(f"Architecture: EdgeAttributedGCN with hidden_dims={hidden_dims}, act={act}")

    # 2. Build Model
    model = EdgeAttributedGCN(
        in_channels=6,
        edge_channels=4,
        out_channels=2,
        hidden_dims=hidden_dims,
        act=act,
        num_nodes=33,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    # 3. Scaler and Physics Constants
    meta_path = resolve_meta_path(os.path.join(data_dir, "meta.npz"))
    train_path = os.path.join(data_dir, "train.npz")
    scaler = DataContractScaler(train_path=train_path, meta_path=meta_path)
    Y = load_ybus(meta_path, device=device)

    meta = np.load(meta_path)
    slack = int(meta["slack"].item()) if hasattr(meta["slack"], "item") else int(meta["slack"])
    n_bus = int(meta["n_bus"].item()) if hasattr(meta["n_bus"], "item") else int(meta["n_bus"])
    keep_buses = torch.tensor([i for i in range(n_bus) if i != slack], dtype=torch.long, device=device)

    if drops is None:
        drops = [10, 20, 30, 40, 50]

    print("\n" + "-" * 75)
    print(f"{'Dropout':<10} | {'Coverage':<10} | {'|V| RMSE (pu)':<16} | {'theta RMSE (deg)':<18} | {'L_phys (pu)':<14}")
    print("-" * 75)

    results = {}
    with torch.no_grad():
        for drop in drops:
            test_file = os.path.join(data_dir, f"test_drop{drop}.npz")
            if not os.path.exists(test_file):
                print(f"Skipping {test_file} (file not found)")
                continue

            dataset = IEEE33GraphDataset(test_file, scaler, meta_path=meta_path, augment=False)
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

            vm_err_sq, va_err_sq, phys_residuals = [], [], []

            for batch in loader:
                batch = batch.to(device)
                out = model(batch, reshape_batch=True)
                vm_pred, va_pred = to_physical(out, scaler, device)

                bs = out.size(0)
                vm_true = batch.vm_true.view(bs, 33)
                va_true = batch.va_true.view(bs, 33)
                p_ps = batch.p_pseudo.view(bs, 33)
                q_ps = batch.q_pseudo.view(bs, 33)

                vm_err, va_err_deg = compute_voltage_errors(
                    vm_pred, va_pred, vm_true, va_true, keep_buses
                )
                vm_err_sq.append((vm_err ** 2).cpu().numpy())
                va_err_sq.append((va_err_deg ** 2).cpu().numpy())

                res = power_residual(vm_pred, va_pred, Y, p_ps, q_ps, keep_buses)
                phys_residuals.append(res.item())

            rmse_v = float(np.sqrt(np.mean(np.concatenate(vm_err_sq, axis=0))))
            rmse_va_deg = float(np.sqrt(np.mean(np.concatenate(va_err_sq, axis=0))))
            avg_l_phys = float(np.mean(phys_residuals))
            cov = 100 - drop

            print(f"{f'{drop}%':<10} | {f'{cov}%':<10} | {rmse_v:<16.6f} | {rmse_va_deg:<18.4f} | {avg_l_phys:<14.3e}")

            results[drop] = {
                "rmse_v": rmse_v,
                "rmse_va_deg": rmse_va_deg,
                "l_phys": avg_l_phys,
                "coverage": cov,
            }

    print("-" * 75)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate GNN State Estimator Checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pt checkpoint file")
    parser.add_argument("--data", type=str, default="../MLP/data/ieee33", help="Path to data directory")
    parser.add_argument("--drop", nargs="+", type=int, default=[10, 20, 30, 40, 50], help="Dropout levels")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--bs", type=int, default=128, help="Batch size")
    args = parser.parse_args()

    evaluate_checkpoint(
        ckpt_path=args.checkpoint,
        data_dir=args.data,
        drops=args.drop,
        device_str=args.device,
        batch_size=args.bs,
    )
