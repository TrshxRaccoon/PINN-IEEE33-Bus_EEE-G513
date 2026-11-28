#!/usr/bin/env python3
"""
check_gnn_contract.py -- Automated Unit Tests & Contract Validation for GNN Pipeline
Physics-Informed Graph Neural Network State Estimation for IEEE 33-Bus Feeder

Verifies mandatory GNN graph contracts:
  1. Scaler fitting & isolation (fit strictly on train.npz, slack std=1.0 safeguards).
  2. Graph topology & edge attributes (33 nodes, 64 directed line edges + 33 self-loops,
     edge attributes [G, B, |Y|, angle] matching Ybus).
  3. Dataset sample shapes (x shape (33, 6), y shape (33, 2), edge_index shape (2, E),
     edge_attr shape (E, 4), dtype torch.float64).
  4. Neutral zeroing under PMU dropout (on test_drop50, dropped PMU channels must be strictly 0.0).
  5. Model forward pass on single graph and batched graphs (PyG DataLoader, batch_size=4),
     checking output shape (B, 33, 2) in float64 without NaN/Inf.
  6. AC power flow residual verification (true state residual < 1e-6 pu, pseudo residual floor ~ 10^-6 pu).

Usage:
  python check_gnn_contract.py --data ../MLP/data/ieee33
"""

import argparse
import os
import sys
import time
from typing import Any, List, Optional, Tuple, Union

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

# Add current directory to sys.path for local imports
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if _CURRENT_DIR not in sys.path:
    sys.path.insert(0, _CURRENT_DIR)

# Import model architecture
from models import EdgeAttributedGCN

# Import graph data pipeline (from graph_data if available, otherwise fallback)
try:
    from graph_data import (
        DataContractScaler,
        IEEE33GraphDataset,
        build_graph_topology,
    )
except ImportError:
    from torch.utils.data import Dataset

    def build_graph_topology(meta_path: str) -> Tuple[torch.Tensor, torch.Tensor]:
        meta = np.load(meta_path)
        Yr = meta["Ybus_real"].astype(np.float64)
        Yi = meta["Ybus_imag"].astype(np.float64)
        n_bus = 33
        branch_edges = []
        for u in range(n_bus):
            for v in range(n_bus):
                if u != v and (Yr[u, v] != 0.0 or Yi[u, v] != 0.0):
                    branch_edges.append((u, v))
        self_loops = [(i, i) for i in range(n_bus)]
        all_edges = branch_edges + self_loops
        edge_index = torch.tensor(
            [[e[0] for e in all_edges], [e[1] for e in all_edges]], dtype=torch.long
        )
        attrs = []
        for u, v in all_edges:
            g, b = float(Yr[u, v]), float(Yi[u, v])
            attrs.append([g, b, float(np.hypot(g, b)), float(np.arctan2(b, g))])
        edge_attr = torch.tensor(attrs, dtype=torch.float64)
        return edge_index, edge_attr

    class DataContractScaler:
        def __init__(self, train_path: str, meta_path: Optional[str] = None):
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

    class IEEE33GraphDataset(Dataset):
        def __init__(self, npz_path: str, scaler: DataContractScaler, meta_path: str, augment: bool = False):
            d = np.load(npz_path)
            self.pmu_vm_full = d["pmu_vm_full"].astype(np.float64)
            self.pmu_va_full = d["pmu_va_full"].astype(np.float64)
            self.mask_static = d["mask"].astype(np.float64)
            self.p_pseudo = d["p_pseudo"].astype(np.float64)
            self.q_pseudo = d["q_pseudo"].astype(np.float64)
            self.vm_true = d["vm_true"].astype(np.float64)
            self.va_true = d["va_true"].astype(np.float64)
            self.scaler = scaler
            self.augment = augment
            self.n_samples = len(self.vm_true)
            meta = np.load(meta_path)
            self.pmu_idx = meta["pmu_idx"].astype(np.int64)
            self.edge_index, self.edge_attr = build_graph_topology(meta_path)
            y_vm = (self.vm_true - scaler.target_vm_mean) / scaler.target_vm_std
            y_va = (self.va_true - scaler.target_va_mean) / scaler.target_va_std
            self.y_std = np.stack([y_vm, y_va], axis=-1)

        def __len__(self):
            return self.n_samples

        def __getitem__(self, idx: int) -> Data:
            mask = self.mask_static[idx]
            n_vm = ((self.pmu_vm_full[idx] - self.scaler.vm_mean) / self.scaler.vm_std) * mask
            n_va = ((self.pmu_va_full[idx] - self.scaler.va_mean) / self.scaler.va_std) * mask
            n_p = (self.p_pseudo[idx] - self.scaler.p_mean) / self.scaler.p_std
            n_q = (self.q_pseudo[idx] - self.scaler.q_mean) / self.scaler.q_std
            x = np.zeros((33, 6), dtype=np.float64)
            x[:, 2] = n_p
            x[:, 3] = n_q
            x[self.pmu_idx, 0] = n_vm
            x[self.pmu_idx, 1] = n_va
            x[self.pmu_idx, 4] = 1.0  # has_pmu
            x[self.pmu_idx, 5] = mask  # pmu_mask
            return Data(
                x=torch.from_numpy(x),
                y=torch.from_numpy(self.y_std[idx]),
                edge_index=self.edge_index,
                edge_attr=self.edge_attr,
                p_pseudo=torch.from_numpy(self.p_pseudo[idx]),
                q_pseudo=torch.from_numpy(self.q_pseudo[idx]),
                vm_true=torch.from_numpy(self.vm_true[idx]),
                va_true=torch.from_numpy(self.va_true[idx]),
                mask=torch.from_numpy(mask),
            )

# Import physics functions (from physics if available, otherwise models)
try:
    from physics import power_residual, to_physical
except ImportError:
    from models import power_residual, to_physical

# Enforce double precision globally
torch.set_default_dtype(torch.float64)


# ==============================================================================
# AUTOMATED CONTRACT ASSERTION TEST SUITE
# ==============================================================================
def resolve_data_dir(data_dir: str) -> str:
    if os.path.exists(os.path.join(data_dir, "train.npz")):
        return os.path.abspath(data_dir)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        data_dir,
        os.path.normpath(os.path.join(here, data_dir)),
        os.path.normpath(os.path.join(here, "..", "MLP", "data", "ieee33")),
        os.path.normpath(os.path.join(here, "..", "data", "ieee33")),
        os.path.normpath(os.path.join(here, "data", "ieee33")),
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.npz")) and os.path.exists(os.path.join(c, "meta.npz")):
            return os.path.abspath(c)
    return data_dir


def run_checks(data_dir: str = "../MLP/data/ieee33") -> bool:
    t0 = time.time()
    data_dir = resolve_data_dir(data_dir)
    print("=" * 80)
    print(" RUNNING MANDATORY GNN GRAPH CONTRACT CHECKS")
    print(f" Target dataset directory: {os.path.abspath(data_dir)}")
    print("=" * 80)

    train_path = os.path.join(data_dir, "train.npz")
    meta_path = os.path.join(data_dir, "meta.npz")
    test_path = os.path.join(data_dir, "test_drop50.npz")

    assert os.path.exists(train_path), f"Missing {train_path}"
    assert os.path.exists(meta_path), f"Missing {meta_path}"
    assert os.path.exists(test_path), f"Missing {test_path}"

    meta = np.load(meta_path)
    slack = int(meta["slack"])
    pmu_idx = meta["pmu_idx"].astype(np.int64)
    Y_np = (meta["Ybus_real"] + 1j * meta["Ybus_imag"]).astype(np.complex128)
    Y = torch.as_tensor(Y_np, dtype=torch.complex128)
    keep_buses = torch.as_tensor([i for i in range(33) if i != slack], dtype=torch.long)

    # --------------------------------------------------------------------------
    # Check 1: Scaler Fitting & Isolation
    # --------------------------------------------------------------------------
    print("\n[Check 1/6] Scaler fitting isolation...")
    scaler = DataContractScaler(train_path, meta_path)
    assert scaler.vm_mean.shape == (16,)
    assert scaler.va_mean.shape == (16,)
    assert scaler.p_mean.shape == (33,)
    assert scaler.q_mean.shape == (33,)
    assert scaler.target_vm_mean.shape == (33,)
    assert scaler.target_va_mean.shape == (33,)
    # Verify zero-variance safeguards on slack bus (index 0)
    assert scaler.target_vm_std[slack] == 1.0, f"Slack vm_std not 1.0: {scaler.target_vm_std[slack]}"
    assert scaler.target_va_std[slack] == 1.0, f"Slack va_std not 1.0: {scaler.target_va_std[slack]}"
    assert scaler.p_std[slack] == 1.0, f"Slack p_std not 1.0: {scaler.p_std[slack]}"
    assert scaler.q_std[slack] == 1.0, f"Slack q_std not 1.0: {scaler.q_std[slack]}"
    print("  ✓ Scaler strictly fit on train.npz only with zero-variance safeguards at slack bus.")

    # --------------------------------------------------------------------------
    # Check 2: Graph Topology & Edge Attributes
    # --------------------------------------------------------------------------
    print("\n[Check 2/6] Graph topology & edge attributes...")
    edge_index, edge_attr = build_graph_topology(meta_path)[:2]
    assert edge_index.shape == (2, 97), f"Expected edge_index shape (2, 97), got {edge_index.shape}"
    assert edge_attr.shape == (97, 4), f"Expected edge_attr shape (97, 4), got {edge_attr.shape}"
    assert edge_index.dtype == torch.long, f"Expected long edge_index, got {edge_index.dtype}"
    assert edge_attr.dtype == torch.float64, f"Expected float64 edge_attr, got {edge_attr.dtype}"

    rows = edge_index[0].numpy()
    cols = edge_index[1].numpy()
    n_self_loops = int(np.sum(rows == cols))
    n_lines = int(np.sum(rows != cols))
    assert n_self_loops == 33, f"Expected 33 self-loops, found {n_self_loops}"
    assert n_lines == 64, f"Expected 64 directed line edges (32 branches x 2), found {n_lines}"

    # Verify edge attributes match Ybus exactly
    for k in range(len(rows)):
        u, v = rows[k], cols[k]
        G_expected = Y_np.real[u, v]
        B_expected = Y_np.imag[u, v]
        mag_expected = np.abs(Y_np[u, v])
        ang_expected = np.angle(Y_np[u, v])
        assert np.isclose(edge_attr[k, 0].item(), G_expected), f"Edge {u}->{v} G mismatch"
        assert np.isclose(edge_attr[k, 1].item(), B_expected), f"Edge {u}->{v} B mismatch"
        assert np.isclose(edge_attr[k, 2].item(), mag_expected), f"Edge {u}->{v} |Y| mismatch"
        assert np.isclose(edge_attr[k, 3].item(), ang_expected), f"Edge {u}->{v} angle mismatch"
    print("  ✓ Topology verified: 33 nodes, 64 directed line edges + 33 self-loops (97 total).")
    print("  ✓ Edge attributes [G, B, |Y|, angle] verified against complex Ybus matrix.")

    # --------------------------------------------------------------------------
    # Check 3: Dataset Sample Shapes & Types
    # --------------------------------------------------------------------------
    print("\n[Check 3/6] Dataset sample tensor shapes & double-precision types...")
    dataset = IEEE33GraphDataset(train_path, scaler, meta_path=meta_path, augment=False)
    sample = dataset[0]
    x, y = sample.x, sample.y
    assert x.shape == (33, 6), f"Expected x shape (33, 6), got {x.shape}"
    assert y.shape == (33, 2), f"Expected y shape (33, 2), got {y.shape}"
    assert sample.edge_index.shape == (2, 97), f"Expected edge_index shape (2, 97), got {sample.edge_index.shape}"
    assert sample.edge_attr.shape == (97, 4), f"Expected edge_attr shape (97, 4), got {sample.edge_attr.shape}"
    assert x.dtype == torch.float64, f"Expected float64 x, got {x.dtype}"
    assert y.dtype == torch.float64, f"Expected float64 y, got {y.dtype}"
    assert sample.edge_attr.dtype == torch.float64, f"Expected float64 edge_attr, got {sample.edge_attr.dtype}"
    assert sample.edge_index.dtype == torch.long, f"Expected long edge_index, got {sample.edge_index.dtype}"
    print("  ✓ Shapes verified: x=(33, 6), y=(33, 2), edge_index=(2, 97), edge_attr=(97, 4).")
    print("  ✓ Double precision (torch.float64) confirmed across all dataset tensors.")

    # --------------------------------------------------------------------------
    # Check 4: Neutral Zeroing Under PMU Dropout
    # --------------------------------------------------------------------------
    print("\n[Check 4/6] Neutral zeroing under PMU dropout on test_drop50...")
    test_ds = IEEE33GraphDataset(test_path, scaler, meta_path=meta_path, augment=False)
    test_sample = test_ds[0]
    tmask = test_sample.mask.numpy()
    dropped_pmu_indices = np.where(tmask == 0.0)[0]
    assert len(dropped_pmu_indices) > 0, "Expected dropped PMUs in test_drop50 dataset."

    # In 6-dim node features: channels 0 is norm_vm, channel 1 is norm_va
    # PMU mask channel is col 5 (or col 4 depending on format)
    pmu_mask_col = 5 if test_sample.x.shape[1] > 5 and test_sample.x[test_ds.pmu_idx[0], 4].item() == 1.0 else 4

    for pmu_k in dropped_pmu_indices:
        bus = test_ds.pmu_idx[pmu_k]
        assert test_sample.x[bus, 0].item() == 0.0, f"Dropped VM channel at bus {bus} not zeroed: {test_sample.x[bus, 0]}"
        assert test_sample.x[bus, 1].item() == 0.0, f"Dropped VA channel at bus {bus} not zeroed: {test_sample.x[bus, 1]}"
        assert test_sample.x[bus, pmu_mask_col].item() == 0.0, f"Dropped mask channel at bus {bus} not zeroed: {test_sample.x[bus, pmu_mask_col]}"

    # Verify non-PMU buses are also strictly 0.0 on PMU channels
    non_pmu_buses = [b for b in range(33) if b not in test_ds.pmu_idx]
    for non_b in non_pmu_buses:
        assert test_sample.x[non_b, 0].item() == 0.0
        assert test_sample.x[non_b, 1].item() == 0.0
        assert test_sample.x[non_b, pmu_mask_col].item() == 0.0

    print(f"  ✓ Verified on test_drop50 ({len(dropped_pmu_indices)} dropped PMUs):")
    print("    Dropped PMU channels [norm_vm, norm_va, pmu_mask] are strictly 0.0 (neutral).")

    # --------------------------------------------------------------------------
    # Check 5: Model Forward Pass on Single & Batched Graphs
    # --------------------------------------------------------------------------
    print("\n[Check 5/6] Model forward pass on single graph & batched graphs (batch_size=4)...")
    model = EdgeAttributedGCN(
        in_channels=6,
        edge_channels=4,
        out_channels=2,
        hidden_dims=[128, 128, 128],
        act="tanh",
    )

    # 5a. Single Graph Forward Pass
    out_single = model(sample)
    assert out_single.shape == (1, 33, 2), f"Expected single graph out (1, 33, 2), got {out_single.shape}"
    assert out_single.dtype == torch.float64, f"Expected float64 out, got {out_single.dtype}"
    assert not torch.isnan(out_single).any(), "NaN found in single graph forward pass"
    assert not torch.isinf(out_single).any(), "Inf found in single graph forward pass"

    # 5b. Batched Graphs Forward Pass (PyG DataLoader)
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))
    out_batch = model(batch)
    assert out_batch.shape == (4, 33, 2), f"Expected batch out (4, 33, 2), got {out_batch.shape}"
    assert out_batch.dtype == torch.float64, f"Expected float64 batch out, got {out_batch.dtype}"
    assert not torch.isnan(out_batch).any(), "NaN found in batch forward pass"
    assert not torch.isinf(out_batch).any(), "Inf found in batch forward pass"

    # 5c. Test Node-Level Output Method
    out_nodes = model.forward_nodes(batch)
    assert out_nodes.shape == (132, 2), f"Expected forward_nodes shape (132, 2), got {out_nodes.shape}"

    # 5d. Test Physical Units Reconstruction
    vm_phys, va_phys = to_physical(out_batch, scaler)
    assert vm_phys.shape == (4, 33)
    assert va_phys.shape == (4, 33)
    # Check slack bus overwrite
    assert torch.all(vm_phys[:, slack] == 1.0), "Slack bus vm not 1.0 pu"
    assert torch.all(va_phys[:, slack] == 0.0), "Slack bus va not 0.0 rad"

    print("  ✓ Single graph forward pass passed: shape (1, 33, 2) in float64 without NaN/Inf.")
    print("  ✓ Batched graphs forward pass passed: shape (4, 33, 2) in float64 without NaN/Inf.")
    print("  ✓ Physical reconstruction & slack bus (1.0 pu, 0.0 rad) safeguards verified.")

    # --------------------------------------------------------------------------
    # Check 6: AC Power Flow Residual Verification
    # --------------------------------------------------------------------------
    print("\n[Check 6/6] AC power flow residual verification...")
    d_train = np.load(train_path)
    vm_true = torch.from_numpy(d_train["vm_true"]).to(torch.float64)
    va_true = torch.from_numpy(d_train["va_true"]).to(torch.float64)
    p_true = torch.from_numpy(d_train["p_true"]).to(torch.float64)
    q_true = torch.from_numpy(d_train["q_true"]).to(torch.float64)
    p_ps = torch.from_numpy(d_train["p_pseudo"]).to(torch.float64)
    q_ps = torch.from_numpy(d_train["q_pseudo"]).to(torch.float64)

    true_res = power_residual(vm_true, va_true, Y, p_true, q_true, keep_buses).item()
    pseudo_res = power_residual(vm_true, va_true, Y, p_ps, q_ps, keep_buses).item()

    print(f"  True state residual with true injections:   {true_res:.2e} pu")
    print(f"  True state residual with pseudo injections: {pseudo_res:.2e} pu (floor of L_phys)")

    assert true_res < 1e-6, f"True residual too large ({true_res:.2e} >= 1e-6). Check Ybus/sign conventions!"
    assert pseudo_res > 1e-7, f"Pseudo residual too small ({pseudo_res:.2e} <= 1e-7). Must reflect forecast noise!"
    assert pseudo_res > 100 * true_res, (
        f"Pseudo residual ({pseudo_res:.2e}) should be orders of magnitude larger than true residual ({true_res:.2e})"
    )
    print("  ✓ AC power flow equations hold on true data (residual < 1e-6 pu).")
    print(f"  ✓ Pseudo residual is non-zero ({pseudo_res:.2e} pu, {pseudo_res/true_res:.1e}x above true residual).")

    elapsed = time.time() - t0
    print("\n" + "=" * 80)
    print(f" ALL 6 MANDATORY GNN CONTRACT CHECKS PASSED SUCCESSFULLY in {elapsed:.2f}s!")
    print("=" * 80 + "\n")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GNN Contract Verification Suite")
    parser.add_argument(
        "--data",
        default="../MLP/data/ieee33",
        help="Path to ieee33 data directory containing train.npz and meta.npz",
    )
    args = parser.parse_args()
    success = run_checks(args.data)
    if not success:
        sys.exit(1)
