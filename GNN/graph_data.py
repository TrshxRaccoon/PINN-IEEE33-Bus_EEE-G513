#!/usr/bin/env python3
"""
graph_data.py -- Graph Data Pipeline for Physics-Informed GNN State Estimator
IEEE 33-Bus Feeder Topology and Dataset Processing.

Strict implementation of:
  - Double precision (torch.float64 and torch.complex128)
  - Zero data leakage: DataContractScaler fit strictly on train.npz only
  - Standardize first, then multiply PMU features by mask so missing channels become 0.0
  - Graph topology: 33 buses, 64 directed branch edges, 33 self-loops (97 total edges)
  - 4-dimensional edge attributes from Ybus: [Re(Y_ij), Im(Y_ij), |Y_ij|, atan2(Im(Y_ij), Re(Y_ij))]
  - 6-dimensional node features: [norm_vm, norm_va, norm_p, norm_q, has_pmu, pmu_mask]
  - PyTorch Geometric Data object construction
"""

import os
import math
from typing import Tuple, Optional, Union, Sequence, Dict, Any
import numpy as np
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

# Ensure double precision by default
torch.set_default_dtype(torch.float64)


# ==============================================================================
# Path Resolution Helpers
# ==============================================================================
def resolve_meta_path(meta_path: str = "../MLP/data/ieee33/meta.npz") -> str:
    """Find meta.npz in current directory or relative fallback locations."""
    if os.path.exists(meta_path):
        return os.path.abspath(meta_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(here, meta_path)),
        os.path.normpath(os.path.join(here, "..", "MLP", "data", "ieee33", "meta.npz")),
        os.path.normpath(os.path.join(here, "..", "data", "ieee33", "meta.npz")),
        os.path.normpath(os.path.join(here, "data", "ieee33", "meta.npz")),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"meta.npz could not be located at '{meta_path}' or candidate paths: {candidates}")


def resolve_train_path(train_path: str = "../MLP/data/ieee33/train.npz") -> str:
    """Find train.npz in current directory or relative fallback locations."""
    if os.path.exists(train_path):
        return os.path.abspath(train_path)
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.normpath(os.path.join(here, train_path)),
        os.path.normpath(os.path.join(here, "..", "MLP", "data", "ieee33", "train.npz")),
        os.path.normpath(os.path.join(here, "..", "data", "ieee33", "train.npz")),
        os.path.normpath(os.path.join(here, "data", "ieee33", "train.npz")),
    ]
    for c in candidates:
        if os.path.exists(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"train.npz could not be located at '{train_path}' or candidate paths: {candidates}")


# ==============================================================================
# 1. GRAPH TOPOLOGY & EDGE ATTRIBUTE CONSTRUCTION
# ==============================================================================
def build_graph_topology(
    meta_path: str = "../MLP/data/ieee33/meta.npz",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Constructs graph topology for IEEE 33-bus feeder from meta.npz:
      - 33 buses (nodes 0..32, slack bus is index 0)
      - 32 physical branches -> 64 directed edges (u, v) and (v, u)
      - Self-loops (i, i) for all 33 buses
      - Total edges E = 64 + 33 = 97
      - Extracts 4D edge attributes from Ybus:
          G_ij = Re(Y_ij)
          B_ij = Im(Y_ij)
          |Y_ij| = sqrt(G_ij^2 + B_ij^2)
          angle_ij = atan2(B_ij, G_ij)

    Returns:
      edge_index: (2, E) torch.long tensor
      edge_attr:  (E, 4) torch.float64 tensor in physical per-unit values
    """
    resolved_path = resolve_meta_path(meta_path)
    meta = np.load(resolved_path)
    Yr = meta["Ybus_real"].astype(np.float64)
    Yi = meta["Ybus_imag"].astype(np.float64)
    n_bus = int(meta["n_bus"].item()) if hasattr(meta["n_bus"], "item") else int(meta["n_bus"])

    # 1. 64 directed branch edges (u != v where Y_uv is non-zero)
    branch_edges = []
    for u in range(n_bus):
        for v in range(n_bus):
            if u != v and (Yr[u, v] != 0.0 or Yi[u, v] != 0.0):
                branch_edges.append((u, v))

    if len(branch_edges) != 64:
        raise ValueError(f"Expected 64 directed branch edges in IEEE 33 feeder, got {len(branch_edges)}")

    # 2. 33 self-loops (i, i) for all buses
    self_loops = [(i, i) for i in range(n_bus)]

    # Concatenate branch edges + self-loops: E = 64 + 33 = 97
    all_edges = branch_edges + self_loops

    src = [e[0] for e in all_edges]
    dst = [e[1] for e in all_edges]
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # 3. Extract 4-dimensional edge attributes
    attrs = []
    for u, v in all_edges:
        g = float(Yr[u, v])
        b = float(Yi[u, v])
        y_mag = float(np.hypot(g, b))
        y_ang = float(np.arctan2(b, g))
        attrs.append([g, b, y_mag, y_ang])

    edge_attr = torch.tensor(attrs, dtype=torch.float64)
    return edge_index, edge_attr


# ==============================================================================
# 2. DATA CONTRACT SCALER (Fit strictly on train.npz only)
# ==============================================================================
class DataContractScaler:
    """
    Standardizer fit STRICTLY on train.npz only (preventing any data leakage).
    Computes mean and std for:
      - PMU |V| and theta across train PMU measurements (16 PMUs)
      - Pseudo injections P and Q across all 33 buses
      - Target true |V| and theta across all 33 buses (slack bus std set to 1.0)
      - Edge attributes mean and std across grid branches
    """
    def __init__(
        self,
        train_path: str = "../MLP/data/ieee33/train.npz",
        meta_path: str = "../MLP/data/ieee33/meta.npz",
    ):
        resolved_train = resolve_train_path(train_path)
        resolved_meta = resolve_meta_path(meta_path)

        d = np.load(resolved_train)

        # 1. PMU measurements (fit on complete/unmasked readings)
        vm_full = d["pmu_vm_full"].astype(np.float64)
        va_full = d["pmu_va_full"].astype(np.float64)
        self.vm_mean = vm_full.mean(axis=0)
        self.vm_std = vm_full.std(axis=0)
        self.vm_std[self.vm_std < 1e-8] = 1.0

        self.va_mean = va_full.mean(axis=0)
        self.va_std = va_full.std(axis=0)
        self.va_std[self.va_std < 1e-8] = 1.0

        # 2. Pseudo injections (33 buses)
        p_ps = d["p_pseudo"].astype(np.float64)
        q_ps = d["q_pseudo"].astype(np.float64)
        self.p_mean = p_ps.mean(axis=0)
        self.p_std = p_ps.std(axis=0)
        self.p_std[self.p_std < 1e-8] = 1.0  # slack bus has zero injection variance -> 1.0

        self.q_mean = q_ps.mean(axis=0)
        self.q_std = q_ps.std(axis=0)
        self.q_std[self.q_std < 1e-8] = 1.0  # slack bus has zero injection variance -> 1.0

        # 3. Target true voltages and angles (33 buses)
        vm_true = d["vm_true"].astype(np.float64)
        va_true = d["va_true"].astype(np.float64)
        self.target_vm_mean = vm_true.mean(axis=0)
        self.target_vm_std = vm_true.std(axis=0)
        self.target_vm_std[self.target_vm_std < 1e-8] = 1.0  # slack bus constant 1.0 pu -> 1.0

        self.target_va_mean = va_true.mean(axis=0)
        self.target_va_std = va_true.std(axis=0)
        self.target_va_std[self.target_va_std < 1e-8] = 1.0  # slack bus constant 0.0 rad -> 1.0

        # 4. Edge attributes mean and std
        _, edge_attr = build_graph_topology(resolved_meta)
        e_attr_np = edge_attr.numpy()
        self.edge_attr_mean = e_attr_np.mean(axis=0)
        self.edge_attr_std = e_attr_np.std(axis=0)
        self.edge_attr_std[self.edge_attr_std < 1e-8] = 1.0


# Alias for convenience
GNNDataContractScaler = DataContractScaler


# ==============================================================================
# 3. IEEE 33-BUS GRAPH DATASET
# ==============================================================================
class IEEE33GraphDataset(Dataset):
    """
    IEEE 33-Bus Graph Dataset yielding torch_geometric.data.Data objects.

    Node features x: (33, 6) float64 tensor:
      - col 0: norm_vm (standardized PMU |V|, 0.0 if not PMU or dropped)
      - col 1: norm_va (standardized PMU theta, 0.0 if not PMU or dropped)
      - col 2: norm_p (standardized P_pseudo across all 33 buses)
      - col 3: norm_q (standardized Q_pseudo across all 33 buses)
      - col 4: has_pmu (1.0 if PMU installed at bus, else 0.0)
      - col 5: pmu_mask (1.0 if PMU active, 0.0 if dropped or no PMU)

    Graph topology:
      - edge_index: (2, E) long tensor (E = 97)
      - edge_attr: (E, 4) float64 tensor [G_ij, B_ij, |Y_ij|, angle_ij]

    Targets and physical labels:
      - y: (33, 2) float64 tensor (standardized target [|V|, theta])
      - p_pseudo: (33,) float64 tensor (physical units)
      - q_pseudo: (33,) float64 tensor (physical units)
      - vm_true: (33,) float64 tensor (physical units)
      - va_true: (33,) float64 tensor (physical units)
      - mask: (16,) float64 tensor (PMU availability mask)
      - drop_frac: float (fraction of dropped PMUs)

    Strict rule:
      PMU features are standardized first, then multiplied by mask, so that
      missing/unmeasured channels become strictly 0.0 (the training mean).
    """
    def __init__(
        self,
        npz_path: str,
        scaler: DataContractScaler,
        meta_path: str = "../MLP/data/ieee33/meta.npz",
        augment: bool = False,
        seed: int = 42,
        standardize_edge_attr: bool = False,
    ):
        if not os.path.exists(npz_path):
            # Try resolving relative path
            here = os.path.dirname(os.path.abspath(__file__))
            cand = os.path.normpath(os.path.join(here, npz_path))
            if os.path.exists(cand):
                npz_path = cand
            else:
                raise FileNotFoundError(f"Dataset file not found at: {npz_path}")

        d = np.load(npz_path)
        self.pmu_vm_full = d["pmu_vm_full"].astype(np.float64)
        self.pmu_va_full = d["pmu_va_full"].astype(np.float64)
        self.mask_static = d["mask"].astype(np.float64)
        self.p_pseudo = d["p_pseudo"].astype(np.float64)
        self.q_pseudo = d["q_pseudo"].astype(np.float64)
        self.vm_true = d["vm_true"].astype(np.float64)
        self.va_true = d["va_true"].astype(np.float64)
        self.drop_frac_static = d["drop_frac"].astype(np.float64) if "drop_frac" in d else None

        self.scaler = scaler
        self.augment = augment
        self.n_samples = len(self.vm_true)
        self.n_pmu = self.pmu_vm_full.shape[1]
        self.rng = np.random.default_rng(seed)

        # Meta & graph topology
        resolved_meta = resolve_meta_path(meta_path)
        meta = np.load(resolved_meta)
        self.pmu_idx = meta["pmu_idx"].astype(np.int64)
        self.n_bus = int(meta["n_bus"].item()) if hasattr(meta["n_bus"], "item") else int(meta["n_bus"])
        self.slack = int(meta["slack"].item()) if hasattr(meta["slack"], "item") else int(meta["slack"])

        edge_index, edge_attr = build_graph_topology(resolved_meta)
        self.edge_index = edge_index  # (2, E) torch.long

        if standardize_edge_attr:
            e_mean = torch.as_tensor(scaler.edge_attr_mean, dtype=torch.float64)
            e_std = torch.as_tensor(scaler.edge_attr_std, dtype=torch.float64)
            self.edge_attr = (edge_attr - e_mean) / e_std
        else:
            self.edge_attr = edge_attr.to(torch.float64)

        # Static indicator: 1.0 if PMU is installed at bus, 0.0 otherwise
        self.has_pmu = np.zeros(self.n_bus, dtype=np.float64)
        self.has_pmu[self.pmu_idx] = 1.0

        # Vectorized target standardization: shape (N, 33, 2)
        y_vm = (self.vm_true - scaler.target_vm_mean) / scaler.target_vm_std
        y_va = (self.va_true - scaler.target_va_mean) / scaler.target_va_std
        self.y_std = np.stack([y_vm, y_va], axis=-1)

        # Vectorized pseudo injection standardization: shape (N, 33)
        self.norm_p = (self.p_pseudo - scaler.p_mean) / scaler.p_std
        self.norm_q = (self.q_pseudo - scaler.q_mean) / scaler.q_std

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> Data:
        # Determine PMU mask
        if self.augment:
            # Draw fresh random dropout level ~ U(0, 0.5) on the fly
            lvl = float(self.rng.uniform(0.0, 0.5))
            n_drop = int(round(lvl * self.n_pmu))
            perm = self.rng.permutation(self.n_pmu)
            mask = np.ones(self.n_pmu, dtype=np.float64)
            mask[perm[:n_drop]] = 0.0
            drop_frac = float(n_drop / self.n_pmu)
        else:
            mask = self.mask_static[idx]
            if self.drop_frac_static is not None:
                drop_frac = float(self.drop_frac_static[idx])
            else:
                drop_frac = float(1.0 - mask.mean())

        # Strict rule: Standardize FIRST, then multiply PMU features by mask
        std_pmu_vm = ((self.pmu_vm_full[idx] - self.scaler.vm_mean) / self.scaler.vm_std) * mask
        std_pmu_va = ((self.pmu_va_full[idx] - self.scaler.va_mean) / self.scaler.va_std) * mask

        # Assemble full 33-bus node feature arrays
        norm_vm_bus = np.zeros(self.n_bus, dtype=np.float64)
        norm_va_bus = np.zeros(self.n_bus, dtype=np.float64)
        pmu_mask_bus = np.zeros(self.n_bus, dtype=np.float64)

        norm_vm_bus[self.pmu_idx] = std_pmu_vm
        norm_va_bus[self.pmu_idx] = std_pmu_va
        pmu_mask_bus[self.pmu_idx] = mask

        # Node feature matrix x: (33, 6)
        x = np.stack(
            [
                norm_vm_bus,
                norm_va_bus,
                self.norm_p[idx],
                self.norm_q[idx],
                self.has_pmu,
                pmu_mask_bus,
            ],
            axis=-1,
        )

        # Construct and return PyG Data object
        return Data(
            x=torch.as_tensor(x, dtype=torch.float64),
            edge_index=self.edge_index,
            edge_attr=self.edge_attr,
            y=torch.as_tensor(self.y_std[idx], dtype=torch.float64),
            p_pseudo=torch.as_tensor(self.p_pseudo[idx], dtype=torch.float64),
            q_pseudo=torch.as_tensor(self.q_pseudo[idx], dtype=torch.float64),
            vm_true=torch.as_tensor(self.vm_true[idx], dtype=torch.float64),
            va_true=torch.as_tensor(self.va_true[idx], dtype=torch.float64),
            mask=torch.as_tensor(mask, dtype=torch.float64),
            drop_frac=drop_frac,
        )
