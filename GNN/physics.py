#!/usr/bin/env python3
"""
physics.py -- Physics Engine and Loss Functions for GNN State Estimation
IEEE 33-Bus AC Power Flow Residuals and Physical Conversions.

Strict implementation of:
  - Double precision (torch.float64 and torch.complex128)
  - AC power-flow injection residual loss:
      V = polar(vm, va) (complex128)
      S_calc = V * conj(V @ Y.T)
      rP = S_calc.real - p_ps
      rQ = S_calc.imag - q_ps
      L_phys = mean(rP[:, keep_buses]**2 + rQ[:, keep_buses]**2)
  - to_physical: de-standardize [|V|, theta] to physical units,
    and enforce slack bus physical constants (vm=1.0 pu, va=0.0 rad)
  - compute_data_loss: MSE loss computed exclusively over the 32 non-slack buses
"""

import os
from typing import Tuple, Optional, Union, Sequence, Any
import numpy as np
import torch

# Ensure double precision by default
torch.set_default_dtype(torch.float64)


# ==============================================================================
# Helper to load Ybus matrix
# ==============================================================================
def load_ybus(
    meta_path: str = "../MLP/data/ieee33/meta.npz",
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    Load complex bus admittance matrix Y from meta.npz as a torch.complex128 tensor.
    Shape: (33, 33).
    """
    if not os.path.exists(meta_path):
        here = os.path.dirname(os.path.abspath(__file__))
        candidates = [
            os.path.normpath(os.path.join(here, meta_path)),
            os.path.normpath(os.path.join(here, "..", "MLP", "data", "ieee33", "meta.npz")),
            "D:/BITS Pilani Goa Campus/Academics/ML4EE/ML_Project/MLP/data/ieee33/meta.npz",
        ]
        for c in candidates:
            if os.path.exists(c):
                meta_path = c
                break

    meta = np.load(meta_path)
    Yr = meta["Ybus_real"].astype(np.float64)
    Yi = meta["Ybus_imag"].astype(np.float64)
    Y = Yr + 1j * Yi
    Y_tensor = torch.as_tensor(Y, dtype=torch.complex128)
    if device is not None:
        Y_tensor = Y_tensor.to(device)
    return Y_tensor


# ==============================================================================
# 1. AC POWER FLOW RESIDUAL LOSS
# ==============================================================================
def power_residual(
    vm: torch.Tensor,
    va: torch.Tensor,
    Y: torch.Tensor,
    p_ps: torch.Tensor,
    q_ps: torch.Tensor,
    keep_buses: Optional[Union[torch.Tensor, Sequence[int]]] = None,
) -> torch.Tensor:
    """
    Computes AC power flow injection residual loss (Section 3 reference implementation):
      V = polar(vm, va) (complex128)
      S_calc = V * conj(V @ Y.T)
      rP = S_calc.real - p_ps
      rQ = S_calc.imag - q_ps
      return (rP[:, keep_buses]**2 + rQ[:, keep_buses]**2).mean()

    Parameters:
      vm: (batch, 33) float64 tensor of bus voltage magnitudes in physical pu
      va: (batch, 33) float64 tensor of bus voltage angles in physical rad
      Y: (33, 33) complex128 bus admittance matrix
      p_ps: (batch, 33) or (batch*33,) float64 pseudo active power injections in pu
      q_ps: (batch, 33) or (batch*33,) float64 pseudo reactive power injections in pu
      keep_buses: 1D index tensor/sequence of buses over which to compute physics loss
                  (default: non-slack buses 1..32)

    Returns:
      Scalar torch.float64 tensor with the physics loss
    """
    # Ensure float64 / complex128 precision
    if vm.dtype != torch.float64:
        vm = vm.to(torch.float64)
    if va.dtype != torch.float64:
        va = va.to(torch.float64)
    if Y.dtype != torch.complex128:
        Y = Y.to(torch.complex128)

    # Ensure 2D (batch, 33) shapes
    if vm.ndim == 1:
        vm = vm.unsqueeze(0)
        va = va.unsqueeze(0)
    nb = vm.shape[1]

    if p_ps.ndim == 1 and p_ps.shape[0] > nb:
        p_ps = p_ps.view(-1, nb)
        q_ps = q_ps.view(-1, nb)
    elif p_ps.ndim == 1:
        p_ps = p_ps.unsqueeze(0)
        q_ps = q_ps.unsqueeze(0)

    if p_ps.dtype != torch.float64:
        p_ps = p_ps.to(torch.float64)
    if q_ps.dtype != torch.float64:
        q_ps = q_ps.to(torch.float64)

    # Default to 32 non-slack buses (1..32)
    if keep_buses is None:
        keep_buses = torch.arange(1, nb, device=vm.device)
    elif not isinstance(keep_buses, torch.Tensor):
        keep_buses = torch.as_tensor(keep_buses, device=vm.device)

    # Complex bus voltage phasors
    V = torch.polar(vm, va)

    # Complex bus power injections: S_i = V_i * conj(I_i) where I = Y @ V
    # For row vectors V: (Y @ V.T).T = V @ Y.T
    S_calc = V * torch.conj(V @ Y.T)

    # Active and reactive injection residuals
    rP = S_calc.real - p_ps
    rQ = S_calc.imag - q_ps

    return (rP[:, keep_buses] ** 2 + rQ[:, keep_buses] ** 2).mean()


# ==============================================================================
# 2. TO_PHYSICAL CONVERSION (De-standardize & overwrite slack bus)
# ==============================================================================
def to_physical(
    y_norm: torch.Tensor,
    scaler: Any,
    device: Optional[torch.device] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    De-standardize standardized predictions y_norm back to physical units:
      vm = y_norm[:, :, 0] * target_vm_std + target_vm_mean
      va = y_norm[:, :, 1] * target_va_std + target_va_mean
    Overwrite slack bus (index 0) with known physical constants:
      vm[:, 0] = 1.0 pu
      va[:, 0] = 0.0 rad

    Supported shapes for y_norm:
      - (batch, 33, 2): standard 3D GNN node-level predictions
      - (batch * 33, 2): unbatched/disjoint PyG predictions
      - (batch, 66): 2D flattened predictions (concatenated [vm, va])
      - (33, 2): single unbatched graph
      - (66,): single unbatched flattened vector

    Returns:
      vm: (batch, 33) float64 tensor in physical per-unit [pu]
      va: (batch, 33) float64 tensor in physical radians [rad]
    """
    if device is None:
        device = y_norm.device
    elif not isinstance(device, torch.device):
        device = torch.device(device)

    nb = 33
    t_vm_mean = torch.as_tensor(scaler.target_vm_mean, dtype=torch.float64, device=device)
    t_vm_std = torch.as_tensor(scaler.target_vm_std, dtype=torch.float64, device=device)
    t_va_mean = torch.as_tensor(scaler.target_va_mean, dtype=torch.float64, device=device)
    t_va_std = torch.as_tensor(scaler.target_va_std, dtype=torch.float64, device=device)

    # Unpack based on input dimensions
    if y_norm.ndim == 3 and y_norm.shape[1] == nb and y_norm.shape[2] == 2:
        vm = y_norm[:, :, 0] * t_vm_std + t_vm_mean
        va = y_norm[:, :, 1] * t_va_std + t_va_mean
    elif y_norm.ndim == 2 and y_norm.shape[1] == 2:
        y_reshaped = y_norm.view(-1, nb, 2)
        vm = y_reshaped[:, :, 0] * t_vm_std + t_vm_mean
        va = y_reshaped[:, :, 1] * t_va_std + t_va_mean
    elif y_norm.ndim == 2 and y_norm.shape[1] == 2 * nb:
        vm = y_norm[:, :nb] * t_vm_std + t_vm_mean
        va = y_norm[:, nb:] * t_va_std + t_va_mean
    elif y_norm.ndim == 2 and y_norm.shape[0] == nb and y_norm.shape[1] == 2:
        vm = (y_norm[:, 0] * t_vm_std + t_vm_mean).unsqueeze(0)
        va = (y_norm[:, 1] * t_va_std + t_va_mean).unsqueeze(0)
    elif y_norm.ndim == 1 and y_norm.shape[0] == 2 * nb:
        vm = (y_norm[:nb] * t_vm_std + t_vm_mean).unsqueeze(0)
        va = (y_norm[nb:] * t_va_std + t_va_mean).unsqueeze(0)
    else:
        raise ValueError(f"Unsupported shape for y_norm: {y_norm.shape}. Expected (batch, 33, 2) or (batch, 66).")

    # Clone to prevent problematic in-place modification during autograd
    vm = vm.clone()
    va = va.clone()

    # Overwrite slack bus (index 0) with known physical constants
    vm[:, 0] = 1.0
    va[:, 0] = 0.0

    return vm, va


# ==============================================================================
# 3. DATA LOSS (Exclusively over 32 non-slack buses)
# ==============================================================================
def compute_data_loss(
    y_pred_norm: torch.Tensor,
    y_true_norm: torch.Tensor,
    keep_buses: Optional[Union[torch.Tensor, Sequence[int]]] = None,
) -> torch.Tensor:
    """
    Computes Mean Squared Error (MSE) loss exclusively over the 32 non-slack buses
    (excluding slack bus 0).

    Supported shapes:
      - (batch, 33, 2): 3D node predictions [norm_vm, norm_va]
      - (batch * 33, 2): disjoint PyG batched predictions
      - (batch, 66): 2D flattened predictions

    Returns:
      Scalar torch.float64 tensor with the data loss
    """
    if keep_buses is None:
        keep_buses = torch.arange(1, 33, device=y_pred_norm.device)
    elif not isinstance(keep_buses, torch.Tensor):
        keep_buses = torch.as_tensor(keep_buses, device=y_pred_norm.device)

    # 3D: (batch, 33, 2)
    if y_pred_norm.ndim == 3 and y_pred_norm.shape[1] == 33 and y_pred_norm.shape[2] == 2:
        diff = y_pred_norm[:, keep_buses, :] - y_true_norm[:, keep_buses, :]
        return (diff ** 2).mean()

    # 2D disjoint: (batch * 33, 2)
    if y_pred_norm.ndim == 2 and y_pred_norm.shape[1] == 2:
        yp = y_pred_norm.view(-1, 33, 2)
        yt = y_true_norm.view(-1, 33, 2)
        diff = yp[:, keep_buses, :] - yt[:, keep_buses, :]
        return (diff ** 2).mean()

    # 2D flattened: (batch, 66)
    if y_pred_norm.ndim == 2 and y_pred_norm.shape[1] == 66:
        vm_pred, va_pred = y_pred_norm[:, :33], y_pred_norm[:, 33:]
        vm_true, va_true = y_true_norm[:, :33], y_true_norm[:, 33:]
        diff_vm = vm_pred[:, keep_buses] - vm_true[:, keep_buses]
        diff_va = va_pred[:, keep_buses] - va_true[:, keep_buses]
        diff_total = torch.cat([diff_vm, diff_va], dim=-1)
        return (diff_total ** 2).mean()

    raise ValueError(f"Unsupported shape for y_pred_norm: {y_pred_norm.shape}")


# ==============================================================================
# 4. EVALUATION METRICS HELPER
# ==============================================================================
def compute_voltage_errors(
    vm_pred: torch.Tensor,
    va_pred: torch.Tensor,
    vm_true: torch.Tensor,
    va_true: torch.Tensor,
    keep_buses: Optional[Union[torch.Tensor, Sequence[int]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Computes per-snapshot voltage magnitude error (in pu) and angle error (in degrees),
    properly accounting for 2pi angle wrap-around, evaluated exclusively on non-slack buses.

    Returns:
      vm_err: (batch, len(keep_buses)) float64 tensor in pu
      va_err_deg: (batch, len(keep_buses)) float64 tensor in degrees
    """
    if keep_buses is None:
        keep_buses = torch.arange(1, 33, device=vm_pred.device)
    elif not isinstance(keep_buses, torch.Tensor):
        keep_buses = torch.as_tensor(keep_buses, device=vm_pred.device)

    # Magnitude error
    vm_err = vm_pred[:, keep_buses] - vm_true[:, keep_buses]

    # Angle difference with circular phase wrapping
    va_diff = torch.atan2(
        torch.sin(va_pred[:, keep_buses] - va_true[:, keep_buses]),
        torch.cos(va_pred[:, keep_buses] - va_true[:, keep_buses]),
    )
    va_err_deg = va_diff * (180.0 / np.pi)

    return vm_err, va_err_deg
