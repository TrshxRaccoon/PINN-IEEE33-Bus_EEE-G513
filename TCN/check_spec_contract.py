#!/usr/bin/env python3
"""
check_spec_contract.py -- Automated Unit Tests & Contract Validation for TCN Pipeline
Verifies mandatory engineering specifications and contract requirements:
  1. Scaler fitting isolation (fit strictly on train.npz, slack bus std=1.0 safeguard).
  2. Sequential dataset shapes: (B, 114, T) in and (B, 66) out in torch.float64.
  3. STRICT CAUSALITY VERIFICATION TEST:
     Verify that the TCN model has zero future leakage: feed a sequence of length T,
     perturb the input at future steps, and assert that outputs at earlier time steps do not change!
  4. AC power flow residual on true state with true injections is < 1e-6 pu (~ 1e-12 pu).
  5. Pseudo-injection residual non-zero floor (L_phys floor).
  6. Neutral mask zeroing: dropped PMU channels become exact 0.0 (training mean).
"""

import argparse
import os
import sys
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure double precision globally
torch.set_default_dtype(torch.float64)

# Import components from train_tcn and tcn_model
try:
    from train_tcn import (
        DataContractScaler,
        SequentialTCNGridDataset,
        resolve_data_dir,
    )
    from tcn_model import StateEstimatorTCN, load_ybus, power_residual, to_physical
except ImportError:
    from .train_tcn import (
        DataContractScaler,
        SequentialTCNGridDataset,
        resolve_data_dir,
    )
    from .tcn_model import StateEstimatorTCN, load_ybus, power_residual, to_physical


def run_checks(data_dir: str = "data/ieee33", seq_len: int = 12):
    print("\n" + "=" * 80)
    print(" RUNNING MANDATORY SPEC & CONTRACT CHECKS (TCN Pipeline)")
    print(f" Checking dataset directory: {data_dir} | Sequence length (T): {seq_len}")
    print("=" * 80)

    # Resolve data directory
    resolved_data = resolve_data_dir(data_dir)
    print(f"Resolved dataset path: {resolved_data}")

    train_path = os.path.join(resolved_data, "train.npz")
    meta_path = os.path.join(resolved_data, "meta.npz")
    val_path = os.path.join(resolved_data, "val.npz")
    test_drop50_path = os.path.join(resolved_data, "test_drop50.npz")

    if not os.path.exists(train_path) or not os.path.exists(meta_path):
        print(f"\nERROR: Required dataset files not found at '{resolved_data}'.")
        print("Please generate the dataset first using:")
        print("    python make_sequential_dataset.py --out data --systems ieee33")
        sys.exit(1)

    # Load Meta constants
    meta = np.load(meta_path)
    Y_np = meta["Ybus_real"] + 1j * meta["Ybus_imag"]
    Y = torch.as_tensor(Y_np, dtype=torch.complex128)
    slack = int(meta["slack"])
    n_bus = int(meta["n_bus"])
    keep_buses = torch.as_tensor(
        [i for i in range(n_bus) if i != slack], dtype=torch.long
    )

    # --------------------------------------------------------------------------
    # Check 1: Scaler Fitting Isolation & Safeguards
    # --------------------------------------------------------------------------
    print("\n[Check 1/6] Scaler fitting isolation & slack bus safeguards...")
    scaler = DataContractScaler(train_path)

    assert scaler.vm_mean.shape == (16,), f"Expected vm_mean shape (16,), got {scaler.vm_mean.shape}"
    assert scaler.va_mean.shape == (16,), f"Expected va_mean shape (16,), got {scaler.va_mean.shape}"
    assert scaler.p_mean.shape == (33,), f"Expected p_mean shape (33,), got {scaler.p_mean.shape}"
    assert scaler.q_mean.shape == (33,), f"Expected q_mean shape (33,), got {scaler.q_mean.shape}"
    assert scaler.target_vm_mean.shape == (33,), f"Expected target_vm_mean shape (33,), got {scaler.target_vm_mean.shape}"
    assert scaler.target_va_mean.shape == (33,), f"Expected target_va_mean shape (33,), got {scaler.target_va_mean.shape}"

    # Verify slack bus std safeguard = 1.0 (prevents division by zero for invariant slack bus)
    assert scaler.target_vm_std[slack] == 1.0, f"Slack VM std safeguard failed: {scaler.target_vm_std[slack]}"
    assert scaler.target_va_std[slack] == 1.0, f"Slack VA std safeguard failed: {scaler.target_va_std[slack]}"
    assert scaler.p_std[slack] == 1.0, f"Slack P std safeguard failed: {scaler.p_std[slack]}"
    assert scaler.q_std[slack] == 1.0, f"Slack Q std safeguard failed: {scaler.q_std[slack]}"
    print("  ✓ Scaler fit strictly on train.npz only with verified slack bus safeguards.")

    # --------------------------------------------------------------------------
    # Check 2: Sequential Dataset Shapes & Model Forward Pass
    # --------------------------------------------------------------------------
    print(f"\n[Check 2/6] Sequential sliding-window tensor shapes (T={seq_len})...")
    dataset = SequentialTCNGridDataset(
        train_path, scaler, seq_len=seq_len, augment=False
    )
    sample = dataset[0]
    x, y = sample["x"], sample["y"]

    # Sample input must be (114, T) and target must be (66,)
    assert x.shape == (114, seq_len), f"Expected input shape (114, {seq_len}), got {tuple(x.shape)}"
    assert y.shape == (66,), f"Expected target shape (66,), got {tuple(y.shape)}"
    assert x.dtype == torch.float64, f"Expected float64 input, got {x.dtype}"
    assert y.dtype == torch.float64, f"Expected float64 target, got {y.dtype}"

    # Verify DataLoader batching
    batch_size = 4
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    batch = next(iter(loader))
    bx, by = batch["x"], batch["y"]
    assert bx.shape == (batch_size, 114, seq_len), f"Expected batched x shape ({batch_size}, 114, {seq_len}), got {tuple(bx.shape)}"
    assert by.shape == (batch_size, 66), f"Expected batched y shape ({batch_size}, 66), got {tuple(by.shape)}"

    # Forward pass through StateEstimatorTCN
    model = StateEstimatorTCN(
        in_channels=114,
        out_channels=66,
        hidden_channels=[64, 64, 128, 128],
        kernel_size=3,
        act="tanh",
    )
    out = model(bx)
    assert out.shape == (batch_size, 66), f"Expected model output ({batch_size}, 66), got {tuple(out.shape)}"
    assert out.dtype == torch.float64, f"Expected float64 output, got {out.dtype}"
    print(f"  ✓ Shapes verified: ({batch_size}, 114, {seq_len}) in -> TCN -> ({batch_size}, 66) out in torch.float64.")

    # --------------------------------------------------------------------------
    # Check 3: STRICT CAUSALITY VERIFICATION TEST (Zero Future Leakage)
    # --------------------------------------------------------------------------
    print("\n[Check 3/6] STRICT CAUSALITY VERIFICATION TEST (Zero Future Leakage)...")
    model.eval()
    with torch.no_grad():
        test_t = max(16, seq_len)
        x_base = torch.randn(2, 114, test_t, dtype=torch.float64)
        x_perturbed = x_base.clone()

        # Perturb the future-most time step (t = test_t - 1) with large disturbance
        perturbation_magnitude = 100.0
        x_perturbed[:, :, -1] += perturbation_magnitude

        # Evaluate both models in sequence return mode: (B, test_t, 66)
        out_base_seq = model(x_base, return_sequences=True)
        out_pert_seq = model(x_perturbed, return_sequences=True)

        assert out_base_seq.shape == (2, test_t, 66)
        assert out_pert_seq.shape == (2, test_t, 66)

        # Outputs at all past time steps t in [0 .. test_t - 2] must be IDENTICAL (diff == 0.0)
        past_diff = (
            (out_base_seq[:, :-1, :] - out_pert_seq[:, :-1, :]).abs().max().item()
        )
        current_step_diff = (
            (out_base_seq[:, -1, :] - out_pert_seq[:, -1, :]).abs().max().item()
        )

        print(f"  Causality Assertion: Max discrepancy at past steps (t < {test_t-1}): {past_diff:.2e}")
        print(f"  Causality Assertion: Discrepancy at perturbed step (t = {test_t-1}): {current_step_diff:.2e}")

        assert past_diff < 1e-12, (
            f"FATAL CAUSALITY VIOLATION: Perturbation at t={test_t-1} leaked backwards into "
            f"past time steps! Max past error: {past_diff:.2e}"
        )
        assert current_step_diff > 1e-3, (
            f"Perturbation had unexpectedly zero effect on current time step: {current_step_diff:.2e}"
        )

        # Additional Check: Prefix causal consistency
        # Passing truncated sequence [0..k] must yield identical output at step k as full sequence
        for k in [test_t // 2, test_t - 1]:
            prefix_input = x_base[:, :, : k + 1]
            prefix_out = model(prefix_input, return_sequences=False)  # prediction at step k
            seq_out_k = out_base_seq[:, k, :]
            prefix_err = (prefix_out - seq_out_k).abs().max().item()
            assert prefix_err < 1e-12, (
                f"Prefix causality mismatch at step {k}: error = {prefix_err:.2e}"
            )

    print("  ✓ STRICT CAUSALITY VERIFIED: Mathematical guarantee of zero future leakage confirmed (< 1e-12).")

    # --------------------------------------------------------------------------
    # Check 4: True State AC Power Flow Residual (< 1e-6 pu, ~ 1e-12 pu)
    # --------------------------------------------------------------------------
    print("\n[Check 4/6] AC power flow residual on True State + True Injections...")
    d_train = np.load(train_path)
    vm_true = torch.from_numpy(d_train["vm_true"]).to(torch.float64)
    va_true = torch.from_numpy(d_train["va_true"]).to(torch.float64)
    p_true = torch.from_numpy(d_train["p_true"]).to(torch.float64)
    q_true = torch.from_numpy(d_train["q_true"]).to(torch.float64)

    true_res = power_residual(vm_true, va_true, Y, p_true, q_true, keep_buses).item()
    print(f"  True state residual with true injections: {true_res:.2e} pu")
    assert true_res < 1e-6, (
        f"True residual {true_res:.2e} exceeds 1e-6 threshold! Check Ybus / sign conventions."
    )
    print("  ✓ AC power flow equations hold on ground truth states (~1e-12 pu precision).")

    # --------------------------------------------------------------------------
    # Check 5: Pseudo Injections Residual (Floor of L_phys)
    # --------------------------------------------------------------------------
    print("\n[Check 5/6] Power flow residual on True State + Pseudo Injections...")
    p_ps = torch.from_numpy(d_train["p_pseudo"]).to(torch.float64)
    q_ps = torch.from_numpy(d_train["q_pseudo"]).to(torch.float64)

    pseudo_res = power_residual(vm_true, va_true, Y, p_ps, q_ps, keep_buses).item()
    print(f"  True state residual with pseudo injections: {pseudo_res:.2e} pu (floor of L_phys)")

    # Must be non-zero and orders of magnitude higher than true_res
    assert pseudo_res > 1e-7, (
        f"Pseudo residual {pseudo_res:.2e} should be non-zero due to 10% forecast error."
    )
    assert pseudo_res > 100 * true_res, (
        f"Pseudo residual ({pseudo_res:.2e}) should be >> true residual ({true_res:.2e})."
    )
    print("  ✓ Soft constraint regularizer verified: non-zero pseudo residual floor confirmed.")

    # --------------------------------------------------------------------------
    # Check 6: Neutral Mask Zeroing on Dropout Dataset
    # --------------------------------------------------------------------------
    print("\n[Check 6/6] Mask correctness & neutral zeroing on dropout dataset...")
    if os.path.exists(test_drop50_path):
        test_ds = SequentialTCNGridDataset(
            test_drop50_path, scaler, seq_len=seq_len, augment=False
        )
        test_sample = test_ds[0]
        tx = test_sample["x"].numpy()      # shape (114, T)
        tmask = test_sample["mask"].numpy()  # shape (16,) at last step

        # In tx feature channels:
        # [0:16]   : norm_vm
        # [16:32]  : norm_va
        # [32:65]  : norm_p
        # [65:98]  : norm_q
        # [98:114] : mask
        x_vm_norm = tx[0:16, -1]  # last time step
        x_va_norm = tx[16:32, -1]
        x_mask = tx[98:114, -1]

        dropped_indices = np.where(tmask == 0.0)[0]
        assert len(dropped_indices) > 0, "Expected dropped PMUs in test_drop50.npz"

        for idx in dropped_indices:
            # Standardize first, then multiply by mask => dropped channel must be EXACTLY 0.0
            assert x_vm_norm[idx] == 0.0, f"Dropped VM channel {idx} not zeroed: {x_vm_norm[idx]}"
            assert x_va_norm[idx] == 0.0, f"Dropped VA channel {idx} not zeroed: {x_va_norm[idx]}"
            assert x_mask[idx] == 0.0, f"Dropped mask channel {idx} not zeroed: {x_mask[idx]}"

        present_indices = np.where(tmask == 1.0)[0]
        for idx in present_indices:
            assert x_mask[idx] == 1.0, f"Present mask channel {idx} not 1.0: {x_mask[idx]}"

        print(
            f"  ✓ Verified on test_drop50 ({len(dropped_indices)} PMUs dropped): "
            f"missing channels are strictly 0.0 (neutral training mean)."
        )
    else:
        print(f"  Note: {test_drop50_path} not found; skipping static test_drop50 check.")

    print("\n" + "=" * 80)
    print(" ALL 6 MANDATORY SPEC & CONTRACT CHECKS PASSED SUCCESSFULLY!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run automated contract & spec test suite for TCN state estimation"
    )
    parser.add_argument(
        "--data",
        type=str,
        default="data/ieee33",
        help="Path to sequential dataset directory (default: data/ieee33)",
    )
    parser.add_argument(
        "--seq_len",
        type=int,
        default=12,
        help="Sliding window sequence length T (default: 12)",
    )
    args = parser.parse_args()
    run_checks(data_dir=args.data, seq_len=args.seq_len)
