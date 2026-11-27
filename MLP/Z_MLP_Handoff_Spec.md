# Handoff Specification: MLP State Estimators for the IEEE 33-Bus Feeder

Audience: an LLM (or engineer) with no prior context. Task: implement, train and evaluate two MLP state estimators (data-driven baseline and physics-informed) on an already-generated dataset. "MLE model" is read as "MLP model". Do not regenerate data.

---

## 1. Project context

**Research question.** Does adding the AC power-flow equations to the training loss as a soft residual reduce voltage magnitude and angle estimation error under random PMU dropout, compared with a purely data-driven estimator trained on the same data?

**Hypothesis.** Yes, and the benefit grows as PMU coverage shrinks.

**Definitions.**
- State: complex voltage at every bus, i.e. magnitude |V| (per-unit) and angle θ (radians, relative to the slack bus).
- PMU: sensor reporting |V| and θ at its bus. Only 16 of the 32 non-slack buses have one.
- PMU dropout: in each snapshot a random subset of those 16 PMUs delivers nothing. Unrelated to `nn.Dropout`.
- Snapshot: one solved AC power flow for one set of loads.
- Pseudo-measurement: noisy forecast of the real/reactive power injected at each bus (about 10% error).
- Soft residual: a penalty on how much the model's estimated state violates the power-flow equations.

**Two models to build, differing only in the loss:**
1. Baseline MLP: `L = L_data`
2. Physics-informed MLP: `L = L_data + λ · L_phys`

Same architecture, same data, same optimizer, same seed/initialization, same epochs.

---

## 2. Data contract (do not change)

Location: `data/ieee33/` (produced by `make_dataset.py`).

| File | Samples | Mask | Use |
|---|---|---|---|
| `train.npz` | 8000 | all ones | fitting |
| `val.npz` | 1000 | dropout per snapshot random in {10,20,30,40,50}% | λ and model selection |
| `test_drop10/20/30/40/50.npz` | 1000 each | one dropout level per file | final reporting |
| `meta.npz` | – | – | admittance matrix and constants |

Test files contain the **same 1000 snapshots** with identical noise; only the mask differs (paired comparison).

**Dimensions:** N = 33 buses, K = 16 PMUs, slack = bus index 0.

**Arrays in each split file** (n = number of samples):

| Key | Shape | Meaning |
|---|---|---|
| `pmu_vm`, `pmu_va` | (n,16) | PMU |V| (pu) and θ (rad), noisy, **missing channels zeroed** |
| `pmu_vm_full`, `pmu_va_full` | (n,16) | same readings **unmasked** (used only to re-mask) |
| `mask` | (n,16) | 1 = PMU present, 0 = missing |
| `p_pseudo`, `q_pseudo` | (n,33) | forecast injections (pu), slack entry = 0 |
| `vm_true`, `va_true` | (n,33) | **labels**: true |V| (pu), θ (rad) |
| `p_true`, `q_true` | (n,33) | true injections, diagnostics only |
| `drop_frac`, `hour` | (n,) | nominal dropout fraction, hour of day |

`meta.npz`: `Ybus_real`, `Ybus_imag` (33×33 each), `slack`, `pmu_idx`, `n_bus`, `base_mva`, `sig_vm`, `sig_va_deg`, `sig_pseudo`.

**Noise model used to create the data:** PMU |V| noise σ = 0.001 pu; PMU θ noise σ = 0.05°; pseudo-injection multiplicative error σ = 10%.

**Conventions (all must be respected):**
- Per-unit on a 1 MVA base.
- Injection sign: generation positive, so loads are negative.
- θ in radians in the files; report RMSE of θ in degrees.
- Dropout is exact per snapshot: round(p·16) PMUs removed (10% → 2, 20% → 3, 30% → 5, 40% → 6, 50% → 8; actual 12.5, 18.75, 31.25, 37.5, 50%).

**Leakage rules (critical):**
- `test_*.npz` and `val.npz` contain unmasked `*_full` arrays. When building inputs always apply `mask`. Feeding unmasked PMU data into a test evaluation invalidates the study.
- Never use `p_true`, `q_true`, `vm_true`, `va_true` as inputs.
- Fit scalers on `train.npz` only.

---

## 3. Physics the model must be informed by

**Network law.** With bus admittance matrix `Y = G + jB` (stored in `meta.npz`), injected current is `I = Y·V`, and complex power injected at bus i is

`S_i = V_i · conj( Σ_k Y_ik V_k )`, with `V_i = |V_i| e^{jθ_i}`.

Equivalent polar form:
- `P_i = |V_i| Σ_k |V_k| (G_ik cos θ_ik + B_ik sin θ_ik)`
- `Q_i = |V_i| Σ_k |V_k| (G_ik sin θ_ik − B_ik cos θ_ik)`, with `θ_ik = θ_i − θ_k`.

Meaning: the net power a bus injects equals what its voltage differences push through connected lines (Ohm's law plus Kirchhoff's current law). The slack bus (|V| = 1.0, θ = 0) is fixed and its injection is unknown, so it is excluded from the equations. Each of the other 32 buses supplies 2 equations, matching its 2 unknowns.

**Residual (the physics term).** For the model's estimated state:
1. `V̂ = |V̂|·exp(jθ̂)`
2. `S_calc = V̂ ⊙ conj(Y V̂)`
3. `r_P = Re(S_calc) − p_pseudo`, `r_Q = Im(S_calc) − q_pseudo`
4. `L_phys = mean over non-slack buses of (r_P² + r_Q²)`

Reference PyTorch implementation:

```python
def power_residual(vm, va, Y, p_ps, q_ps, keep):
    V = torch.polar(vm, va)                  # (B,N) complex
    S = V * torch.conj(V @ Y.T)              # V_i * conj(sum_k Y_ik V_k)
    rP, rQ = S.real - p_ps, S.imag - q_ps
    return (rP[:, keep]**2 + rQ[:, keep]**2).mean()
```

**Why it is soft.** The pseudo-injections are about 10% wrong, so `L_phys` is not zero at the true state. A hard equality would force the model to fit flawed forecasts. λ trades PMU data against physics.

**Scale warning.** Admittance entries are about 10²-10³ pu, so a 0.001 pu voltage error produces a power mismatch of order 0.1-1 pu. `L_phys` is numerically large relative to standardized `L_data`, so useful λ values are small (1e-4 to 1).

**Mandatory sanity checks:**
- With true voltages and `p_true`, `q_true`, the residual must be about 1e-8 pu or lower (verifies Y, signs, units).
- With true voltages and `p_pseudo`, `q_pseudo`, the residual is clearly nonzero (forecast error). Record this value: it is the floor of `L_phys`.

---

## 4. Input and output specification

**Input vector (114 features), exact order:**

| Block | Size | Processing |
|---|---|---|
| PMU |V| | 16 | z-score using train mean/std of `pmu_vm_full`, **then multiply by mask** |
| PMU θ | 16 | z-score using train mean/std of `pmu_va_full`, **then multiply by mask** |
| `p_pseudo` | 33 | z-score |
| `q_pseudo` | 33 | z-score |
| `mask` | 16 | raw 0/1 |

Standardize first, mask second. A missing channel then equals 0, the training mean, which is neutral. Masking raw values first would inject fake 0-volt readings.

**Output vector (66):** `[|V| at 33 buses, θ at 33 buses]`, z-scored with train statistics of `[vm_true, va_true]`.
- Slack columns have zero variance: set their std to 1, exclude them from the loss, and overwrite their physical values with the known constants (`mean` of the training labels, i.e. 1.0 and 0.0) after de-standardizing.
- Convert to physical units before computing `L_phys`: `y = out·std + mean`, then split into `vm`, `va`.

---

## 5. Architecture

Use identical architecture for both models.

- Input 114 → hidden → hidden → output 66.
- Recommended default: 2 hidden layers × 256 units, **smooth activation (tanh or SiLU)**, linear output layer.
- Smooth activations are preferred because gradients of the physics term pass through the network twice-composed with trigonometric functions.
- No dropout layers by default. Optional small weight decay (≤1e-5).
- `torch.float64` for all tensors, including model weights: `L_phys` subtracts large nearly-equal values.
- Y as `torch.complex128` built from `Ybus_real`, `Ybus_imag`.

---

## 6. Training procedure

**Common:**
- Adam, lr 1e-3, batch 128, up to 200 epochs.
- `ReduceLROnPlateau` on validation score (factor 0.5, patience 10), gradient-norm clipping at 1.0.
- Keep the checkpoint with the best validation score.
- Model-selection score on `val.npz`: `RMSE_V/0.01 + RMSE_θ(deg)/0.1` (RMSE over non-slack buses).
- 3-5 seeds; report mean ± std.

**Baseline MLP:** `loss = mean((ŷ − y)²)` over the 64 non-slack standardized outputs.

**Physics-informed MLP:**
```
out   = model(x)
vm,va = to_physical(out)                 # de-standardize, overwrite slack
loss  = L_data + lam * power_residual(vm, va, Y, p_pseudo, q_pseudo, keep)
```
- λ warm-up: λ = 0 for the first ~10% of epochs, then ramp linearly to its target over the next ~10%. An untrained network gives a huge, unstable residual.
- λ sweep: {1e-4, 1e-3, 1e-2, 1e-1, 1}; one model per value; choose by validation score; report the chosen model on test.
- Optional extension: adaptive λ via dual ascent, `λ ← clip(λ + η·(L_data − d), 0, λ_max)`.

**Two training conditions (run both, report both):**
1. **Clean:** train on `train.npz` with every PMU present (mask all ones), as specified.
2. **Augmented:** every training step, draw a fresh random dropout per sample, level ~ U(0, 0.5), `round(level·16)` PMUs removed, applied to the unmasked stored readings before standardizing and masking. Clean training creates a train/test distribution shift, which makes the baseline look weaker than it is; the augmented condition is the fair comparison.

---

## 7. Evaluation

For each of `test_drop10..50`, with the stored `mask` applied:
- RMSE of |V| in pu and RMSE of θ in degrees, over the 32 non-slack buses.
- Also report the physics residual `L_phys` on the test predictions for both models.
- Optional: RMSE restricted to buses without a PMU, and bus-by-bus error.

Report a table and a plot of RMSE (|V| and θ) vs PMU coverage (100 − dropout %), both models, both training conditions, with seed error bars. Include the closed-form linear least-squares model as a reference curve.

---

## 8. Reference results from the linear version (benchmark for sanity)

Single-layer linear model, 3000 training samples, 30 epochs, 1 seed (indicative only):

| Condition | Level | Baseline |V| RMSE | Physics |V| RMSE | θ RMSE (both) |
|---|---|---|---|---|
| Clean | 10% | 0.00117 pu | 0.00108 pu | 0.0151° |
| Clean | 50% | 0.00350 pu | 0.00254 pu | 0.0297° |
| Augmented | 10% | 0.00075 pu | 0.00075 pu | 0.0130° |
| Augmented | 50% | 0.00082 pu | 0.00082 pu | 0.0162° |

Expectations for the MLP: it should match or beat the linear baseline at every level. If it does not, suspect a bug. Absolute errors are small (about 1e-3 pu) because pseudo-injections at all buses already carry much of the state information. A physics benefit may only be visible at higher dropout, with higher `--sig_pseudo` or fewer PMUs; if no benefit appears, report that honestly together with the λ values tried.

---

## 9. Common pitfalls

1. Feeding unmasked test PMU data (leakage).
2. Masking before standardizing.
3. Fitting scalers on val/test.
4. Using float32 for the residual.
5. Forgetting to exclude the slack bus from `L_data` and `L_phys`.
6. Mixing units: the residual uses pu and radians; RMSE of θ is reported in degrees.
7. Computing the residual on standardized outputs instead of physical ones.
8. Using the wrong sign for injections, or using `p_true` as the physics target (use `p_pseudo`).
9. Selecting λ on the test set.
10. Comparing models trained with different seeds, epochs or optimizers.

---

## 10. Deliverables

1. `train_mlp.py` with CLI: `--data`, `--hidden`, `--act`, `--epochs`, `--lr`, `--bs`, `--lams`, `--seeds`, `--aug`, `--out`.
2. Unit tests or assertions for: tensor shapes (114 in, 66 out), residual ≈ 1e-8 on true state with true injections, mask correctness, no train/val/test overlap in scaler fitting.
3. Results table (CSV) and RMSE-vs-coverage figure.
4. A short written conclusion: does the physics term help, by how much, at which dropout levels, and under which training condition.
