# Physics-Informed Edge-Attributed GCN: Operating, Training & Validation Guide
## IEEE 33-Bus Distribution Grid State Estimation under Progressive PMU Telemetry Dropout

---

### 1. Architectural Overview & System Design

This subproject implements an **Edge-Attributed Graph Convolutional Network (`EdgeAttributedGCN`)** designed for power system state estimation on the **IEEE 33-bus radial distribution feeder** (`case33bw`, $12.66\text{ kV}$, $10\text{ MVA}$ base).

#### Key Design Decisions:
1. **Physical Branch Awareness (Option B)**:
   * 33 buses form the graph nodes ($N=33$).
   * 32 physical distribution lines form 64 directed edges plus 33 self-loops ($E=97$ edges).
   * Edge attributes $\mathbf{e}_{ij} \in \mathbb{R}^4$ are extracted directly from the complex bus admittance matrix $\mathbf{Y}_{\text{bus}}$:
     $$\mathbf{e}_{ij} = \left[ G_{ij}, \; B_{ij}, \; |Y_{ij}|, \; \angle Y_{ij} \right]$$
     where $Y_{ij} = G_{ij} + j B_{ij}$.
2. **6-Dimensional Nodal Feature Representation**:
   * For each bus $i \in \{0, \dots, 32\}$, the input feature vector is $\mathbf{x}_i \in \mathbb{R}^6$:
     1. Standardized PMU $|V|_i$ ($0.0$ if unmeasured or dropped)
     2. Standardized PMU $\theta_i$ ($0.0$ if unmeasured or dropped)
     3. Standardized $P_{\text{pseudo}, i}$ (active power forecast)
     4. Standardized $Q_{\text{pseudo}, i}$ (reactive power forecast)
     5. `has_pmu` indicator ($1.0$ if PMU is installed at bus $i$, else $0.0$)
     6. `pmu_mask` indicator ($1.0$ if active, $0.0$ if dropped or uninstalled)
3. **Data Contract ("Standardize First, Mask Second")**:
   * Raw PMU measurements are standardized using unmasked training statistics:
     $$x_{\text{pmu}} = \left(\frac{V_{\text{raw}} - \mu_{\text{train}}}{\sigma_{\text{train}}}\right) \odot \text{mask}$$
   * When a PMU drops ($\text{mask}=0$), its feature becomes exactly $0.0$ (the training mean, neutral).
4. **Strict Double Precision**:
   * All tensors, network weights, and loss computations use `torch.float64` and `torch.complex128` to avoid numerical cancellation when computing AC power flows $\mathbf{S}_{\text{calc}} = \mathbf{V} \odot (\mathbf{V} \mathbf{Y}^T)^*$.
5. **Slack Bus Safeguards**:
   * Slack bus (Bus 0) is known a priori ($|V|=1.0\text{ pu}, \theta=0.0\text{ rad}$) and is excluded from both the supervised loss $\mathcal{L}_{\text{data}}$ and physics loss $\mathcal{L}_{\text{phys}}$.
6. **Expanded Lambda Penalty Sweep**:
   * Evaluates $\lambda \in [0.0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0, 1000.0]$ with a 20% epoch warm-up schedule.

---

### 2. File Organization Inside `ML_Project/GNN/`

```
ML_Project/GNN/
├── graph_data.py          # Graph topology builder, 4D edge attributes, DataContractScaler, IEEE33GraphDataset
├── models.py              # EdgeAttributedConv (MessagePassing) & EdgeAttributedGCN architecture
├── physics.py             # AC power-flow residual calculator, de-standardization, slack bus restoration
├── train_gnn.py           # Unified training, lambda sweep, and test evaluation orchestrator
├── evaluate_gnn.py        # Standalone evaluation tool for trained checkpoints
├── compare_models.py      # Interactive single-snapshot comparison tool (GNN vs. MLP vs. Ground Truth)
├── plot_gnn_results.py    # Publication plotting suite (300 DPI figures)
├── check_gnn_contract.py  # Automated 6-check unit test suite validating the contract
├── GNN_OPERATING_GUIDE.md # This comprehensive operating guide
└── README.md              # Quickstart guide
```

---

### 3. Step-by-Step Operating Instructions

All commands below should be executed from within the `d:\BITS Pilani Goa Campus\Academics\ML4EE\ML_Project\GNN` directory.

#### Step 1: Verify the Graph Contract (Pre-Flight Check)
Before training any model, run the automated 6-check contract test suite:
```powershell
python check_gnn_contract.py --data ../MLP/data/ieee33
```
* **What it does**: Verifies scaler isolation on `train.npz`, graph topology (33 nodes, 97 edges), double-precision types, neutral zeroing under 50% PMU dropout, forward pass shape, and AC power flow residual physics equations.
* **Expected runtime**: $< 0.5$ seconds.
* **Success condition**: All 6 checks pass with `ALL 6 MANDATORY GNN CONTRACT CHECKS PASSED SUCCESSFULLY`.

---

#### Step 2: Train the Clean Sweep (Stress Testing Telemetry Failure)
In this condition, the model is trained purely on 100% complete PMU data and tested under progressive dropouts (10% to 50%):
```powershell
python train_gnn.py --data ../MLP/data/ieee33 --epochs 100 --bs 64 --lr 1e-3 --out results_clean
```
* **What it does**:
  1. Sweeps all 9 penalty values: $\lambda \in [0.0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0, 1000.0]$.
  2. Applies the 20% epoch warm-up schedule (0–10% epochs: $\lambda=0$; 10–20% epochs: linear ramp; 20–100%: $\lambda$).
  3. Validates every epoch on `val.npz` and preserves the best checkpoint per $\lambda$.
  4. Saves all individual models: `results_clean/model_Clean_lam<LAM>_seed42.pt`.
  5. Saves the overall best model: `results_clean/best_model_Clean_seed42.pt`.
  6. Evaluates each model across all five test files (`test_drop10.npz` to `test_drop50.npz`).
  7. Automatically generates `results_clean/benchmark_clean.csv` and `results_clean/conclusion_clean.md`.

---

#### Step 3: Train the Augmented Sweep (Fair Dynamic Comparison)
In this condition, on-the-fly random PMU dropout $p \sim \mathcal{U}(0, 0.5)$ is drawn per sample during training to prevent train/test distribution shift:
```powershell
python train_gnn.py --data ../MLP/data/ieee33 --epochs 100 --bs 64 --lr 1e-3 --aug --out results_augmented
```
* **What it does**:
  1. Trains across all 9 penalty values under dynamic on-the-fly PMU channel dropout.
  2. Saves individual checkpoints: `results_augmented/model_Augmented_lam<LAM>_seed42.pt`.
  3. Saves the best checkpoint: `results_augmented/best_model_Augmented_seed42.pt`.
  4. Evaluates across all five test levels (`test_drop10.npz` to `test_drop50.npz`).
  5. Automatically generates `results_augmented/benchmark_augmented.csv` and `results_augmented/conclusion_augmented.md`.

---

#### Step 4: Standalone Checkpoint Evaluation & Validation
To evaluate any trained model checkpoint against the test suite or a specific dropout level:
```powershell
# Evaluate the best Augmented model across all dropout levels (10% to 50%)
python evaluate_gnn.py --checkpoint results_augmented/best_model_Augmented_seed42.pt --data ../MLP/data/ieee33

# Evaluate a specific Clean physics model under severe 50% PMU dropout
python evaluate_gnn.py --checkpoint results_clean/model_Clean_lam0.1_seed42.pt --drop 50
```
* **Output**: Formats a terminal table with Voltage Magnitude RMSE (pu), Voltage Angle RMSE (deg), and Physics Residual $\mathcal{L}_{\text{phys}}$ (pu) for each test condition.

---

#### Step 5: Generate Publication-Quality Plots
Once training is complete, generate high-resolution (300 DPI) publication figures:
```powershell
python plot_gnn_results.py
```
* **Generated Figures** (saved in `plots/`):
  1. `rmse_vs_coverage_clean.png`: $|V|$ and $\theta$ RMSE vs PMU Coverage (Clean Training) comparing Baseline GCN ($\lambda=0$), Physics-Informed GCN ($\lambda > 0$), and MLP benchmarks.
  2. `rmse_vs_coverage_augmented.png`: $|V|$ and $\theta$ RMSE vs PMU Coverage (Augmented Training).
  3. `lambda_sensitivity_sweep_clean.png`: Error and physics residual across the full $\lambda$ spectrum ($0$ to $1000$).
  4. `lambda_sensitivity_sweep_augmented.png`: Sensitivity under dynamic dropout.

---

#### Step 6: Run Interactive Single-Snapshot Comparisons (Demo)
Use the interactive demo tool to inspect individual test snapshots, examine active vs. dropped PMU telemetry channels, and view a bus-by-bus prediction table:
```powershell
# Compare Clean models on Snapshot #0 under 50% PMU dropout
python compare_models.py --condition Clean --drop 50 --snapshot 0

# Compare Augmented models and generate a 2-panel visual comparison graph
python compare_models.py --condition Augmented --drop 50 --snapshot 0 --plot

# Inspect a specific GNN checkpoint on Snapshot #5 with 30% dropout
python compare_models.py --gnn_ckpt results_augmented/best_model_Augmented_seed42.pt --drop 30 --snapshot 5
```
* **Output**:
  * Displays active vs. dropped PMU channels across all 16 candidate buses.
  * Formats a complete 33-bus table showing Ground Truth $|V|$, GNN $|V|$, $|V|$ Error, Ground Truth $\theta$, GNN $\theta$, $\theta$ Error, and MLP predictions.
  * Saves `snapshot_<ID>_drop<DROP>_<CONDITION>_comparison.png` when `--plot` is supplied.

---

### 4. Technical Reference Summary

| Parameter / Metric | Value / Specification | Rationale |
| :--- | :--- | :--- |
| **Grid Model** | IEEE 33-Bus Feeder (`case33bw`) | Standard benchmark distribution network with high $R/X$ ratio |
| **Architecture** | `EdgeAttributedGCN` | Message passing with edge-conditioned admittance features |
| **Node Features** | 6 dimensions per bus | Standardized $[\|V\|_{\text{pmu}}, \theta_{\text{pmu}}, P_{\text{ps}}, Q_{\text{ps}}, \text{has\_pmu}, \text{pmu\_mask}]$ |
| **Edge Features** | 4 dimensions per branch | Physical $Y_{\text{bus}}$ parameters $[G_{ij}, B_{ij}, \|Y_{ij}\|, \angle Y_{ij}]$ |
| **Edge Normalization** | Symmetric $\tilde{\mathbf{D}}^{-1/2} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-1/2}$ | Normalized spectral graph convolution |
| **Activation** | Hyperbolic Tangent ($\tanh$) | Continuous higher-order derivatives for AC power-flow backprop |
| **Arithmetic Precision**| `torch.float64` & `torch.complex128` | Prevents catastrophic cancellation in complex matrix multiplication |
| **Slack Bus** | Bus 0 ($1.0\text{ pu}, 0.0\text{ rad}$) | Excluded from training loss; restored during de-standardization |
| **Expanded $\lambda$ Grid**| $[0.0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10, 100, 1000]$ | Full sensitivity spectrum including strong physical penalty bounds |
| **Warm-Up Schedule** | 20% epoch linear ramp | Stabilizes early gradient updates before applying physical constraints |
