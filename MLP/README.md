# Multi-Layer Perceptron (MLP) State Estimator

This module implements the **Physics-Informed Multi-Layer Perceptron (PI-MLP)** state estimator for the IEEE 33-bus radial distribution grid. The MLP maps global snapshot telemetry directly to full grid states in a single, ultra-fast forward pass ($0.223\text{ ms}$ on CUDA).

---

## Directory & File Roles

| File / Folder | Role & Description |
| :--- | :--- |
| `train_mlp.py` | **Core Training Engine**: Implements the `StateEstimatorMLP` model in `torch.float64` double precision with $\tanh$ activations. Orchestrates both Clean Training (fixed complete telemetry) and Dropout-Augmented Training (on-the-fly random $0\%-50\%$ masking). Performs the $\lambda$ physics regularizer sweep across $[0.0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0, 1000.0]$ with a 20% epoch warm-up schedule. Evaluates checkpoints across test files and saves benchmarks. |
| `check_spec_contract.py` | **Automated Contract Test Suite**: Runs 5 mandatory verification checks: (1) Scaler statistics fitted strictly on `train.npz`, (2) Exact 114 input / 66 output shapes, (3) Neutral zeroing ($0.0$ value) for dropped PMUs under "standardize first, mask second", (4) Double precision dtype enforcement (`float64`, `complex128`), and (5) Non-zero AC power-flow gradients during backpropagation. |
| `plot_results.py` | **Publication Plotting Tool**: Generates publication-grade figures (300 DPI) plotting $|V|$ and $\theta$ RMSE as a function of PMU measurement coverage for both Clean and Augmented training regimes. |
| `data/ieee33/` | **Grid Dataset**: Contains the pre-generated power flow snapshots for the IEEE 33-bus system: `train.npz` (8,000 snapshots), `val.npz` (1,000 snapshots), `test_drop10.npz` through `test_drop50.npz` (1,000 identical paired snapshots with 10% to 50% PMU dropouts), and `meta.npz` ($\mathbf{Y}_{\text{bus}}$, slack bus index, PMU positions). |
| `demo/` | **Interactive CLI Demo Suite**: Contains `compare_models.py` to inspect single snapshots side-by-side (Ground Truth vs. Baseline MLP vs. Physics-Informed MLP), displaying real-time telemetry dropout flags and bus voltage errors. Also includes sample visualization graphs. |
| `plots/` | **Visualization Figures**: Stores generated PNG curves: `rmse_vs_coverage_augmented.png` and `rmse_vs_coverage_clean.png`. |
| `results_augmented/` | **Augmented Model Artifacts**: Stores trained model weights (`model_Augmented_lam*_seed42.pt`, `best_model_Augmented_seed42.pt`) and the complete numerical benchmark evaluation across all dropouts (`benchmark_augmented.csv`). |
| `results_clean/` | **Clean Model Artifacts**: Stores trained model weights (`model_Clean_lam*_seed42.pt`, `best_model_Clean_seed42.pt`) and benchmark metrics under out-of-distribution communication blackouts (`benchmark_clean.csv`). |

---

## Model Specifications

* **Input Dimension**: $114$ features
  * Block 1 (16): Standardized PMU $|V| \odot \text{mask}$
  * Block 2 (16): Standardized PMU $\theta \odot \text{mask}$
  * Block 3 (33): Standardized $P_{\text{pseudo}}$ active power injections
  * Block 4 (33): Standardized $Q_{\text{pseudo}}$ reactive power injections
  * Block 5 (16): Binary PMU availability mask ($1.0 = \text{online}, 0.0 = \text{dropped}$)
* **Hidden Layers**: $2$ dense layers $\times 256$ units with $\tanh$ activations.
* **Output Dimension**: $66$ outputs ($|V|_{0..32}$ and $\theta_{0..32}$ for all 33 buses).
* **Parameters**: $112{,}194$ trainable parameters ($0.86\text{ MB}$).
* **Execution Precision**: Strictly `torch.float64` / `torch.complex128`.

---

## How to Run

1. **Verify Contract Compliance**:
   ```bash
   python check_spec_contract.py
   ```
2. **Train Clean Baseline & Physics Models**:
   ```bash
   python train_mlp.py --data data/ieee33 --epochs 100 --bs 64 --lr 1e-3 --out results_clean
   ```
3. **Train Augmented Models (Dropout Robustness)**:
   ```bash
   python train_mlp.py --data data/ieee33 --epochs 100 --bs 64 --lr 1e-3 --aug --out results_augmented
   ```
4. **Generate Plots**:
   ```bash
   python plot_results.py
   ```
5. **Run Single-Snapshot Interactive Inspection**:
   ```bash
   cd demo
   python compare_models.py --drop 50 --idx 0
   cd ..
   ```
