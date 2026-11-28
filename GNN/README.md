# Edge-Attributed Graph Convolutional Network (GNN)

This module implements the **Physics-Informed Edge-Attributed Graph Convolutional Network (`EdgeAttributedGCN`)** for state estimation on the IEEE 33-bus radial distribution grid. Unlike flat architectures, the GNN explicitly encodes the grid's electrical connectivity and branch admittance properties into graph message-passing layers.

---

## Directory & File Roles

| File / Folder | Role & Description |
| :--- | :--- |
| `models.py` | **GNN Architecture Definition**: Implements `EdgeAttributedConv` (custom PyTorch Geometric message-passing layer with edge conditioning and symmetric degree normalization) and `EdgeAttributedGCN` (3-layer graph network with skip connections, residual updates, and $\tanh$ activations). |
| `graph_data.py` | **Graph Topology & Data Pipeline**: Builds the physical line graph (33 nodes, 32 distribution lines $\to$ 97 directed edges with self-loops). Extracts 4D edge attributes $\mathbf{e}_{ij} = [G_{ij}, B_{ij}, \|Y_{ij}\|, \angle Y_{ij}]$ directly from the complex $\mathbf{Y}_{\text{bus}}$ matrix. Implements the 6D node feature constructor and `IEEE33GraphDataset`. |
| `physics.py` | **Double-Precision AC Power Flow Physics**: Implements complex-number AC power residual calculations $\mathbf{S}_{\text{calc}} = \hat{\mathbf{V}} \odot (\hat{\mathbf{V}}\mathbf{Y}_{\text{bus}}^T)^*$, de-standardization functions, and physical slack bus restoration ($|V|_0 = 1.0\text{ pu}, \theta_0 = 0.0\text{ rad}$). |
| `train_gnn.py` | **Unified Training & Sweep Engine**: Orchestrates Clean and Augmented training regimes across all 9 penalty values: $\lambda \in [0.0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0, 1000.0]$. Implements the 20% epoch warm-up schedule, validates each epoch, preserves the best models, and evaluates test benchmarks across 10% to 50% PMU dropouts. |
| `evaluate_gnn.py` | **Standalone Model Evaluator**: CLI evaluation tool to test any saved GNN checkpoint against specific test files (`test_drop10.npz` to `test_drop50.npz`), outputting per-bus voltage RMSE, angle RMSE, and maximum bus error. |
| `compare_models.py` | **Single-Snapshot Visualizer**: Compares GNN predictions against ground truth and baseline MLP on individual grid snapshots, highlighting active vs. dropped PMU channels. |
| `plot_gnn_results.py` | **Publication Plotting Tool**: Plots GNN performance curves across dropout levels and visualizes the $\lambda$ sensitivity Pareto frontier. |
| `check_gnn_contract.py` | **Automated Contract Test Suite**: 6-check unit test suite validating scaler isolation, graph topology (33 nodes, 97 edges), double-precision types, neutral zeroing under 50% PMU dropout, model forward-pass output shapes, and physics residual backpropagation. |
| `plots/` | **Visualization Figures**: Stores generated GNN evaluation curves: `gnn_rmse_vs_coverage.png` and `lambda_sensitivity.png`. |
| `results_augmented/` | **Augmented Model Artifacts**: Stores trained model weights (`model_Augmented_lam*_seed42.pt`, `best_model_Augmented_seed42.pt`) and the evaluation benchmark table (`benchmark_augmented.csv`). |
| `results_clean/` | **Clean Model Artifacts**: Stores trained model weights (`model_Clean_lam*_seed42.pt`, `best_model_Clean_seed42.pt`) and benchmark metrics under out-of-distribution blackouts (`benchmark_clean.csv`). |

---

## Model Specifications

* **Graph Representation**:
  * **Nodes ($N = 33$)**: 6 features per bus: $[|V|_{\text{pmu}}, \theta_{\text{pmu}}, P_{\text{pseudo}}, Q_{\text{pseudo}}, \text{has\_pmu}, \text{pmu\_mask}]$.
  * **Edges ($E = 97$)**: 64 bidirectional directed edges representing the 32 physical distribution lines, plus 33 self-loops from the diagonal of $\mathbf{Y}_{\text{bus}}$.
  * **Edge Attributes (4D)**: Conductance $G_{ij}$, susceptance $B_{ij}$, magnitude $|Y_{ij}|$, and angle $\angle Y_{ij}$.
* **Hidden Layers**: $3$ message-passing layers with 128 channels each and $\tanh$ activations.
* **Output Dimension**: 2 outputs per node ($|V|_i$ and $\theta_i$), totaling 66 values across all 33 buses.
* **Parameters**: $86{,}402$ parameters ($0.67\text{ MB}$).
* **Execution Precision**: Strictly `torch.float64` / `torch.complex128`.

---

## How to Run

1. **Verify Graph Contract**:
   ```bash
   python check_gnn_contract.py --data ../MLP/data/ieee33
   ```
2. **Train Clean Baseline & Physics Models**:
   ```bash
   python train_gnn.py --data ../MLP/data/ieee33 --epochs 100 --bs 64 --lr 1e-3 --out results_clean
   ```
3. **Train Augmented Models (Dropout Robustness)**:
   ```bash
   python train_gnn.py --data ../MLP/data/ieee33 --epochs 100 --bs 64 --lr 1e-3 --aug --out results_augmented
   ```
4. **Evaluate Saved Checkpoint**:
   ```bash
   python evaluate_gnn.py --checkpoint results_augmented/best_model_Augmented_seed42.pt --data ../MLP/data/ieee33
   ```
5. **Plot GNN Benchmark Curves**:
   ```bash
   python plot_gnn_results.py
   ```
