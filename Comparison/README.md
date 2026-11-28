# Cross-Architectural Comparison & Benchmark Suite

This module provides the unified benchmarking, profiling, and synthesis engine comparing all three deep state estimation architectures: the **Multi-Layer Perceptron (MLP)**, the **Edge-Attributed Graph Convolutional Network (GNN)**, and the **Temporal Convolutional Network (TCN)**.

---

## Directory & File Roles

| File / Folder | Role & Description |
| :--- | :--- |
| `compare_all_models.py` | **Cross-Model Benchmark & Profiling Tool**: Unified orchestration script that: (1) Parses benchmark results across `MLP/`, `GNN/`, and `TCN/`, (2) Evaluates Clean vs. Augmented performance across 10%–50% dropouts, (3) Analyzes the $\lambda$ physics regularizer sensitivity, (4) Benchmarks real-time hardware latency (CUDA and CPU) and parameter efficiency, and (5) Exports summary tables (`tables/`) and multi-panel publication figures (`plots/`). |
| `tables/` | **Standardized Benchmark Data**: Contains generated CSV summaries: `best_models_comparison.csv` (comprehensive performance comparison across all architectures and dropouts) and `hardware_profiles.csv` (total parameters, trainable parameters, checkpoint sizes, and CUDA latency). |
| `plots/` | **High-Resolution Comparative Figures**: Stores publication-quality comparison graphics: `vm_rmse_vs_coverage_comparison.png`, `va_rmse_deg_vs_coverage_comparison.png`, `lambda_sensitivity_cross_model.png`, `master_comparative_dashboard.png`, `robustness_degradation_slopes.png`, and `pareto_accuracy_vs_physics.png`. |

---

## Core Benchmark Findings Summary

1. **Top Raw Accuracy (Static Regime)**:
   * **Physics-Informed MLP ($\lambda = 0.1$, Augmented)** delivers the lowest voltage magnitude error ($|V|\text{ RMSE} = 0.000516\text{ pu}$ or $0.0516\%$) and angle error ($0.0137^\circ$) with the fastest execution ($0.223\text{ ms}$).
2. **Top Telemetry Blackout Resilience (Dynamic Regime)**:
   * **Temporal Convolutional Network (TCN, $\lambda = 0.01$)** bounds voltage degradation to merely **$+25.7\%$** when subjected to sudden 50% sensor loss, whereas the memoryless MLP error spikes by **$+164.0\%$**.
3. **Top Topological Adaptability**:
   * **Edge-Attributed GCN** explicitly models branch admittance ($G_{ij}, B_{ij}$). Although constrained by the 17-hop radial feeder horizon ($|V|\text{ RMSE} = 0.001295\text{ pu}$), it is the only model inherently structured for zero-shot transfer under grid line switching and topology reconfiguration.
4. **Hardware Latency (Real-Time 50 Hz IEEE C37.118 Compliance)**:
   * **MLP**: $0.223\text{ ms}$ | **GNN**: $2.262\text{ ms}$ | **TCN**: $2.199\text{ ms}$. All models execute far below the $20\text{ ms}$ real-time ceiling.

---

## How to Run

To run the complete cross-architectural evaluation and regenerate all summary tables and comparison plots:

```bash
python compare_all_models.py --root ../
```

To skip PyTorch CUDA latency profiling and only regenerate statistical tables and plots:
```bash
python compare_all_models.py --root ../ --no_latency
```
