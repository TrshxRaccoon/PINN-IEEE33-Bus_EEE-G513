# Physics-Informed Deep State Estimation Under Sensor Dropout in Radial Distribution Networks
### Benchmarking MLP, Edge-Attributed GNN, and Temporal Convolutional Network (TCN) Trade-offs on the IEEE 33-Bus Feeder

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-torch__geometric-brightgreen.svg)](https://pyg.org/)
[![Pandapower](https://img.shields.io/badge/Grid%20Simulation-Pandapower-yellow.svg)](https://www.pandapower.org/)

---

## Executive Overview

Real-time monitoring of modern electric distribution grids is challenged by high penetration of Distributed Energy Resources (DERs), bidirectional power flows, and incomplete sensor telemetry. While high-voltage transmission networks are heavily metered, distribution systems suffer from low observability and severe communication packet loss. Traditional Weighted Least Squares (WLS) state estimators diverge under telemetry blackouts, whereas standard black-box neural networks degrade catastrophically and violate fundamental electrical circuit laws.

This repository implements a **Physics-Informed Deep State Estimation (PI-DSSE)** framework benchmarked on the standard **IEEE 33-bus radial distribution feeder** (`case33bw`, $12.66\text{ kV}$, $10\text{ MVA}$ base). We evaluate three distinct deep learning paradigms under progressive **$10\%$ to $50\%$ PMU communication blackouts** and **$\pm 10\%$ load forecast noise**:

1. **Global Spatial Mapping**: Multi-Layer Perceptron (**MLP**)
2. **Topological Graph Message-Passing**: Edge-Attributed Graph Convolutional Network (**GNN**)
3. **Temporal Dynamic Inertia**: Causal Dilated 1D Temporal Convolutional Network (**TCN**)

```
               Telemetry Input (Sparse PMUs + Pseudo-Injections + Blackout Mask)
                                              │
                    ┌─────────────────────────┼─────────────────────────┐
                    ▼                         ▼                         ▼
              ┌───────────┐             ┌───────────┐             ┌───────────┐
              │    MLP    │             │    GNN    │             │    TCN    │
              │  Global   │             │Topological│             │  Temporal │
              │  Dense    │             │Relational │             │  Inertia  │
              └─────┬─────┘             └─────┬─────┘             └─────┬─────┘
                    │                         │                         │
                    └─────────────────────────┼─────────────────────────┘
                                              │
                                              ▼
                        Predicted Full Grid State (|V|_i, θ_i) for all 33 buses
                                              │
                                              ▼
                    AC Physics Constraint: S_calc = V ⊙ (V · Y_bus^T)*
```

---

## Key Scientific Findings

* **Trend 1: Accuracy vs. Resilience Trade-Off**:
  * Under **Clean Training** (unprepared for sensor failure), a $50\%$ sensor blackout causes the MLP estimation error to spike by **$+164.0\%$**, whereas the TCN absorbs the failure with only **$+25.7\%$** error growth owing to its causal historical memory.
  * Under **Dropout-Augmented Training**, the MLP becomes the accuracy champion ($|V|\text{ RMSE} = 0.000515\text{ pu}$, $0.223\text{ ms}$ latency) by leveraging its all-to-all receptive field to impute missing channels from surviving ones.
* **Trend 2: The Topological Horizon of GNNs**:
  * The 3-layer GNN is constrained by localized message passing across the 17-hop radial feeder, capping its accuracy at $0.001295\text{ pu}$ ($\sim 2.5\times$ worse than the MLP). However, it is the only architecture inherently capable of zero-shot transfer under dynamic feeder reconfiguration.
* **Trend 3: The Physics Regularization Pareto Cliff**:
  * Moderate AC power-flow penalties ($\lambda \in [0.001, 0.1]$) provide valuable physical guardrails without degrading empirical accuracy.
  * Over-regularization ($\lambda \ge 10.0$) triggers a sharp performance collapse because the network is forced to overfit the $\pm 10\%$ forecast noise present in pseudo-measurements.
* **Real-Time 50 Hz Compliance**:
  * All three architectures achieve CUDA inference latencies under $2.5\text{ ms}$ (MLP: $0.223\text{ ms}$, TCN: $2.199\text{ ms}$, GNN: $2.262\text{ ms}$), well within the $20\text{ ms}$ streaming PMU window.

---

## Quantitative Benchmark Summary

### Performance at 50% PMU Dropout (Augmented Training, Best $\lambda$)

| Model | Architecture | Best $\lambda$ | $\|V\|$ RMSE (pu) | $\theta$ RMSE (deg) | Physics Residual $\mathcal{L}_{\text{phys}}$ | CUDA Latency | Real-Time 50 Hz |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | Dense Feedforward ($2 \times 256$) | $0.1$ | **$0.000516$** | **$0.0137^\circ$** | **$3.77 \times 10^{-5}$** | **$0.223\text{ ms}$** | **PASS** ($<20\text{ ms}$) |
| **TCN** | Dilated Causal 1D CNN ($5$ blocks) | $0.01$ | $0.000884$ | $0.0167^\circ$ | $1.46 \times 10^{-4}$ | $2.199\text{ ms}$ | **PASS** ($<20\text{ ms}$) |
| **GNN** | Edge-Attributed GCN ($3 \times 128$) | $0.001$ | $0.001295$ | $0.0223^\circ$ | $1.08 \times 10^{-3}$ | $2.262\text{ ms}$ | **PASS** ($<20\text{ ms}$) |

### Out-of-Distribution Drift Under 50% Sensor Blackout (Clean Training)

| Model | Base $\|V\|$ RMSE ($\lambda=0$) | PI-SE $\|V\|$ RMSE ($\lambda^*$) | Voltage Error Reduction | 10% $\to$ 50% Degradation Rate | Resilience Classification |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | $0.002038\text{ pu}$ | $0.001726\text{ pu}$ | **$+15.31\%$** | $+164.0\%$ | Fragile to surprise blackouts |
| **TCN** | $0.001330\text{ pu}$ | $0.001342\text{ pu}$ | $-0.90\%$ ($\theta$: **$+16.4\%$**) | **$+25.7\%$** | **Highly Resilient (Temporal Inertia)** |
| **GNN** | $0.002117\text{ pu}$ | $0.002117\text{ pu}$ | $+0.00\%$ | $+41.4\%$ | Moderate |

---

## Problem Formulation & Physics Integration

### 1. State & Telemetry Vector
The true electrical state is the voltage phasor at all $N = 33$ buses:
$$\mathbf{x} = \left[ |\mathbf{V}|, \; \boldsymbol{\theta} \right]^T \in \mathbb{R}^{66}$$

The models receive a 114-dimensional input vector adhering to the strict **"Standardize First, Mask Second"** data contract:
$$\mathbf{z} = \Big[ \underbrace{(|V|_{\text{PMU}} - \mu) / \sigma \odot \mathbf{m}}_{16}, \; \underbrace{(\theta_{\text{PMU}} - \mu) / \sigma \odot \mathbf{m}}_{16}, \; \underbrace{\tilde{\mathbf{P}}_{\text{pseudo}}}_{33}, \; \underbrace{\tilde{\mathbf{Q}}_{\text{pseudo}}}_{33}, \; \underbrace{\mathbf{m}}_{16} \Big] \in \mathbb{R}^{114}$$

* **PMU Coverage**: 16 PMUs deployed across 32 non-slack buses ($50\%$ physical deployment).
* **PMU Noise**: $\sigma_{|V|} = 0.001\text{ pu}$, $\sigma_\theta = 0.05^\circ$ (IEEE C37.118 compliant).
* **Pseudo-measurements**: $P_{\text{pseudo}} = P_{\text{true}}(1 + \mathcal{N}(0, 0.1^2))$, $Q_{\text{pseudo}} = Q_{\text{true}}(1 + \mathcal{N}(0, 0.1^2))$.
* **Slack Bus Restoration**: Substation Bus 0 is fixed at $|V|=1.0\text{ pu}, \theta=0.0^\circ$ and restored physically post-inference.

### 2. Composite Physics Loss
Models are trained with double-precision (`float64`, `complex128`) arithmetic:
$$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \, \mathcal{L}_{\text{phys}}$$

$$\mathbf{S}_{\text{calc}} = \hat{\mathbf{V}} \odot \left( \hat{\mathbf{V}} \, \mathbf{Y}_{\text{bus}}^T \right)^* = \mathbf{P}_{\text{calc}} + j \mathbf{Q}_{\text{calc}}$$

$$\mathcal{L}_{\text{phys}} = \frac{1}{32} \sum_{i=1}^{32} \left[ \left( P_{\text{calc}, i} - P_{\text{pseudo}, i} \right)^2 + \left( Q_{\text{calc}, i} - Q_{\text{pseudo}, i} \right)^2 \right]$$

---

## Repository Structure

```
PINN-IEEE33-Bus_EEE-G513/
├── make_dataset.py           # Pandapower AC load flow & stochastic blackout generator
├── inspect_dataset.py        # Dataset inspection and integrity verification utility
│
├── MLP/                      # Multi-Layer Perceptron Subproject
│   ├── train_mlp.py          # Unified MLP training engine (Clean & Augmented sweeps)
│   ├── check_spec_contract.py# 5-check data contract test suite
│   ├── plot_results.py       # Publication plotting utility
│   ├── results_augmented/    # Checkpoints & benchmarks for augmented training
│   ├── results_clean/        # Checkpoints & benchmarks for clean training
│   └── demo/                 # CLI single-snapshot visual inspection tool
│
├── GNN/                      # Edge-Attributed Graph Convolutional Network
│   ├── models.py             # EdgeAttributedConv & EdgeAttributedGCN architecture
│   ├── graph_data.py         # 4D admittance edge attribute builder & PyG loader
│   ├── physics.py            # Double-precision AC power flow residual calculator
│   ├── train_gnn.py          # Training and lambda sweep orchestrator
│   ├── check_gnn_contract.py # 6-check unit test suite
│   └── results_augmented/    # Trained GNN checkpoints & benchmarks
│
├── TCN/                      # Temporal Convolutional Network Subproject
│   ├── tcn_model.py          # Causal dilated 1D CNN with Chomp1d and skip connections
│   ├── make_sequential_dataset.py # Generates historical multi-step sliding windows
│   ├── train_tcn.py          # Training engine under temporal load sequences
│   └── results_augmented/    # Trained TCN checkpoints & benchmarks
│
├── Comparison/               # Cross-Architectural Benchmarking & Synthesis
│   ├── compare_all_models.py # Cross-model evaluation & latency profiling tool
│   ├── RESEARCH_VERDICT.md   # Final comparative engineering verdict
│   ├── tables/               # Automated summary tables (hardware_profiles.csv, etc.)
│   └── plots/                # High-resolution comparative benchmark curves
│
└── Visualiser/               # Interactive Grid State Estimation Web App
    ├── app.py                # Streamlit dashboard for telemetry & error inspection
    ├── precomputed_data.csv  # Precomputed snapshot metrics for live exploration
    └── README.md             # Visualiser documentation & usage instructions
```

---

## Installation & Setup

### Prerequisites
* Python 3.10 or higher
* CUDA-compatible GPU (recommended for acceleration, though CPU inference is supported)

### 1. Clone the Repository
```bash
git clone https://github.com/reallyrehans/PINN-IEEE33-Bus_EEE-G513.git
cd PINN-IEEE33-Bus_EEE-G513
```

### 2. Install Dependencies
```bash
pip install torch torchvision torchaudio
pip install torch_geometric
pip install pandapower numpy pandas matplotlib streamlit
```

---

## Usage Guide

### 1. Generate Datasets (Sprint 1 Pipeline)
Generate synthetic load snapshots, PMU noise, and dropout masks across the IEEE 33-bus feeder:
```bash
python make_dataset.py --out MLP/data --n_train 8000 --n_val 1000 --n_test 1000
```
For temporal sequences used by the TCN:
```bash
python TCN/make_sequential_dataset.py --in_dir MLP/data/ieee33 --out_dir TCN/data/ieee33 --window 10
```

### 2. Verify Data & Physics Contracts
Run automated unit tests to ensure compliance with the double-precision data contracts:
```bash
python MLP/check_spec_contract.py
python GNN/check_gnn_contract.py --data MLP/data/ieee33
python TCN/check_spec_contract.py
```

### 3. Train Models
Train models across the $\lambda$ regularization sweep ($\lambda \in [0.0, 10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1.0, 10.0, 100.0, 1000.0]$):

* **Train MLP (Augmented Dropout)**:
  ```bash
  cd MLP
  python train_mlp.py --data data/ieee33 --epochs 100 --aug --out results_augmented
  cd ..
  ```
* **Train GNN (Augmented Dropout)**:
  ```bash
  cd GNN
  python train_gnn.py --data ../MLP/data/ieee33 --epochs 100 --aug --out results_augmented
  cd ..
  ```
* **Train TCN (Augmented Dropout)**:
  ```bash
  cd TCN
  python train_tcn.py --data data/ieee33 --epochs 100 --aug --out results_augmented
  cd ..
  ```

### 4. Cross-Architectural Benchmark & Latency Profiling
Run the complete cross-model evaluation suite to regenerate performance tables and latency metrics:
```bash
cd Comparison
python compare_all_models.py --root ../
```

### 5. Launch the Interactive Dashboard
Launch the interactive Streamlit dashboard to explore single-snapshot estimations, bus error distributions, and PMU blackout patterns:
```bash
cd Visualiser
streamlit run app.py
```

---

## Hardware Profiling & Latency Breakdown

Evaluated on CUDA (`float64` double precision, batch size 1, 300 iterations synchronized):

| Model | Parameters | Checkpoint Size | Latency (CUDA) | Latency (CPU) | 50 Hz IEEE C37.118 Status |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | $112{,}194$ | $0.86\text{ MB}$ | **$0.223\text{ ms}$** | $0.412\text{ ms}$ | **PASS** ($<20\text{ ms}$) |
| **GNN** | $86{,}402$ | $0.67\text{ MB}$ | **$2.262\text{ ms}$** | $4.831\text{ ms}$ | **PASS** ($<20\text{ ms}$) |
| **TCN** | $256{,}706$ | $1.97\text{ MB}$ | **$2.199\text{ ms}$** | $5.120\text{ ms}$ | **PASS** ($<20\text{ ms}$) |

---


### Project Contributors
* **Manan Arora** ([f20240363@goa.bits-pilani.ac.in](mailto:f20240363@goa.bits-pilani.ac.in))
* **Rehan Siddiqui** ([f20240377@goa.bits-pilani.ac.in](mailto:f20240377@goa.bits-pilani.ac.in))
* **Vinayak Singh** ([f20240354@goa.bits-pilani.ac.in](mailto:f20240354@goa.bits-pilani.ac.in))

**Institution**: Department of Electrical and Electronics Engineering, BITS Pilani K K Birla Goa Campus.  
**Course**: EEE G513 / Machine Learning for Electrical Engineering (ML4EE).  
**Course Instructor**: Prof. Shivin Srivastava.
