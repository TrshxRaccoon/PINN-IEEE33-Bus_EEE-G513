# Physics-Informed State Estimation (PI-SE) in Modern Distribution Grids
## Comprehensive Research Paper Analysis & Interview Defense Guide

---

### Executive Overview

This document provides:
1. **In-depth technical summaries** of four state-of-the-art research papers on Physics-Informed Neural Networks (PINNs) and Graph Neural Networks (GNNs) for Power Systems State Estimation (PSSE / DSSE).
2. **Explicit mathematical formulations, equations, and quantitative benchmark results** from each paper.
3. **Critical research gaps and practical deployment limitations** identified in each work.
4. **Structured interview defense responses (Questions 1 to 5)** meticulously aligned with your current research project on Physics-Informed State Estimation under sparse PMU measurements and communication dropouts.

---

# PART I: Detailed Summaries of the 4 Research Papers

---

## Paper 1: Physics-Informed Neural Networks for Enhanced State Estimation in Unbalanced Distribution Power Systems

* **Authors**: Petros Iliadis, Stefanos Petridis, Angelos Skembris, Dimitrios Rakopoulos, Elias Kosmatopoulos
* **Publication**: *Applied Sciences* (MDPI), Vol. 15, Issue 14, Art. 7507, July 2025
* **Target Domain**: Unbalanced Three-Phase Distribution System State Estimation (DSSE) with High DER Penetration

---

### 1. Topic & Problem Scope
Modern distribution systems face high penetration of Distributed Energy Resources (DERs), bidirectional power flows, and intrinsic three-phase asymmetry. Traditional state estimators (such as Weighted Least Squares, WLS) struggle with convergence and computational latency due to non-convex AC power flow equations and ill-conditioned admittance matrices. Purely data-driven Deep Neural Networks (DNNs) achieve fast inference but act as unconstrained black boxes, violating basic electrical laws (Ohm’s and Kirchhoff’s laws) when subjected to unseen operating conditions, sensor noise, or partial observability. This paper proposes a PINN framework to enforce physical current injection consistency directly during training for unbalanced distribution systems.

---

### 2. Methodology & Architecture
* **Input-Output Representation**:
  * The network maps nodal complex power injections $S = P + jQ$ to bus complex voltages $V = V_{\text{real}} + jV_{\text{imag}}$.
  * To operate within standard deep learning frameworks without native complex tensor support, complex vectors of dimension $N$ are converted into concatenated real-valued vectors of size $2N$:
    $$\mathbf{x}_{\text{in}} = \begin{bmatrix} \operatorname{Re}(\mathbf{S}) \\ \operatorname{Im}(\mathbf{S}) \end{bmatrix} \in \mathbb{R}^{2N}, \quad \hat{\mathbf{y}}_{\text{out}} = \begin{bmatrix} \operatorname{Re}(\hat{\mathbf{V}}) \\ \operatorname{Im}(\hat{\mathbf{V}}) \end{bmatrix} \in \mathbb{R}^{2N}$$
* **Network Backbone**:
  * Fully connected Multilayer Perceptron (MLP) consisting of 2 hidden layers with 256 neurons each.
  * Activation function: Hyperbolic Tangent ($\tanh$) to ensure continuous, non-zero higher-order derivatives for gradient backpropagation.
* **Physics Integration**:
  * Rather than reconstructing power flow iteratively, the network computes predicted current injections $\hat{\mathbf{I}} = \mathbf{Y}_{\text{bus}} \hat{\mathbf{V}}$ via a vectorized matrix multiplication with the complex nodal admittance matrix $\mathbf{Y}_{\text{bus}}$ and penalizes discrepancies against true current injections.
  * Training uses double-precision floating-point arithmetic (`float64`) to eliminate numerical instabilities arising from high dynamic ranges of impedance values.

---

### 3. Mathematical Equations & Loss Formulation
1. **Three-Phase Nodal Power Injection**:
   For bus $i$ and phase $\phi \in \{a, b, c\}$:
   $$S_i^\phi = V_i^\phi \left( \sum_{j=1}^N \sum_{\psi \in \{a,b,c\}} Y_{ij}^{\phi\psi} V_j^\psi \right)^*$$
   $$P_i^\phi = \operatorname{Re}\left(S_i^\phi\right), \quad Q_i^\phi = \operatorname{Im}\left(S_i^\phi\right)$$
   where $Y_{ij}^{\phi\psi}$ denotes the mutual admittance between phase $\phi$ at bus $i$ and phase $\psi$ at bus $j$.

2. **Total PINN Loss Function**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{voltage}} + \lambda \, \mathcal{L}_{\text{current}}$$
   where $\lambda$ is a manually tuned fixed regularization weight ($\lambda \approx 10^{-3}$).

3. **Supervised Voltage Loss (MSE)**:
   $$\mathcal{L}_{\text{voltage}} = \frac{1}{m} \sum_{i=1}^m \left| V_i - \hat{V}_i \right|^2$$

4. **Physics-Based Current Consistency Loss (MAE)**:
   $$\hat{\mathbf{I}} = \mathbf{Y}_{\text{bus}} \hat{\mathbf{V}}$$
   $$\mathcal{L}_{\text{current}} = \frac{1}{m} \sum_{i=1}^m \left| I_i - \hat{I}_i \right|$$

---

### 4. Key Results & Benchmarks
* **Simulation Setup**: OpenDSS synthetic time-series simulations generating 10,000 snapshots (80% train, 20% test) across 4 IEEE benchmark feeders: IEEE 13-bus, 34-bus, 37-bus, and 123-bus.
* **Accuracy Improvements under Noise-Free Conditions**:
  * **Current Estimation RMSE Reduction**: Improved by **97.87%** (IEEE 13), **85.45%** (IEEE 34), **97.88%** (IEEE 37), and **97.07%** (IEEE 123) over unconstrained baseline MLPs.
  * **Voltage Estimation RMSE Reduction**: Improved by **37.13%** (IEEE 13), **13.77%** (IEEE 34), **63.67%** (IEEE 37), and **22.44%** (IEEE 123).
* **Convergence Speed**: The PINN converged within the first **20 epochs**, whereas the baseline required several hundred epochs to reach comparable accuracy.
* **Robustness under Gaussian Noise (5% input noise)**:
  * On IEEE 37-bus, PINN current RMSE was **3.35 A** compared to **59.65 A** for baseline MLP (>94% improvement).
* **Robustness under Missing Measurements (10% to 20% input dropout)**:
  * On IEEE 123-bus with 20% dropout, PINN maintained a **94.05%** lower current RMSE than the baseline.

---

### 5. Critical Research Gaps & Practical Limitations
1. **Assumption of Full Power Injection Observability**: The model assumes power injections are known at *all* nodes as input, which is rarely realistic in low-voltage distribution grids where smart meters report intermittently.
2. **Fixed Admittance Matrix $\mathbf{Y}_{\text{bus}}$**: The model cannot handle topology reconfigurations, feeder switching, or line faults without retraining the network.
3. **Static Loss Weighting**: The hyperparameter $\lambda$ is static ($10^{-3}$), which can cause gradient competition between voltage loss and current loss during different training stages.
4. **Poor Accuracy at Zero-Injection Buses**: The current loss formulation produces noticeable angular errors at passive/transshipment buses where current injection is mathematically zero.
5. **No Exploitation of Graph Topology**: The model uses a flat MLP architecture, discarding the sparse spatial connectivity graph of the feeder.

---

## Paper 2: Distribution System State Estimation Using Physics-Guided Deep Learning Approach

* **Authors**: Y. Raghuvamsi, Kiran Teeparthi, Vinod Kumar D.M., Imran Abdul, Srihari Parri
* **Publication**: *Electric Power Systems Research* (Elsevier), Vol. 236, Art. 110922, Nov 2024
* **Target Domain**: Branch Current-based Distribution System State Estimation (BC-DSSE) under High RES Penetration and Missing Telemetry

---

### 1. Topic & Problem Scope
Traditional WLS state estimation based on bus voltages suffers from high computational complexity and Jacobian matrix recalculations at every iteration. While Branch Current-based DSSE (BC-DSSE) simplifies the formulation, it remains vulnerable to divergence when measurement devices fail or telemetry packets are dropped. Existing deep learning approaches lack physical interpretability and temporal awareness. This paper proposes a **Physics-Guided Temporal Convolutional Network (Ph-TCN)** configured in an autoencoder framework to estimate branch currents while leveraging the constant Jacobian property of BC-DSSE.

---

### 2. Methodology & Architecture
* **Autoencoder Structure with Embedded Physics Decoder**:
  * **Encoder**: Temporal Convolutional Network (TCN) taking historical time-series of active/reactive power injections ($P_{\text{inj}}, Q_{\text{inj}}$) and branch current flows ($I_{\text{flow}}^r, I_{\text{flow}}^x$) to estimate system branch currents in rectangular coordinates $\hat{\mathbf{x}} = [V_s, \mathbf{i}_r, \mathbf{i}_x]^T$.
  * **Decoder (Non-Trainable Physics Layer)**: Uses the linear measurement equations of BC-DSSE with a **constant Jacobian matrix** $\mathbf{H}$ to reconstruct measurement vectors $\hat{\mathbf{z}} = \mathbf{H} \hat{\mathbf{x}}$.
* **TCN Architectural Components**:
  * **Dilated Causal Convolutions**: Ensures output at time $t$ depends strictly on $t, t-1, \dots$ (no future leakage) with exponentially expanding receptive field ($d = 1, 2, 4$).
  * **Residual Blocks**: Stacks 1D dilated causal convolutions, weight normalization, ReLU activations, dropout layers (0.2), and skip connections to eliminate vanishing/exploding gradients.

---

### 3. Mathematical Equations & Loss Formulation
1. **BC-DSSE Linear Measurement Model**:
   $$\mathbf{z} = \mathbf{h}(\mathbf{x}) + \boldsymbol{\eta}$$
   Using Equivalent Current Measurements (ECMs):
   $$I_k^{\text{EQ}} = \left[ \frac{P_k + jQ_k}{V_k} \right]^* = \operatorname{Re}\left(I_k^{\text{EQ}}\right) + j \operatorname{Im}\left(I_k^{\text{EQ}}\right)$$
   Because state variables are branch currents $\mathbf{i}_r, \mathbf{i}_x$ and substation voltage $V_s$, the resulting Jacobian matrix $\mathbf{H}$ contains elements of only $0, 1, -1$ and remains constant across iterations.

2. **Dilated Convolution Operator**:
   $$\mathcal{D}_c(x) = (p *_d f)(x) = \sum_{j=0}^{k-1} f(j) \, p_{x - d \cdot j}$$
   where $k$ is filter size, $d$ is dilation rate, and $p$ is the historical input sequence.

3. **Cumulative Supervised + Self-Supervised Loss**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{supervised}}(\mathbf{x}, \hat{\mathbf{x}}) + \mathcal{L}_{\text{physics}}(\mathbf{z}, \hat{\mathbf{z}})$$
   $$\mathcal{L}_{\text{supervised}} = \frac{1}{m} \sum_{k=1}^m \left( x_k - \hat{x}_k \right)^2 \quad (\text{MSE})$$
   $$\mathcal{L}_{\text{physics}} = \mathcal{L}_\delta(\mathbf{z}, \hat{\mathbf{z}}) \quad (\text{Problem-Specific Modified Huber Loss})$$

4. **Modified Huber Loss**:
   $$\mathcal{L}_\delta(\mathbf{z}, \hat{\mathbf{z}}) = \begin{cases} (\hat{\mathbf{z}} - \mathbf{z})^2, & \text{for } |\hat{\mathbf{z}} - \mathbf{z}| \le \delta \\ \delta |\hat{\mathbf{z}} - \mathbf{z}| - \frac{1}{2} \delta^2, & \text{otherwise} \end{cases}$$
   where threshold $\delta = \sigma \cdot \mathbf{z}$ is dynamically scaled by the measurement standard deviation $\sigma$.

---

### 4. Key Results & Benchmarks
* **Simulation Setup**: 8,760 hourly load and generation profiles (1 full year) derived from National Renewable Energy Laboratory (NREL) datasets, scaled to IEEE 13-bus, IEEE 37-bus, and IEEE 123-bus test systems. Corrupted with 2% smart meter noise and 1% current sensor noise.
* **Accuracy Comparisons**:
  * **IEEE 13-bus**: Ph-TCN achieved **RMSE = 0.4012 A** ($R^2 = 0.9142$) vs. WLS (4.7664 A), MLP (3.8772 A), standard TCN (2.6459 A), and Ph-MLP (1.1267 A).
  * **IEEE 37-bus**: Ph-TCN achieved **RMSE = 0.2210 A** ($R^2 = 0.9506$) vs. WLS (3.9842 A), standard TCN (2.0422 A), and Ph-RNN (1.1430 A).
  * **IEEE 123-bus**: Ph-TCN achieved **RMSE = 0.5128 A** ($R^2 = 0.9438$) compared to 5.1245 A for standard TCN.
* **Robustness under Missing Data (10%, 30%, 50% packet dropouts)**:
  * At **50% missing data** on IEEE 37-bus, Ph-TCN maintained **RMSE = 5.0856 A** ($R^2 = 0.8645$), whereas standard TCN degraded to **16.1267 A** ($R^2 = 0.6843$), and conventional WLS completely failed to converge.
* **Computational Time**:
  * Inference execution time on IEEE 123-bus was **0.24 seconds**, well within typical distribution control cycle requirements.

---

### 5. Critical Research Gaps & Practical Limitations
1. **Requires Full System Observability for Target Generation**: The encoder must be trained on ground-truth state estimates generated from observable WLS solutions; it cannot be trained natively under persistent unobservability.
2. **Fixed Radial Topology**: The constant Jacobian matrix $\mathbf{H}$ relies entirely on a fixed tree topology. Any tie-switch closure or feeder reconfiguration invalidates $\mathbf{H}$.
3. **Hourly Temporal Resolution**: The model is evaluated only on 1-hour intervals, failing to capture sub-second dynamic transients, inverter switching, or sudden solar PV ramping.
4. **Strict Gaussian Noise Assumption**: Evaluated only under Gaussian noise; does not investigate heavy-tailed or adversarial non-Gaussian errors.

---

## Paper 3: Physics-Informed Graphical Neural Network for Power System State Estimation

* **Authors**: Quang-Ha Ngo, Bang L.H. Nguyen, Tuyen V. Vu, Jianhua Zhang, Tuan Ngo
* **Publication**: *Applied Energy* (Elsevier), Vol. 358, Art. 122602, Feb 2024
* **Target Domain**: Dynamic and Quasi-Static State Estimation in Graph-Structured Power Networks under Load Perturbations

---

### 1. Topic & Problem Scope
Traditional power system state estimation methods treat electrical networks as Euclidean data, neglecting the irregular graph topology of power grids. Conventional model-based Dynamic State Estimation (DSE) methods (like Extended Kalman Filters) require exact dynamic system models and continuous full measurements, making them vulnerable to telemetry loss. Conversely, purely data-driven GNNs can overfit when training data is sparse or imbalanced. This paper develops a **Physics-Informed Graphical Neural Network (PI-GNN)** that hybridizes localized model-based Kalman Filtering in PMU-equipped sub-areas with Graph Convolutional Networks (GCN) across the entire grid.

---

### 2. Methodology & Architecture
* **Hybrid Fusion Framework**:
  1. **Model-Based Sub-Area Estimation**: In localized zones with micro-PMUs ($\mu$PMUs), a dynamic state estimator based on differential branch current equations in the three-phase $abc$-frame runs an unknown-input Kalman Filter to generate high-precision local states $\hat{\mathbf{x}}_M$.
  2. **Data-Driven Graph Learning**: A Spectral Graph Convolutional Network (GCN) ingests nodal measurements and topology adjacency matrices $\mathcal{G}(\mathcal{V}, \mathcal{E})$ to estimate system-wide voltage states $\hat{\mathbf{x}}_D$.
  3. **Physics-Guided Overlap Fusion**: In regions where model-based and GCN predictions overlap ($\hat{\mathbf{x}}_O$), a difference penalty factor ($\operatorname{diff}_k$) enforces consistency between model-based physics and neural network predictions.

---

### 3. Mathematical Equations & Loss Formulation
1. **Dynamic Branch Current Differential Equation ($abc$-frame)**:
   $$\frac{d \mathbf{i}_{ij,abc}}{dt} = -\frac{R_{ij}}{L_{ij}} \mathbf{i}_{ij,abc} + \frac{1}{L_{ij}} \left(\mathbf{v}_{i,abc} - \mathbf{v}_{j,abc}\right)$$

2. **State-Space Dynamic Representation**:
   $$\begin{cases} \mathbf{x}_k = \mathbf{A} \mathbf{x}_{k-1} + \mathbf{B} \mathbf{u}_{k-1} + \mathbf{w}_{k-1} \\ \mathbf{z}_{x,k} = \mathbf{C} \mathbf{x}_k + \mathbf{v}_{x,k} \\ \mathbf{z}_{u,k} = \mathbf{D} \mathbf{u}_k + \mathbf{v}_{u,k} \end{cases}$$

3. **Kalman Predictor-Corrector Update**:
   $$\hat{\mathbf{x}}_{k|k-1} = \mathbf{A} \hat{\mathbf{x}}_{k-1} + \mathbf{B} \hat{\mathbf{u}}_{k-1}$$
   $$\hat{\mathbf{x}}_k = \hat{\mathbf{x}}_{k|k-1} + \mathbf{K}_k \left( \mathbf{z}_{x,k} - \mathbf{C} \hat{\mathbf{x}}_{k|k-1} \right)$$
   $$\mathbf{P}_{x,k} = (\mathbf{I} - \mathbf{K}_k \mathbf{C}) \mathbf{P}_{k|k-1}$$

4. **Spectral Graph Convolution Layer**:
   $$\mathbf{H}^{l+1} = \sigma\left( \tilde{\mathbf{D}}^{-\frac{1}{2}} \tilde{\mathbf{A}} \tilde{\mathbf{D}}^{-\frac{1}{2}} \mathbf{H}^l \mathbf{W}_l \right)$$
   where $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}_N$ is the graph adjacency matrix with self-loops, and $\tilde{\mathbf{D}}_{ii} = \sum_j \tilde{A}_{ij}$ is the diagonal degree matrix.

5. **Physics-Informed Fusion Loss**:
   $$\hat{\mathbf{x}}_k^O = \frac{\hat{\mathbf{x}}_k^M + \hat{\mathbf{x}}_k^{D*}}{2}, \quad \operatorname{diff}_k = \hat{\mathbf{x}}_k^M - \hat{\mathbf{x}}_k^{D*}$$
   $$\mathcal{L}(\hat{\mathbf{x}}, \mathbf{x}) = \operatorname{MSE}(\mathbf{x} - \hat{\mathbf{x}}_D) + \operatorname{MSE}\left[ \operatorname{diff}_k \odot \left(\mathbf{x}_O - \hat{\mathbf{x}}_O\right) \right]$$

---

### 4. Key Results & Benchmarks
* **Simulation Systems**: IEEE 5-bus, IEEE 123-bus, and large-scale IEEE 8,500-bus distribution networks under sudden load variations (70% to 130% nominal).
* **Estimation Performance (Test MSE)**:
  * **IEEE 5-bus**: PINN achieved **$\text{MSE} = 8.7 \times 10^{-5}$** vs. GNN ($2.0 \times 10^{-4}$) and MLP ($3.0 \times 10^{-4}$).
  * **IEEE 123-bus**: PINN achieved **$\text{MSE} = 0.0034$** vs. GNN ($0.0040$) and MLP ($0.0051$).
  * **IEEE 8,500-bus**: PINN voltage magnitude $\text{MSE} = \mathbf{0.0055}$ vs. GNN ($0.0172$) and MLP ($0.0161$); phase angle $\text{MSE} = \mathbf{0.0007}$ vs. GNN ($0.0008$).
* **Batch Size Trade-off**: Demonstrated that batch sizes between 4 and 16 achieved optimal convergence trade-offs for large graph models.

---

### 5. Critical Research Gaps & Practical Limitations
1. **Reliance on Localized PMU Availability**: Requires at least some sub-areas to have full $\mu$PMU observability to initialize the model-based Kalman filter.
2. **Simplified Line Parameter Assumptions**: Neglects shunt capacitance and assumes perfectly known resistance and inductance parameters ($R_{ij}, L_{ij}$).
3. **No Bad Data / Cyber-Attack Handling**: The loss formulation does not account for false data injection (FDI) attacks or gross measurement outliers.
4. **Ad-Hoc Fusion Weighting**: The weighting factor between model-based and GNN outputs uses simple scalar averaging ($\hat{\mathbf{x}}_O = (\hat{\mathbf{x}}_M + \hat{\mathbf{x}}_D)/2$) rather than covariance-based optimal weighting.

---

## Paper 4: High-Resolution Real-Time Power Systems State Estimation: A Combined Physics-Embedded and Data-Driven Perspective

* **Authors**: Jianxiong Hu, Qi Wang, Yujian Ye, Yi Tang
* **Publication**: *IEEE Transactions on Power Systems*, Vol. 40, No. 2, pp. 1532–1545, March 2025
* **Target Domain**: High-Temporal-Spatial-Resolution Real-Time State Estimation using Hybrid SCADA and Wide Area Measurement System (WAMS) Data

---

### 1. Topic & Problem Scope
Online Dynamic Security Assessment (DSA) requires real-time operating states at sub-second resolution. However, high-resolution PMU measurements (WAMS) are sparse and costly, while legacy SCADA measurements are globally redundant but updated slowly (every 2–4 seconds). Model-based WLS cannot handle millisecond streaming updates, and conventional supervised learning (SL) models suffer from a fundamental trade-off: **optimizing state variables ($V, \theta$) does not guarantee the accuracy of calculated observation variables ($P, Q, I$), and vice versa (the "one-to-multiple" mapping issue)**. This paper introduces a **Physics-Embedded State Estimation (PESE)** framework combining a Multi-Head Graph Attention Network (MGAT) with an Augmented Lagrangian adaptive physics training formulation.

---

### 2. Methodology & Architecture
* **Two-Stage PESE Model**:
  1. **Physics Feature Transformation**:
     * **SCADA Processing**: Runs WLS once per SCADA cycle to provide a low-frequency baseline state $\mathbf{X}^s$.
     * **WAMS Linear Branch Measurement Transformation (LBMT)**: Converts high-speed branch current measurements $I_{ij}$ from PMUs into virtual nodal voltage measurements $\mathbf{O}_v$ for adjacent unmetered buses.
     * Combines $\mathbf{X}^s$, real PMU measurements $\mathbf{O}_r$, and virtual measurements $\mathbf{O}_v$ into unified node features $\mathbf{X}^0$ using one-hot type identifiers.
  2. **Data-Driven Residual Correction (MGAT + ResNet)**:
     * A $K$-head Graph Attention Network (MGAT) extracts spatial features across buses.
     * The network does not predict states from scratch; it predicts the residual difference between the true current state $\mathbf{x}$ and the recent baseline $\mathbf{X}^s$:
       $$g_\theta(\mathbf{X}^0) = h_\theta^{L,K}(\mathbf{X}^0) + \mathbf{X}^s$$

---

### 3. Mathematical Equations & Loss Formulation
1. **WLS Objective**:
   $$\mathbf{x}^* = \arg\min_{\mathbf{x}} [\mathbf{z} - \mathbf{h}(\mathbf{x})]^T \mathbf{W} [\mathbf{z} - \mathbf{h}(\mathbf{x})]$$

2. **Virtual Voltage Calculation via LBMT**:
   For bus $j$ adjacent to PMU-equipped bus $i$:
   $$\mathbf{V}_j = \frac{1}{|S_j|} \sum_{i \in S_j} \left( \mathbf{V}_i - \frac{\dot{\mathbf{I}}_{ij} - Y_{i0} \mathbf{V}_i}{Y_{ij}} \right)$$

3. **Multi-Head Graph Attention Layer**:
   $$\mathbf{X}_i^l = \Vert_{k=1}^K \sigma \left( \sum_{j \in \mathcal{N}_i} \alpha_{ij}^{k,l} \mathbf{W}^{k,l} \mathbf{X}_j^{l-1} \right)$$
   $$\alpha_{ij}^{k,l} = \operatorname{softmax}_j \left( \operatorname{LeakyReLU}\left( (\mathbf{a}^l)^T [\mathbf{W}^{k,l} \mathbf{X}_i^{l-1} \, \Vert \, \mathbf{W}^{k,l} \mathbf{X}_j^{l-1}] \right) \right)$$

4. **Augmented Lagrangian Physics-Embedded Loss**:
   $$J(\theta) = \frac{1}{B} \sum_{i=1}^B \left[ M_z\left(g_\theta(\mathbf{X}_i^0)\right) + \lambda \left( M_x\left(g_\theta(\mathbf{X}_i^0)\right) - d \right) \right]$$
   where $M_z(g) = (\mathbf{h}(\mathbf{x}) - \mathbf{h}(g(\mathbf{z})))^2$ is the observation error (physics power flow loss), $M_x(g) = (\mathbf{x} - g(\mathbf{z}))^2$ is the state error, and $d$ is a state-fitting error tolerance threshold.

5. **Adaptive Multiplier Update**:
   $$\lambda \leftarrow \operatorname{clip}\left( \lambda + \eta_\lambda \left( \frac{1}{B} \sum_{i=1}^B M_x\left(g_\theta(\mathbf{X}_i^0)\right) - d \right), \, 0, \, \bar{\lambda} \right)$$
   *Mathematical Guarantee (Lemmas 1 & 2)*: When state error $M_x \le d$, $\lambda \to 0$, forcing the optimizer to minimize observation error $M_z$ without deteriorating state accuracy.

---

### 4. Key Results & Benchmarks
* **Simulation Systems**: IEEE 39-bus and IEEE 118-bus systems with dynamic wind and load fluctuations.
* **Accuracy & Resolution**:
  * Outperformed pure Supervised Learning (SL) and conventional physics-guided neural networks across all PMU density scenarios.
  * In IEEE 118-bus, state and observation variable MAEs remained below **0.003 p.u.** and **0.1 p.u.** respectively.
* **Online Execution Speed**:
  * Inference latency was under **0.5 milliseconds**, more than $1000\times$ faster than WLS iterations and fast enough for 30–60 Hz PMU streaming rates.
* **Topology Switching Robustness**:
  * Randomly disconnected transmission lines (Lines 6–8, 11–12, 26–29); MAE of states increased by only **0.0005**, demonstrating topological adaptability via graph attention mechanisms.
* **Bad Data Resilience**:
  * Injected 10x noise into 10% of measurements; the model maintained stable state estimates due to spatial attention filtering.

---

### 5. Critical Research Gaps & Practical Limitations
1. **Requires Periodic Global SCADA Convergence**: The model relies on a convergent WLS baseline state $\mathbf{X}^s$. If the SCADA system experiences complete communication failure, the baseline state becomes stale.
2. **Computational Overhead of Physics Loss Gradients**: Calculating the observation loss $M_z(g) = (\mathbf{h}(g(\mathbf{z})) - \mathbf{z})^2$ and its backpropagation gradient through nonlinear AC power flow equations increases training time per epoch by ~40% compared to pure supervised learning.
3. **Hyperparameter Sensitivity of Threshold $d$**: If $d$ is chosen smaller than the achievable neural network approximation error, $\lambda$ explodes to $\bar{\lambda}$, destabilizing the dual optimization.
4. **Transmission Grid Focus**: Evaluated primarily on balanced, meshed transmission grids (IEEE 39, IEEE 118); does not address high $R/X$ ratios and phase unbalance typical of distribution feeders.

---

## Cross-Paper Comparison Matrix

| Feature / Dimension | Paper 1 (Iliadis et al., 2025) | Paper 2 (Raghuvamsi et al., 2024) | Paper 3 (Ngo et al., 2024) | Paper 4 (Hu et al., 2025) |
| :--- | :--- | :--- | :--- | :--- |
| **Primary Architecture** | 2-Layer MLP ($\tanh$) | TCN Autoencoder + Residual Blocks | Spectral GCN + Kalman Filter | Multi-Head GAT (MGAT) + ResNet |
| **Physics Integration Level** | Current injection loss $\hat{\mathbf{I}} = \mathbf{Y}\hat{\mathbf{V}}$ | Constant Jacobian decoder $\hat{\mathbf{z}} = \mathbf{H}\hat{\mathbf{x}}$ | Dynamic differential line equations ($abc$) | LBMT virtual nodes + AC power flow loss |
| **Loss Balancing Method** | Static scalar weight $\lambda \approx 10^{-3}$ | Modified Huber Loss with dynamic $\delta$ | Discrepancy factor $\operatorname{diff}_k$ penalty | Augmented Lagrangian adaptive $\lambda$ |
| **Grid Representation** | Vector concatenation ($2N$) | Time-series sequence (8760 steps) | Graph Adjacency $\tilde{\mathbf{A}} = \mathbf{A} + \mathbf{I}$ | Graph Attention Weights $\alpha_{ij}$ |
| **Handled Observability** | 10%–20% random input dropout | 10%–50% missing telemetry | Sub-area PMU observability | Sparse PMU + SCADA hybrid fusion |
| **Network Type** | Unbalanced distribution | Unbalanced distribution | Balanced & 3-phase microgrids | Meshed transmission systems |
| **Test Systems** | IEEE 13, 34, 37, 123-bus | IEEE 13, 37, 123-bus | IEEE 5, 123, 8500-bus | IEEE 39-bus, IEEE 118-bus |
| **Inference Latency** | Single forward pass (<5 ms) | ~0.14–0.24 s | Batch graph inference (<20 ms) | **< 0.5 ms** (Real-Time DSA) |
| **Primary Limitation** | Static $\mathbf{Y}_{\text{bus}}$, assumes known injections | Fixed radial topology only | Requires localized full observability | Dependent on SCADA WLS baseline |

---

# PART II: Interview Defense Guide (Questions 1 to 5)

---

### Question 1: Explain the Topic and the Scope of the Problem

#### Model Answer:
"My research project focuses on **Physics-Informed State Estimation (PI-SE) for Electrical Distribution Grids under Sparse Measurement and Communication Dropout Conditions**.

#### Context & Problem Scope:
1. **The Energy Transition & Distribution Grid Complexity**:
   With the rapid adoption of rooftop solar photovoltaics, electric vehicle charging infrastructure, and localized battery storage, distribution networks are transforming from passive, unidirectional radial networks into active, dynamic grids with bidirectional power flows and rapid voltage fluctuations.
2. **The Observability Deficit**:
   Unlike high-voltage transmission networks, which are heavily monitored by Supervisory Control and Data Acquisition (SCADA) systems and Phasor Measurement Units (PMUs), distribution feeders are historically under-metered due to the massive cost of deploying synchronized sensors across millions of secondary distribution nodes.
3. **The Failure Modes of Existing Estimators**:
   * **Conventional Model-Based State Estimation (WLS)**: Requires mathematical system observability (a full-rank Jacobian matrix $\mathbf{H}$). When sensors are sparse, or when communication channels drop data packets, WLS suffers from numerical singularity, ill-conditioning, and non-convergence. Furthermore, high resistance-to-reactance ($R/X$) ratios in distribution lines make decoupled power flow approximations invalid.
   * **Purely Data-Driven Deep Learning (MLPs, CNNs)**: While computationally fast during online inference, unconstrained neural networks behave as black boxes. When subjected to communication dropouts, measurement noise, or out-of-distribution load surges, pure MLPs produce estimates that violate fundamental physical laws—such as Kirchhoff’s Current Law (KCL) and AC power-flow conservation—leading to unacceptable control decisions.

#### Objective of My Project:
To design, implement, and benchmark a state estimation framework that **embeds the physical laws of power systems (nodal admittance matrix $\mathbf{Y}_{\text{bus}}$, Ohm’s law, and AC power balance) directly into neural network architectures and loss functions**. The estimator must maintain high fidelity in voltage magnitude ($|V|$) and voltage phase angle ($\theta$) even when **up to 50% of PMU measurement telemetry is lost due to communication failure**."

---

### Question 2: Explain Your Approach to Solving It

#### Model Answer:
"To overcome the dual challenges of unobservability and physical inconsistency, my approach integrates domain knowledge (power systems physics) with deep learning through a **Physics-Informed State Estimator (PI-SE)** operating in rectangular coordinates.

```
                    ┌──────────────────────────────────────────────┐
                    │  Input Telemetry: Masked Sparse PMU Data     │
                    │   x_in = [V_real_meas, V_imag_meas]          │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │           Deep Neural Estimator              │
                    │   Fully Connected / Graph Feature Blocks     │
                    └──────────────────────┬───────────────────────┘
                                           │
                                           ▼
                    ┌──────────────────────────────────────────────┐
                    │   Predicted Full State (All N Buses):        │
                    │       y_pred = [V_real_full, V_imag_full]    │
                    └───────┬──────────────────────────────┬───────┘
                            │                              │
         Supervised Path    │                              │  Physics-Informed Path
                            ▼                              ▼
  ┌────────────────────────────────────┐ ┌───────────────────────────────────────┐
  │         Data-Driven Loss           │ │         Physics Loss Module           │
  │ L_data = MSE(V_pred, V_true)       │ │  1. Current: I = Y_bus * V_pred       │
  │ (Supervised alignment on targets)  │ │  2. Power:   S = V_pred * conj(I)     │
  │                                    │ │  3. Residual: MSE(P_calc, P_spec) +   │
  │                                    │ │               MSE(Q_calc, Q_spec)     │
  └─────────────────┬──────────────────┘ └──────────────────┬────────────────────┘
                    │                                       │
                    └───────────────────┬───────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │   Total Loss:                           │
                   │   L_total = L_data + λ * L_physics      │
                   └─────────────────────────────────────────┘
```

#### Detailed Breakdown of the Approach:
1. **Rectangular Coordinate State Formulation**:
   Instead of using polar coordinates ($|V| \angle \theta$), which introduce transcendental trigonometric operations ($\sin, \cos$) and periodic phase wrap-around discontinuities during gradient backpropagation, the state vector is formulated in rectangular coordinates:
   $$\mathbf{V}_{\text{bus}} = \mathbf{V}_{\text{real}} + j \mathbf{V}_{\text{imag}} \in \mathbb{C}^N$$
   The input layer ingests sparse, noisy PMU measurements $[\mathbf{V}_{\text{meas}}^{\text{real}}, \mathbf{V}_{\text{meas}}^{\text{imag}}] \in \mathbb{R}^{2N}$ (with missing sensor entries zero-imputed), and the output predicts the complete voltage vector across all $N$ buses $[\hat{\mathbf{V}}^{\text{real}}, \hat{\mathbf{V}}^{\text{imag}}] \in \mathbb{R}^{2N}$.

2. **Hardware-Accelerated Physics Loss Module**:
   The complex bus admittance matrix $\mathbf{Y}_{\text{bus}} = \mathbf{G} + j\mathbf{B}$ is extracted directly from feeder branch parameters and registered as a fixed, non-trainable tensor buffer in PyTorch. 
   During every forward pass:
   * **Step A (Nodal Current Injections via Kirchhoff's Current Law)**:
     $$\hat{\mathbf{I}}_{\text{real}} = \mathbf{G} \hat{\mathbf{V}}_{\text{real}} - \mathbf{B} \hat{\mathbf{V}}_{\text{imag}}$$
     $$\hat{\mathbf{I}}_{\text{imag}} = \mathbf{B} \hat{\mathbf{V}}_{\text{real}} + \mathbf{G} \hat{\mathbf{V}}_{\text{imag}}$$
   * **Step B (Complex Power Injection Residuals)**:
     $$\hat{P}_i = \hat{V}_{\text{real},i} \hat{I}_{\text{real},i} + \hat{V}_{\text{imag},i} \hat{I}_{\text{imag},i}$$
     $$\hat{Q}_i = \hat{V}_{\text{imag},i} \hat{I}_{\text{real},i} - \hat{V}_{\text{real},i} \hat{I}_{\text{imag},i}$$
   * **Step C (Physics Residual Loss)**:
     $$\mathcal{L}_{\text{physics}} = \frac{1}{N} \sum_{i=1}^N \left( \hat{P}_i - P_i^{\text{spec}} \right)^2 + \frac{1}{N} \sum_{i=1}^N \left( \hat{Q}_i - Q_i^{\text{spec}} \right)^2$$

3. **Composite Objective Function**:
   $$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{data}} + \lambda_{\text{phys}} \, \mathcal{L}_{\text{physics}}$$
   where $\mathcal{L}_{\text{data}} = \operatorname{MSE}(\hat{\mathbf{V}}, \mathbf{V}_{\text{true}})$ anchors the network to available measurements, while $\mathcal{L}_{\text{physics}}$ acts as a physical regularizer that penalizes violations of electrical network laws across all unmetered buses.

4. **Handling Severe Communication Dropout**:
   To make the network resilient to sensor failure, dropout masks are injected into the input tensors during evaluation and training across five levels (10%, 20%, 30%, 40%, 50%), forcing the network to infer unobserved states via the physical coupling embedded in $\mathbf{Y}_{\text{bus}}$."

---

### Question 3: The Dataset

#### Model Answer:
"Our dataset was generated using a simulation pipeline that couples high-fidelity AC power flow solvers with diurnal customer consumption modeling.

```
   ┌────────────────────────────────────────────────────────┐
   │             Base Feeder Network Topology               │
   │           (IEEE 33-bus Baran & Wu / IEEE 69-bus)       │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │         Diurnal Load Curve Generation (24-Hour)        │
   │  - Superposition of morning ramp & evening peak        │
   │  - Spatial diversity factor (mag scale 0.85-1.15)      │
   │  - Consumer Gaussian noise jitter (scale = 0.04)       │
   │  - Variable lagging power factor (0.90 to 0.95)        │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │     pandapower AC Power Flow Engine (Exact Solution)   │
   │  - Solves nonlinear power flow for each timestamp      │
   │  - Extracts exact V_mag, Theta, P_inj, Q_inj           │
   │  - Extracts exact complex admittance matrix Y_bus      │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │           Sparse PMU Deployment & Telemetry            │
   │  - Strategic placement: ~27% coverage (Substation,     │
   │    branch junctions, lateral feeder ends)              │
   │  - Gaussian white noise injection (IEEE C37.118 std)   │
   │  - Stochastic communication dropouts (10% to 50%)      │
   └───────────────────────────┬────────────────────────────┘
                               │
                               ▼
   ┌────────────────────────────────────────────────────────┐
   │         Partitioned Structured Dataset (.npz)          │
   │     Train (70%)  |  Validation (15%)  |  Test (15%)    │
   │       (Strict temporal order; zero data leakage)       │
   └────────────────────────────────────────────────────────┘
```

#### Detailed Dataset Specifications:
1. **Simulation Platform**:
   Generated using `pandapower`, a validated, industrial-grade power flow simulation tool built on top of PyPower / MATPOWER.
2. **Benchmark Topologies**:
   * **IEEE 33-bus Radial Distribution Feeder (`case33bw`)**: 33 buses, 32 branches, nominal voltage 12.66 kV.
   * **IEEE 69-bus Radial Distribution Feeder (`case69`)**: 69 buses, 68 branches, nominal voltage 12.66 kV, characterized by long lateral feeders and pronounced voltage drops.
3. **Realistic Load Modeling**:
   * Generated **3,000 sequential AC operating points** mimicking continuous time-series demand.
   * Demand profiles feature a **smooth diurnal curve** with a morning ramp (7:00–9:00 AM) and an evening peak (6:00–10:00 PM).
   * **Spatial Diversity**: Each load bus incorporates independent magnitude scaling (0.85 to 1.15) and phase jitter ($\pm 30$ min) to simulate varied residential and commercial behavior.
   * **Reactive Power Coupling**: Power factor varies stochastically between **0.90 and 0.95 lagging**, reflecting realistic distribution loads.
4. **Sparse PMU Deployment (~26–27% Coverage)**:
   * **IEEE 33-bus**: 9 PMUs installed out of 33 buses (Buses: `0, 2, 5, 8, 12, 17, 21, 24, 29`), positioned at the substation slack bus, primary branching nodes, and feeder ends.
   * **IEEE 69-bus**: 18 PMUs installed out of 69 buses (Buses: `0, 3, 8, 11, 15, 20, 27, 33, 40, 46, 50, 53, 58, 61, 64, 66, 67, 68`).
5. **Sensor Measurement Noise**:
   In accordance with the **IEEE C37.118 standard** for synchrophasor measurements:
   * Voltage magnitude noise: Zero-mean Gaussian white noise with $\sigma = 0.5\%$ ($0.005$ p.u.).
   * Voltage phase angle noise: Zero-mean Gaussian noise with $\sigma = 0.005\text{ rad} \approx 0.28^\circ$.
6. **Communication Dropout Datasets**:
   * Evaluated across 5 progressive packet dropout rates: **10%, 20%, 30%, 40%, and 50%**.
   * At each snapshot, a random subset of installed PMUs fails to transmit, testing the model's ability to extrapolate using physical laws.
7. **Data Partitioning**:
   * Split chronologically into **70% Training (2,100 snapshots)**, **15% Validation (450 snapshots)**, and **15% Testing (450 snapshots)**.
   * Chronological contiguous splitting avoids temporal leakage, ensuring the model is evaluated on future operating points."

---

### Question 4: Previous Works Trying to Solve Your Problem and How Your Approach is Different

#### Model Answer:
"Our research builds upon, synthesizes, and improves beyond four key categories of prior work:

```
┌───────────────────────────┬──────────────────────────────────┬─────────────────────────────────────┐
│ Methodology               │ Key Mechanism                    │ Fatal Weakness / Research Gap       │
├───────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ 1. Conventional WLS       │ Iterative Gauss-Newton           │ Diverges under unobservability and  │
│    (Schweppe, 1970)       │ minimization of [z - h(x)]^T R z │ communication dropouts (>10%).      │
├───────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ 2. Black-Box DL           │ Pure Supervised MLP/CNN          │ Violates Kirchhoff's Laws;          │
│    (Standard Data-Driven) │ mapping z -> x                   │ catastrophic error under dropouts.  │
├───────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ 3. Current-Based PINN     │ Loss on current injections       │ Assumes power known at all buses;   │
│    (Iliadis et al., 2025) │ L = L_v + λ * L_current          │ fails at zero-injection nodes.      │
├───────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ 4. Physics TCN Autoenc.   │ Constant Jacobian BC-DSSE        │ Requires observable pre-training;   │
│    (Raghuvamsi, 2024)     │ Decoder z = H * x                │ rigid to fixed radial topology.     │
├───────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ 5. Hybrid GNN-Kalman      │ Local Kalman Filter in PMU zones │ Requires localized full PMU zones;  │
│    (Ngo et al., 2024)     │ + GNN full-grid message passing  │ ad-hoc scalar fusion weighting.     │
├───────────────────────────┼──────────────────────────────────┼─────────────────────────────────────┤
│ 6. Physics-Embedded GAT   │ SCADA baseline + WAMS LBMT       │ Complex multi-stage pipeline;       │
│    (Hu et al., 2025)      │ + Multi-Head GAT + Aug. Lag.     │ transmission-focused; needs SCADA.  │
└───────────────────────────┴──────────────────────────────────┴─────────────────────────────────────┘
```

#### Detailed Differentiation of My Approach:
1. **Versus Conventional WLS**:
   WLS requires continuous topological observability ($m \ge 2n - 1$) and repeatedly computes the inverse of the gain matrix $(\mathbf{H}^T \mathbf{R}^{-1} \mathbf{H})^{-1}$. When PMUs drop 30% to 50% of telemetry, the gain matrix becomes singular and WLS fails. Our PINN performs a non-iterative forward pass in milliseconds, completing state reconstruction regardless of missing channels.
2. **Versus Pure Supervised Deep Learning**:
   Pure data-driven models overfit to specific training patterns and lack physical constraints. Under severe sensor dropout (e.g., 50%), their predictions drift into physically impossible states. By embedding $\mathbf{Y}_{\text{bus}}$ into the loss function, our model penalizes non-physical voltage profiles, bounding the error space.
3. **Versus Iliadis et al. (2025)**:
   * Iliadis et al. assume that power injections are known at *all* nodes as input features. In contrast, our approach addresses realistic distribution grids where **only a sparse subset of buses (~27%) has sensors**.
   * While Iliadis et al. evaluate only 10% and 20% input dropout, we test extreme stress scenarios up to **50% packet loss**.
4. **Versus Raghuvamsi et al. (2024)**:
   * Raghuvamsi et al. formulate BC-DSSE using a constant Jacobian $\mathbf{H}$, which strictly requires an observable system to generate supervised targets.
   * Our approach models the complete nonlinear AC power flow balance directly in rectangular coordinates, providing a direct mapping from sparse voltages to complex power balance.
5. **Versus Ngo et al. (2024) & Hu et al. (2025)**:
   * Ngo et al. require dense micro-PMU coverage in sub-areas to run local Kalman filters, which is unavailable in typical distribution feeders.
   * Hu et al. require a converged SCADA WLS baseline every cycle. Our model is standalone and does not rely on a secondary, slow SCADA solver to maintain baseline states."

---

### Question 5: Your Progress Plan, Work Division, Major Checkpoints, and Final Goal

#### Model Answer:
"To execute this project with academic rigor and clear engineering milestones, I have structured our roadmap into a **4-Phase Sprint Plan**, supported by clear work division and quantitative checkpoints.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          PROJECT ROADMAP TIMELINE                           │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 1 (Weeks 1-3) [COMPLETED - Sprint 1]:                                │
│  - pandapower load-flow generator (IEEE 33-bus & 69-bus)                    │
│  - Diurnal load curve synthesis & sparse PMU placement                      │
│  - PhysicsLossModule with Y_bus tensor registration                         │
│  - Baseline MLP vs. PINN training & dropout benchmarking (10%-50%)          │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 2 (Weeks 4-6) [CURRENT SPRINT - Sprint 2]:                           │
│  - Replace flat MLP with Graph Attention Network (GATv2 / MGAT)             │
│  - Implement Augmented Lagrangian dynamic loss balancing (λ adaptation)     │
│  - Benchmark voltage magnitude (|V|) and phase angle (θ) RMSE               │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 3 (Weeks 7-9) [NEXT MILESTONE - Sprint 3]:                           │
│  - Dynamic topology switching (feeder reconfiguration & line outages)       │
│  - Bad data rejection test (10x noise injection & gross outliers)           │
│  - Non-Gaussian noise evaluation (heavy-tailed measurement noise)           │
├─────────────────────────────────────────────────────────────────────────────┤
│  PHASE 4 (Weeks 10-12) [FINAL VALIDATION & PAPER SUBMISSION]:               │
│  - Scale to unbalanced multi-phase IEEE 123-bus feeder in OpenDSS           │
│  - Real-time hardware deployment latency benchmarking on edge GPU           │
│  - Comprehensive ablation study and final manuscript preparation            │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### Detailed Work Division & Team Responsibilities:
1. **Data Engineering & Power Systems Simulation (Lead: Module 1)**:
   * Development of `dataset_generator.py` using `pandapower`.
   * Diurnal demand curve generation, stochastic reactive power allocation, and simulation of IEEE C37.118 sensor noise and communication dropout masks.
2. **Model Architecture & Physics Loss Implementation (Lead: Module 2)**:
   * Implementation of `models.py`: Baseline estimators, Physics-Informed modules, tensorized admittance matrix operations ($\mathbf{G}, \mathbf{B}$), and power residual equations.
   * Migration to Graph Attention Networks (`torch_geometric.nn.GATv2Conv`) to encode physical line connectivity.
3. **Training Optimization & Adaptive Weighting (Lead: Module 3)**:
   * Execution of `train.py`: Hyperparameter sweeps (learning rate, batch size, gradient clipping).
   * Implementation of Augmented Lagrangian adaptive weighting algorithms to balance data loss versus physics residual loss dynamically.
4. **Benchmarking, Robustness Testing & Documentation (Lead: Module 4)**:
   * Comparative evaluation against WLS and standard deep learning baselines across 10%, 20%, 30%, 40%, and 50% dropouts.
   * Generation of publication-quality error distribution plots, metric tables, and research report synthesis.

#### Major Quantitative Checkpoints & Deliverables:
* **Checkpoint 1 (Completed)**: Dataset generated and validated for IEEE 33-bus (3,000 snapshots); clean and dropout-corrupted `.npz` archives structured; Baseline MLP and PINN implemented in PyTorch.
* **Checkpoint 2 (Week 5)**: Achieve $\ge 40\%$ reduction in voltage angle RMSE ($\theta$) under 50% communication dropout using PINN compared to Baseline MLP.
* **Checkpoint 3 (Week 7)**: Implement Graph Attention Network (GAT) to maintain state estimation error below **0.005 p.u. (magnitude)** and **$0.5^\circ$ (angle)** under line switching topology changes.
* **Checkpoint 4 (Week 9)**: Validate resilience against bad data (10% gross outliers with 10x noise) without loss of convergence.
* **Checkpoint 5 (Week 12)**: Achieve sub-millisecond inference (<1.0 ms) on an NVIDIA Jetson / workstation GPU for real-time distribution management integration.

#### Final Project Goal:
To deliver an **open-source, production-ready, physics-informed graph neural network framework for real-time distribution system state estimation** that guarantees physical consistency, runs in sub-milliseconds, and outperforms conventional WLS by at least $80\%$ in accuracy under extreme sensor unobservability and communication failure."

---
*Document prepared for academic interview defense and research progress presentation.*  
*Generated on: October 15, 2026*
