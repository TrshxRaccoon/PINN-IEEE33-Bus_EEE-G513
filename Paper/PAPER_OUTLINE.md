# IEEE 4-Page Conference Paper: Structure & Comparative Draft

## Title (Working Options)
1. **Resilient Distribution System State Estimation Under Sparse PMUs and Telemetry Dropouts: A Comparative Study of Physics-Informed MLP, GNN, and TCN**
2. **Overcoming Unobservability and Sensor Blackouts in Distribution Grids: Spatial, Topological, and Temporal Trade-offs**

---

## Researcher-Perspective Comparative Abstract

> **Abstract—** Conventional Weighted Least Squares (WLS) state estimation fails under distribution-level unobservability and telemetry dropouts, while standard black-box deep learning (DL) models violate Kirchhoff's laws and suffer catastrophic out-of-distribution (OOD) degradation during unexpected sensor blackouts. Recent physics-informed and graph-based approaches attempt to bridge this gap, yet rely on restrictive assumptions—such as known injections at all buses (current-based PINNs), linearized constant-Jacobian decoders requiring observable pre-training (TCN autoencoders), localized full-PMU zones (hybrid GNN-Kalman), or complex multi-stage SCADA-dependent transmission pipelines (physics-embedded GATs).
>
> In contrast, this study presents a unified, single-stage empirical benchmark of three distinct deep architectures—a global dense Multi-Layer Perceptron (MLP), an Edge-Attributed Graph Convolutional Network (GNN), and a causal Temporal Convolutional Network (TCN)—for sparse distribution system state estimation on the IEEE 33-bus feeder (~50% PMU coverage, 16 PMUs) under 10%–50% sensor dropouts and noisy ($\pm 10\%$) pseudo-measurements. Our findings expose three critical insights overlooked in prior literature: (1) an **accuracy–resilience dichotomy**, where dropout-augmented MLPs achieve the lowest static error ($0.000516$ pu $|V|$ RMSE, $0.223\text{ ms}$) via global spatial cross-correlation, whereas TCNs leverage a 12-hour temporal receptive field to survive *unseen* 50% blackouts with only $+25.7\%$ error growth versus $+164.0\%$ for clean-trained MLPs; (2) a **topological horizon** in radial feeders, where 3-hop GNN message passing ($0.001295$ pu) cannot span a 17-hop radial diameter without oversmoothing; and (3) a **physics regularization "Pareto cliff"**, demonstrating that full AC power-flow penalties improve consistency at moderate weights ($\lambda \in [0.001, 0.1]$) but degrade accuracy at $\lambda \ge 10.0$ by forcing models to fit $\pm 10\%$ pseudo-measurement forecast noise.

---

## Literature Positioning Matrix (Table I Candidate for Paper)

| Methodology / Reference | Key Mechanism | Fatal Weakness / Research Gap | How Our Work Addresses / Contrasts It |
| :--- | :--- | :--- | :--- |
| **1. Conventional WLS** (Schweppe, 1970) | Iterative Gauss-Newton minimization of $[z - h(x)]^T R^{-1} [z - h(x)]$ | Diverges under unobservability and communication dropouts ($>10\%$). | Non-iterative ($0.22\text{–}2.20\text{ ms}$); operates cleanly under ~50% base PMU coverage + up to 50% random dropout without matrix inversion. |
| **2. Black-Box DL** (Standard Data-Driven) | Pure supervised MLP/CNN mapping $z \to x$ | Violates Kirchhoff's laws; catastrophic error under dropouts. | Reproduces the OOD collapse in Clean MLP ($+164.0\%$ error at 50% dropout) and resolves it via dropout augmentation or 12-h TCN inertia ($+25.7\%$). |
| **3. Current-Based PINN** (Iliadis et al., 2025) | Loss on current injections $L = L_v + \lambda L_{\text{current}}$ | Assumes power known at all buses; fails at zero-injection nodes. | Uses full AC power-flow residual $\mathbf{S}_{\text{calc}} - \mathbf{S}_{\text{meas}}$ with $\pm 10\%$ pseudo-measurement noise; identifies the $\lambda \ge 10.0$ **Pareto cliff**. |
| **4. Physics TCN Autoenc.** (Raghuvamsi, 2024) | Constant-Jacobian BC-DSSE decoder $z = Hx$ | Requires observable pre-training; rigid to fixed radial topology. | Trains TCN directly for nonlinear AC state mapping from 12-h history under ~50% observability—no constant-Jacobian or observable pre-training needed. |
| **5. Hybrid GNN-Kalman** (Ngo et al., 2024) | Local Kalman Filter in PMU zones + GNN full-grid message passing | Requires localized full PMU zones; ad-hoc scalar fusion weighting. | Evaluates a single-stage `EdgeAttributedGCN` without full-PMU zones or Kalman fusion; exposes the **3-hop vs. 17-hop topological horizon** on radial feeders. |
| **6. Physics-Embedded GAT** (Hu et al., 2025) | SCADA baseline + WAMS LBMT + Multi-Head GAT + Aug. Lag. | Complex multi-stage pipeline; transmission-focused; needs SCADA. | Lightweight single-stage distribution estimator without SCADA dependencies; quantifies exact latency ($0.22\text{–}2.20\text{ ms}$) vs. resilience trade-offs. |

---

## Target 4-Page Layout Map (`IEEEtran` 2-Column)

```text
+------------------------------------+------------------------------------+
| PAGE 1                             | PAGE 2                             |
| [Title, Authors, Abstract]         | [III. B. Nonlinear AC Physics Loss]|
| I. INTRODUCTION                    |   - Contrast vs. Iliadis &         |
|   - Unobservability & WLS failure  |     Raghuvamsi assumptions         |
|   - The 4 Literature Gaps          | IV. EXPERIMENTAL SETUP             |
|   - Core Contributions             |   - IEEE 33-Bus (16 PMUs, ±10% P/Q)|
| II. RELATED WORK                   |   - Clean vs. Augmented protocols  |
|   - Table I: Comparison against    |   - Table II: Architecture specs   |
|     the 6 reference methodologies  |                                    |
| III. METHODOLOGY                   |                                    |
|   - A. MLP, GNN, TCN Formulations  |                                    |
+------------------------------------+------------------------------------+
| PAGE 3                             | PAGE 4                             |
| V. RESULTS & DISCUSSION            | V. D. Hardware Latency & Trade-offs|
|   - A. OOD Blackout Resilience     |   - Table IV: CUDA inference speed |
|     (Black-Box DL vs. TCN/Aug-MLP) | VI. CONCLUSION & DESIGN RULES      |
|   - Fig. 1: Dropout curve (10-50%) |   - Mapping findings back to the   |
|   - Table III: Master Accuracy     |     6 reference paradigms          |
|   - B. The Topological Horizon     | REFERENCES                         |
|     (Why 3-hop GNN trails MLP)     |   - Schweppe, Iliadis, Raghuvamsi, |
|   - C. The Physics Pareto Cliff    |     Ngo, Hu + core DSSE citations  |
+------------------------------------+------------------------------------+
```

---

## Section-by-Section Paper Blueprint

### Section I: Introduction
* **Motivation:** Modern distribution grids face volatile DERs and sparse PMU instrumentation (~50% coverage). Conventional WLS (Schweppe, 1970) diverges once observability is lost ($>10\%$ telemetry dropout).
* **The Literature Gap:** While data-driven and physics-informed estimators have emerged, prior work either suffers catastrophic OOD collapse under sensor loss (Black-Box DL) or imposes rigid assumptions: exact all-bus injections (Iliadis et al., 2025), observable pre-training with linearized Jacobians (Raghuvamsi, 2024), localized full-PMU zones (Ngo et al., 2024), or heavy SCADA-dependent transmission pipelines (Hu et al., 2025).
* **Contributions:**
  1. Unified single-stage benchmark of global spatial (MLP), topological (Edge-Attributed GNN), and temporal (TCN) paradigms under 50% base PMU sparsity and 10%–50% dynamic dropouts.
  2. Explicit separation of **Clean** (unseen blackout) vs. **Augmented** (dropout-hardened) regimes, quantifying the accuracy–resilience dichotomy.
  3. Discovery of two practical design limits: the **topological horizon** in 17-hop radial feeders and the **physics Pareto cliff** under $\pm 10\%$ pseudo-measurement noise.

### Section II: Related Work & Critical Positioning
* Structured directly around **Table I** (the 6 reference methodologies):
  * *From Iterative WLS to Black-Box DL:* Why non-iterative inference solves latency, but unregularized mapping violates Kirchhoff's laws and collapses under missing inputs.
  * *Physics-Informed Formulations:* Contrasting current-injection PINNs (Iliadis et al.) and constant-Jacobian TCN autoencoders (Raghuvamsi) against our nonlinear AC power-flow residual $\mathcal{L}_{\text{phys}}$ with noisy pseudo-measurements.
  * *Graph & Hybrid Architectures:* Contrasting multi-stage/zonal GNNs (Ngo et al., Hu et al.) against a single-stage `EdgeAttributedGCN` on a radial distribution feeder.

### Section III: Methodology
* **System Formulation:** State $\mathbf{x} = [|\mathbf{V}|, \boldsymbol{\theta}]^T$; measurement vector $\mathbf{z}$ combining sparse PMUs ($|V_i|, \theta_i$) and noisy pseudo-injections ($P_i, Q_i$).
* **Three Paradigms:**
  * **MLP (Global Spatial):** 4 dense layers, 1024 hidden units, ReLU, BatchNorm (78-dim input snapshot). Captures all-to-all bus correlations in $\mathcal{O}(1)$ depth.
  * **GNN (Topological Message Passing):** 3-layer `EdgeAttributedGCN` (`hidden_dims=[128, 128, 128]`, 86,402 params) embedding physical line admittances $[G_{ij}, B_{ij}, |Y_{ij}|, \angle Y_{ij}]$.
  * **TCN (Temporal Causal):** Dilated causal 1D CNN (kernel 3, dilations $1, 2, 4, 8, 16$) with a 12-hour historical receptive field, operating directly on nonlinear AC states without Raghuvamsi's constant-Jacobian decoder.
* **Soft AC Physics Regularization:**
  $$\mathcal{L} = \mathcal{L}_{\text{MSE}} + \lambda \mathcal{L}_{\text{phys}}, \quad \mathbf{S}_{\text{calc}} = \mathbf{V} \odot (\mathbf{V}\mathbf{Y}_{\text{bus}}^T)^*$$
  Unlike Iliadis et al.'s assumption of exact bus injections, $\mathcal{L}_{\text{phys}}$ is treated as a tunable soft constraint against noisy $P_{ps}, Q_{ps}$.

### Section IV: Experimental Setup
* **IEEE 33-Bus Feeder:** 33 buses, 32 branches, 17-hop radial diameter, slack bus at index `0` ($|V|=1.0$ pu, $\theta=0.0$ rad).
* **Telemetry & Noise:** 16 PMUs (~50% coverage); $\sigma_{|V|} = 0.001$ pu, $\sigma_\theta = 0.05^\circ$; $\pm 10\%$ Gaussian forecast noise on unmetered $P, Q$.
* **Evaluation Protocol:**
  * *Clean Regime:* Trained on 100% intact PMUs, tested under 10%–50% random dropout (exposing OOD black-box failure).
  * *Augmented Regime:* Trained with 0%–50% random PMU masking to test learned spatial/temporal imputation.

### Section V: Results & Discussion
* **A. Resolving Black-Box Fragility (Clean vs. Augmented):**
  * *Clean 50% Blackout:* MLP $|V|$ RMSE explodes **+164.0%** ($1.00 \times 10^{-3} \to 2.64 \times 10^{-3}$ pu), validating the Black-Box DL weakness. TCN absorbs the unseen blackout with only **+25.7%** growth ($1.058 \times 10^{-3} \to 1.330 \times 10^{-3}$ pu) via 12-hour load inertia.
  * *Augmented 50% Dropout:* MLP becomes the static accuracy champion (**$0.000516$ pu $|V|$ RMSE**, **$0.0137^\circ$ $\theta$ RMSE**) by exploiting global cross-correlations.
* **B. The Topological Horizon (Why Single-Stage GNN Trails MLP):**
  * At 50% augmented dropout, the 3-layer GNN ($0.001295$ pu) trails MLP. On a 17-hop radial feeder, 3-hop message passing cannot propagate distant PMU information without deeper stacking (which causes oversmoothing)—explaining why prior works (Ngo et al., Hu et al.) resorted to Kalman zones or SCADA baselines.
* **C. The Physics "Pareto Cliff" ($\lambda$ Sweep):**
  * Moderate regularization ($\lambda \in [0.001, 0.1]$; e.g., TCN best $\lambda = 0.01$) acts as an effective physical guardrail.
  * Aggressive enforcement ($\lambda \ge 10.0$) causes accuracy to collapse because the model is forced to fit $\pm 10\%$ forecast noise in unmetered pseudo-measurements—a critical caveat for PINN formulations like Iliadis et al.
* **D. Hardware Inference Latency (CUDA):**
  * MLP: **$0.223\text{ ms}$** | GNN: **$1.157\text{ ms}$** | TCN: **$2.199\text{ ms}$**. All three beat iterative WLS and fit well inside 20–50 ms PMU/SCADA windows.

### Section VI: Conclusion & Design Guidelines
* **For sub-millisecond static accuracy:** Use **Dropout-Augmented MLP** ($0.223\text{ ms}$, $0.000516$ pu), provided outage patterns are represented in training.
* **For unmodeled/surprise communication blackouts:** Deploy **TCN** ($2.199\text{ ms}$, $+25.7\%$ OOD growth) without needing Raghuvamsi's observable pre-training.
* **For radial distribution GNNs:** Shallow 3-hop GNNs face a 17-hop topological horizon; future work should pair multi-scale/global readouts with edge-attributed message passing.
* **For physics-informed loss:** Keep $\lambda \le 0.1$ whenever unmetered buses rely on noisy load forecasts.
