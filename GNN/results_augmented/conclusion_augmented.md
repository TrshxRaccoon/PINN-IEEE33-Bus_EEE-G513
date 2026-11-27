# Edge-Attributed GCN State Estimation Benchmark: Augmented Condition

## 1. Executive Summary & Core Findings
* **Network Backbone**: Edge-Attributed Graph Convolutional Network (`EdgeAttributedGCN`) with physical $Y_{\text{bus}}$ edge conditioning.
* **Training Regime**: **Augmented** (Target: IEEE 33-Bus Radial Feeder, $12.66\text{ kV}$).
* **Lambda Sweep Evaluated**: `[0.0, 0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]` (including extended high penalties: $10, 10^2, 10^3$).
* **Best Physics Penalty ($\lambda$)**: $\lambda = 0.001$.
* **Performance at Severe Communication Failure (50% PMU Dropout)**:
  * Baseline GCN ($\lambda = 0.0$): $|V|\text{ RMSE} = 0.001335644730917599\text{ pu}, \quad \theta\text{ RMSE} = 0.02230752783589278^\circ$.
  * Physics-Informed GCN ($\lambda = 0.001$): $|V|\text{ RMSE} = 0.0012948994475554167\text{ pu}, \quad \theta\text{ RMSE} = 0.022314275172331244^\circ$.
  * **Relative Error Reduction**: $|V|$ RMSE change: **+3.1%**, $\theta$ RMSE change: **-0.0%**.

---

## 2. Comprehensive Test Benchmark Matrix

| Architecture | Model Condition | Penalty $\lambda$ | PMU Dropout | PMU Coverage | $|V|$ RMSE (pu) | $\theta$ RMSE (deg) | $\mathcal{L}_{\text{phys}}$ Residual |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| GCN | Augmented | 0.0 | 10% | 90% | 0.000869 | 0.0189° | 5.527e-04 |
| GCN | Augmented | 0.0 | 20% | 80% | 0.000963 | 0.0193° | 6.156e-04 |
| GCN | Augmented | 0.0 | 30% | 70% | 0.001097 | 0.0202° | 7.082e-04 |
| GCN | Augmented | 0.0 | 40% | 60% | 0.001202 | 0.0211° | 7.666e-04 |
| GCN | Augmented | 0.0 | 50% | 50% | 0.001336 | 0.0223° | 8.935e-04 |
| GCN | Augmented | 0.0001 | 10% | 90% | 0.000858 | 0.0190° | 6.352e-04 |
| GCN | Augmented | 0.0001 | 20% | 80% | 0.000951 | 0.0195° | 7.124e-04 |
| GCN | Augmented | 0.0001 | 30% | 70% | 0.001086 | 0.0203° | 8.276e-04 |
| GCN | Augmented | 0.0001 | 40% | 60% | 0.001188 | 0.0212° | 9.066e-04 |
| GCN | Augmented | 0.0001 | 50% | 50% | 0.001324 | 0.0223° | 1.059e-03 |
| GCN | Augmented | 0.001 | 10% | 90% | 0.000853 | 0.0189° | 7.341e-04 |
| GCN | Augmented | 0.001 | 20% | 80% | 0.000944 | 0.0194° | 8.019e-04 |
| GCN | Augmented | 0.001 | 30% | 70% | 0.001073 | 0.0203° | 9.020e-04 |
| GCN | Augmented | 0.001 | 40% | 60% | 0.001162 | 0.0212° | 9.490e-04 |
| GCN | Augmented | 0.001 | 50% | 50% | 0.001295 | 0.0223° | 1.079e-03 |
| GCN | Augmented | 0.01 | 10% | 90% | 0.000876 | 0.0191° | 6.176e-04 |
| GCN | Augmented | 0.01 | 20% | 80% | 0.000962 | 0.0195° | 6.964e-04 |
| GCN | Augmented | 0.01 | 30% | 70% | 0.001085 | 0.0203° | 8.184e-04 |
| GCN | Augmented | 0.01 | 40% | 60% | 0.001170 | 0.0212° | 8.928e-04 |
| GCN | Augmented | 0.01 | 50% | 50% | 0.001302 | 0.0223° | 1.056e-03 |
| GCN | Augmented | 0.1 | 10% | 90% | 0.000854 | 0.0190° | 5.153e-04 |
| GCN | Augmented | 0.1 | 20% | 80% | 0.000948 | 0.0195° | 5.870e-04 |
| GCN | Augmented | 0.1 | 30% | 70% | 0.001082 | 0.0203° | 6.896e-04 |
| GCN | Augmented | 0.1 | 40% | 60% | 0.001178 | 0.0211° | 7.521e-04 |
| GCN | Augmented | 0.1 | 50% | 50% | 0.001313 | 0.0223° | 8.930e-04 |
| GCN | Augmented | 1.0 | 10% | 90% | 0.000852 | 0.0189° | 3.834e-04 |
| GCN | Augmented | 1.0 | 20% | 80% | 0.000946 | 0.0194° | 4.262e-04 |
| GCN | Augmented | 1.0 | 30% | 70% | 0.001088 | 0.0203° | 4.874e-04 |
| GCN | Augmented | 1.0 | 40% | 60% | 0.001190 | 0.0212° | 5.296e-04 |
| GCN | Augmented | 1.0 | 50% | 50% | 0.001337 | 0.0223° | 6.160e-04 |
| GCN | Augmented | 10.0 | 10% | 90% | 0.000858 | 0.0191° | 1.477e-04 |
| GCN | Augmented | 10.0 | 20% | 80% | 0.000948 | 0.0196° | 1.647e-04 |
| GCN | Augmented | 10.0 | 30% | 70% | 0.001082 | 0.0203° | 2.006e-04 |
| GCN | Augmented | 10.0 | 40% | 60% | 0.001181 | 0.0212° | 2.247e-04 |
| GCN | Augmented | 10.0 | 50% | 50% | 0.001324 | 0.0224° | 2.701e-04 |
| GCN | Augmented | 100.0 | 10% | 90% | 0.000898 | 0.0193° | 5.218e-05 |
| GCN | Augmented | 100.0 | 20% | 80% | 0.000994 | 0.0197° | 5.724e-05 |
| GCN | Augmented | 100.0 | 30% | 70% | 0.001130 | 0.0207° | 6.922e-05 |
| GCN | Augmented | 100.0 | 40% | 60% | 0.001220 | 0.0216° | 7.543e-05 |
| GCN | Augmented | 100.0 | 50% | 50% | 0.001375 | 0.0228° | 9.126e-05 |
| GCN | Augmented | 1000.0 | 10% | 90% | 0.001079 | 0.0201° | 2.994e-05 |
| GCN | Augmented | 1000.0 | 20% | 80% | 0.001171 | 0.0208° | 3.336e-05 |
| GCN | Augmented | 1000.0 | 30% | 70% | 0.001297 | 0.0218° | 4.150e-05 |
| GCN | Augmented | 1000.0 | 40% | 60% | 0.001383 | 0.0228° | 4.562e-05 |
| GCN | Augmented | 1000.0 | 50% | 50% | 0.001533 | 0.0244° | 5.660e-05 |

---

## 3. Engineering Insights & Topological Regularization
1. **Physical Branch Awareness**: Incorporating complex branch admittance $Y_{ij} = G_{ij} + jB_{ij}$ into the GCN message-passing layers informs unobserved buses of neighboring voltage drop dynamics governed by Ohm's Law.
2. **Lambda Penalty Regime**:
   * Low $\lambda \in [10^{-4}, 10^{-2}]$: Gently steers predictions without disturbing data-fitting.
   * Intermediate $\lambda \in [0.1, 1.0]$: Strong physical regularization yielding robust voltage profiles under telemetry loss.
   * Extreme $\lambda \in [10, 10^2, 10^3]$: Evaluates whether over-constraining the model with noisy pseudo-measurements (10% forecast noise) causes optimization tension or bounds unobserved states.
3. **Double Precision Numerics**: All tensors, weights, and losses are computed in `torch.float64` / `torch.complex128`, preventing gradient explosion through the non-linear AC power-flow equations.
