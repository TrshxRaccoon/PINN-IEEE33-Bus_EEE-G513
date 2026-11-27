# Tri-Model Benchmark Summary: Physics-Informed State Estimation

Target Feeder: IEEE 33-Bus Radial Distribution Feeder (`case33bw`)

## 1. Executive Performance at Severe Telemetry Blackout (50% PMU Dropout)

| Model | Best $\lambda$ | Base $|V|$ RMSE (pu) | PI-SE $|V|$ RMSE (pu) | $|V|$ Gain (%) | Base $\theta$ RMSE (deg) | PI-SE $\theta$ RMSE (deg) | $\theta$ Gain (%) | Base $L_{phys}$ | PI-SE $L_{phys}$ |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | 0.1 | 0.5320e-3 | 0.5150e-3 | **+3.20%** | 0.0139° | 0.0138° | **+0.72%** | 4.676e-05 | 3.844e-05 |
| **GNN** | 0.001 | 1.3356e-3 | 1.2949e-3 | **+3.05%** | 0.0223° | 0.0223° | **-0.03%** | 8.935e-04 | 1.079e-03 |
| **TCN** | 0.01 | 0.8820e-3 | 0.8840e-3 | **-0.23%** | 0.0165° | 0.0167° | **-1.21%** | 2.128e-04 | 1.459e-04 |


## 2. Clean Training: Out-of-Distribution Drift & Resilience (50% PMU Dropout)

| Model | Best $\lambda$ | Base $|V|$ RMSE (pu) | PI-SE $|V|$ RMSE (pu) | $|V|$ Gain (%) | Base $\theta$ RMSE (deg) | PI-SE $\theta$ RMSE (deg) | $\theta$ Gain (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **MLP** | 0.1 | 2.0380e-3 | 1.7260e-3 | **+15.31%** | 0.0271° | 0.0225° | **+16.97%** |
| **GNN** | 10.0 | 2.1170e-3 | 2.1170e-3 | **+0.00%** | 0.0351° | 0.0351° | **-0.00%** |
| **TCN** | 1000.0 | 1.3300e-3 | 1.3420e-3 | **-0.90%** | 0.0250° | 0.0209° | **+16.40%** |
