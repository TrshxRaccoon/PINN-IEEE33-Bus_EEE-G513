# Visual Interactive Demo: Data Specification & Integration Guide

**Target System:** IEEE 33-Bus Power Distribution Network  
**Purpose:** Precomputed dataset and interface specification for the interactive 2D State Estimation visualizer.  
**Audience:** Front-end engineers, data engineers, and AI implementation agents.

---

## 1. Executive Summary

This project demonstrates Power System State Estimation (PSSE) using Physics-Informed Neural Networks (PINN) and Graph Neural Networks (GNN) on the benchmark **IEEE 33-bus radial distribution grid**.

To ensure instantaneous rendering and zero model-inference latency in the user interface (e.g., Streamlit, Dash, or web/D3.js), **all model inferences, state predictions, and evaluation metrics are precomputed** across all experimental permutations.

The interactive app visualizes:
1. A **2D single-line diagram** of the IEEE 33-bus system.
2. Color-coded nodes reflecting either **Physical Voltages** ($V_m, V_a$) or **Estimation Errors** (RMSE, Physics Violations).
3. Visual indicators (e.g., glowing circular rings or distinct borders) denoting active PMU telemetry vs. unmonitored / sensor-dropout buses.
4. Real-time updates as users adjust dataset conditions, PMU coverage, and physics loss weight ($\lambda$).

---

## 2. Directory Structure

```text
ML_Project/Visual_Demo/
├── sample_precomputed_data.csv             # Reference CSV with consolidated precomputed slices
└── DATA_SPECIFICATION_AND_INTEGRATION_GUIDE.md # This architecture specification
```

---

## 3. Consolidated CSV Data Schema

The precomputed dataset stores rows in a **tidy (long-form)** structure. Each row contains the complete state and error profile for **one bus** under a specific experimental condition:

$$\text{Row Identifier: } (\text{condition}, \text{model}, \text{pmu\_coverage}, \lambda, \text{node})$$

### Data Dictionary

| Column Header | Data Type | Units / Format | Allowed Values / Examples | Description |
| :--- | :--- | :--- | :--- | :--- |
| **`condition`** | `str` | Nominal | `Clean`, `Augmented` | Training regime of the ML model.<br>• **`Clean`**: Baseline models trained on static dropout.<br>• **`Augmented`**: Models trained with dynamic dropout augmentation.<br>⚠️ **Important:** `Ground_Truth` and `Pseudo` exist **ONLY** under `Clean`. |
| **`model`** | `str` | Nominal | `Ground_Truth`, `Pseudo`, `MLP`, `GCN`, `GNN` | State estimation algorithm or reference source. |
| **`pmu_coverage`** | `int` | Percentage ($\%$) | `50`, `60`, `70`, `80`, `90`, `100` | Active PMU telemetry coverage percentage across the grid. |
| **`lambda`** | `str` / `float` | Dimensionless | `0.0`, `0.0001`, `0.001`, `0.01`, `0.1`, `1.0`, `10.0`, `100.0`, `N/A` | Physics regularization penalty weight ($\lambda$). Set to `N/A` for `Ground_Truth` and `Pseudo`. |
| **`node`** | `int` | Index ($1$-based) | `1` to `33` | Bus number in the standard IEEE 33-bus network. |
| **`is_pmu`** | `int` | Binary flag | `0` or `1` | `1` if this bus has active PMU telemetry; `0` if unmonitored or dropped out. |
| **`vm_true`** | `float` | per-unit ($\text{p.u.}$) | $\approx 0.9000 \text{ to } 1.0000$ | Ground truth voltage magnitude. |
| **`va_true`** | `float` | Degrees ($^\circ$) | $\approx -2.500 \text{ to } 0.000^\circ$ | Ground truth voltage angle (slack bus is $0.0^\circ$). |
| **`vm_val`** | `float` | per-unit ($\text{p.u.}$) | $\approx 0.9000 \text{ to } 1.0500$ | Model-predicted (or actual) voltage magnitude. |
| **`va_val`** | `float` | Degrees ($^\circ$) | $\approx -3.000 \text{ to } +0.500^\circ$ | Model-predicted (or actual) voltage angle. |
| **`vm_rmse`** | `float` | per-unit ($\text{p.u.}$) | $\ge 0.0000$ | Root Mean Squared Error of voltage magnitude at this node. |
| **`va_rmse`** | `float` | Degrees ($^\circ$) | $\ge 0.0000$ | Root Mean Squared Error of voltage angle at this node. |
| **`physics_residual`** | `float` | per-unit ($\text{p.u.}$) | $\ge 0.0000$ | Kirchhoff / AC power flow balance violation $\|f(V)\|$ at this bus. |

---

## 4. UI Control to Data Mapping & Interactivity Rules

### Rule 1: Clean/Augmented Availability
* **`Ground_Truth` and `Pseudo` are physical dataset references, NOT trained models.**
* Therefore, selecting `Ground_Truth` or `Pseudo` in the **Data Selector** must **automatically lock / disable the Clean/Augmented selector to "Clean"** (or hide the toggle).
* The **Augmented** toggle is only active when an ML model is selected (`MLP`, `GCN`, `GNN`).

### Rule 2: Lambda Slider Availability
* The $\lambda$ slider controls physics loss weighting during training.
* For `Ground_Truth` and `Pseudo`, the $\lambda$ slider must be **disabled** (queries `lambda == 'N/A'`).
* For ML models (`MLP`, `GCN`, `GNN`), $\lambda = 0.0$ represents the purely data-driven baseline, and $\lambda > 0$ represents PINN.

### Rule 3: View Mode Toggle (Voltages vs. Errors)
* **Mode A: Physical Voltages** $\rightarrow$ Color maps to `vm_val` ($V_m$) or `va_val` ($V_a$).
* **Mode B: Estimation Errors** $\rightarrow$ Color maps to `vm_rmse`, `va_rmse`, or `physics_residual`.
  * *Note:* If `Ground_Truth` is selected under Error Mode, errors across all 33 nodes are zero.

### UI Interaction State Matrix

| Selected Model | Condition Selector | Lambda Slider | View Mode: Voltages | View Mode: Errors |
| :--- | :--- | :--- | :--- | :--- |
| **`Ground_Truth`** | **Locked to `Clean`** | **Disabled (`N/A`)** | Active (`vm_true`, `va_true`) | Active (All values = 0) |
| **`Pseudo`** | **Locked to `Clean`** | **Disabled (`N/A`)** | Active (Pseudo states) | Active (Pseudo baseline error) |
| **`MLP`** | Enabled (`Clean` / `Aug`) | Enabled ($0 \dots 10^3$) | Active (`vm_val`, `va_val`) | Active (`vm_rmse`, `va_rmse`, residual) |
| **`GCN`** | Enabled (`Clean` / `Aug`) | Enabled ($0 \dots 10^3$) | Active (`vm_val`, `va_val`) | Active (`vm_rmse`, `va_rmse`, residual) |
| **`GNN`** | Enabled (`Clean` / `Aug`) | Enabled ($0 \dots 10^3$) | Active (`vm_val`, `va_val`) | Active (`vm_rmse`, `va_rmse`, residual) |

---

## 5. IEEE 33-Bus 2D Topology Reference

Use these radial feeder definitions and normalized coordinates to construct the 2D layout:

### Radial Branch Connections (32 Branches, 1-Indexed)

* **Main Feeder Backbone (Buses 1–18):**  
  `(1-2), (2-3), (3-4), (4-5), (5-6), (6-7), (7-8), (8-9), (9-10), (10-11), (11-12), (12-13), (13-14), (14-15), (15-16), (16-17), (17-18)`
* **Lateral Branch 1 (Originates at Bus 2):**  
  `(2-19), (19-20), (20-21), (21-22)`
* **Lateral Branch 2 (Originates at Bus 3):**  
  `(3-23), (23-24), (24-25)`
* **Lateral Branch 3 (Originates at Bus 6):**  
  `(6-26), (26-27), (27-28), (28-29), (29-30), (30-31), (31-32), (32-33)`

### Static 2D Node Placement Coordinates $(X, Y)$

```python
IEEE33_BUS_COORDS = {
    # Main trunk (horizontal line at Y=0)
    1: (0, 0),   2: (1, 0),   3: (2, 0),   4: (3, 0),   5: (4, 0),   6: (5, 0),
    7: (6, 0),   8: (7, 0),   9: (8, 0),  10: (9, 0),  11: (10, 0), 12: (11, 0),
   13: (12, 0), 14: (13, 0), 15: (14, 0), 16: (15, 0), 17: (16, 0), 18: (17, 0),
    
    # Lateral 1 (branching upwards from Bus 2 at Y=+1 to +4)
   19: (1, 1),  20: (1, 2),  21: (1, 3),  22: (1, 4),
    
    # Lateral 2 (branching downwards from Bus 3 at Y=-1 to -3)
   23: (2, -1), 24: (2, -2), 25: (2, -3),
    
    # Lateral 3 (branching upwards from Bus 6 at Y=+1 to +7)
   26: (5, 1),  27: (5, 2),  28: (6, 2),  29: (7, 2),  30: (8, 2),  31: (8, 3),
   32: (8, 4),  33: (9, 4)
}
```

---

## 6. Visual Presentation Guidelines

1. **PMU Status Indicator:**
   * If `is_pmu == 1`: Render a bright cyan or lime circular border/ring around the bus node.
   * If `is_pmu == 0`: Render with a muted dashed border or smaller diameter to highlight sensor dropout.
2. **Color Palettes:**
   * **Physical Voltages ($V_m$):** Diverging palette (e.g. `RdYlGn` or `coolwarm`) centered at $1.0\text{ p.u.}$ Red indicates severe undervoltage sag (<$0.95\text{ p.u.}$).
   * **Errors / RMSE / Residuals:** Sequential warm palette (e.g. `YlOrRd` or `Plasma`). Darker colors highlight problem buses where estimation drifted.
3. **Hover Tooltip Example:**
   ```text
   Bus: 18 (End of Main Feeder)
   PMU Telemetry: Active (or Dropped)
   Voltage Magnitude: 0.9234 p.u. (True: 0.9250 p.u.)
   Vm RMSE: 0.0016 p.u.
   Physics Residual: 0.00021 p.u.
   ```

---

## 7. Sample Implementation Snippet (Streamlit Example)

```python
import streamlit as st
import pandas as pd

# Load precomputed data
df = pd.read_csv("Visual_Demo/sample_precomputed_data.csv")

# 1. Model Selector
model = st.sidebar.selectbox("Data / Model Selector", ["Ground_Truth", "Pseudo", "MLP", "GCN", "GNN"])

# 2. Condition Selector (Disabled/locked to Clean for Ground_Truth & Pseudo)
if model in ["Ground_Truth", "Pseudo"]:
    condition = "Clean"
    st.sidebar.selectbox("Condition", ["Clean"], disabled=True, help="Augmented condition only applies to trained ML models.")
else:
    condition = st.sidebar.selectbox("Condition", ["Clean", "Augmented"])

# 3. PMU Coverage Slider
coverage = st.sidebar.select_slider("PMU Coverage (%)", options=[50, 75, 100])

# 4. Lambda Slider (Disabled for Ground_Truth & Pseudo)
if model in ["Ground_Truth", "Pseudo"]:
    lam = "N/A"
    st.sidebar.select_slider("Physics Weight (Lambda)", options=["N/A"], disabled=True)
else:
    lam = st.sidebar.select_slider("Physics Weight (Lambda)", options=["0.0", "0.001", "0.01", "0.1", "1.0", "10.0", "100.0"])

# 5. View Mode Toggle
view_mode = st.radio("View Mode", ["Physical Voltages", "Estimation Errors"])
if view_mode == "Physical Voltages":
    param = st.selectbox("Parameter", ["vm_val", "va_val"], format_func=lambda x: "Voltage Magnitude (Vm)" if x == "vm_val" else "Voltage Angle (Va)")
else:
    param = st.selectbox("Parameter", ["vm_rmse", "va_rmse", "physics_residual"], format_func=lambda x: "Vm RMSE" if x == "vm_rmse" else ("Va RMSE" if x == "va_rmse" else "Physics Residual"))

# 6. Filter Query (Always yields exactly 33 rows)
filtered_33_buses = df[
    (df["condition"] == condition) &
    (df["model"] == model) &
    (df["pmu_coverage"] == coverage) &
    (df["lambda"] == lam)
].sort_values("node")

# filtered_33_buses now feeds directly to the 2D network diagram renderer!
```
