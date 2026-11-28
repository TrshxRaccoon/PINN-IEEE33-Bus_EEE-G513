# Interactive Grid State Estimation Visualiser

This module provides a web-based **Streamlit interactive visualizer** for inspecting Physics-Informed Distribution System State Estimation (PI-DSSE) across the IEEE 33-bus radial distribution grid.

---

## Directory & File Roles

| File / Folder | Role & Description |
| :--- | :--- |
| `app.py` | **Streamlit Web Application**: An interactive multi-tab dashboard that renders: (1) Single-line diagram of the IEEE 33-bus feeder with live PMU status indicators (Active vs. Dropped), (2) Real-time bus voltage profiles ($|V|$ and $\theta$) comparing Ground Truth against model predictions, (3) Error heatmap and hop-distance analysis from the substation slack bus, and (4) Interactive parameter sliders to simulate arbitrary PMU blackout scenarios and load scaling. |
| `precomputed_data.csv` | **Precomputed Benchmark Metric Database**: Stores pre-extracted snapshot performance metrics across models (MLP, GNN, TCN), training regimes (Clean vs. Augmented), and dropout levels (10% to 50%), enabling fast online rendering without requiring live GPU re-evaluation. |

---

## Features & Visualizations

* **Topology View**: Displays the radial tree structure (17 hops from Bus 0 to Bus 17) with color-coded node states (slack bus, metered PMU, active PMU, and dropped PMU).
* **Model Comparison**: Side-by-side voltage magnitude and phase angle plots comparing Ground Truth, Baseline models ($\lambda = 0$), and Physics-Informed models ($\lambda > 0$).
* **Spatial Error Analysis**: Graphs estimation residuals as a function of topological distance (hop count) from the substation, visually demonstrating the GNN topological horizon and MLP spatial correlations.
* **Dropout Simulator**: Allows users to dynamically toggle PMU availability and observe instantaneous state reconstruction quality.

---

## How to Run

1. Ensure prerequisites are installed:
   ```bash
   pip install streamlit plotly pandas numpy
   ```
2. Launch the Streamlit application:
   ```bash
   streamlit run app.py
   ```
3. Open your browser at `http://localhost:8501`.
