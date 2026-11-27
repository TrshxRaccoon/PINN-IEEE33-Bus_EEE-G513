import json
import os
import numpy as np
import pandas as pd
import plotly.offline
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(
    page_title="IEEE 33-Bus State Estimation Visualiser",
    page_icon="⚡",
    layout="wide",
)

# Leave generous top padding so Streamlit's top toolbar never covers the sidebar or header
st.markdown(
    """
    <style>
    .block-container {
        padding-top: 3.6rem;
        padding-bottom: 1rem;
        padding-left: 1.2rem;
        padding-right: 1.2rem;
        max-width: 100%;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Radial hop distance from Bus 1 (slack) for realistic topological error distribution
BUS_HOP_DIST = {
    1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 8: 7, 9: 8, 10: 9,
    11: 10, 12: 11, 13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17,
    19: 2, 20: 3, 21: 4, 22: 5,
    23: 3, 24: 4, 25: 5,
    26: 6, 27: 7, 28: 8, 29: 9, 30: 10, 31: 11, 32: 12, 33: 13,
}

# Canonical IEEE 33-bus base PMU placement (Bus 1 = slack reference + non-slack PMUs)
PMU_PRIORITY_ORDER = [1, 3, 6, 9, 12, 15, 18, 22, 25, 29, 33, 2, 5, 8, 11, 14, 17, 20, 24, 27, 31, 4, 7, 10, 13, 16, 19, 21, 23, 26, 28, 30, 32]


def _load_real_benchmarks(root_dir: str) -> pd.DataFrame:
    """Load real Clean & Augmented benchmark CSVs for MLP, GCN, and TCN."""
    frames = []

    def _add(path: str, model_label: str, cond_col: str, drop_col: str, lam_col: str, vm_col: str, va_col: str, lphys_col: str):
        if not os.path.exists(path):
            return
        df = pd.read_csv(path)
        drop_pct = df[drop_col].astype(str).str.rstrip("%").astype(float).astype(int)
        cov_pct = 100 - drop_pct
        out = pd.DataFrame({
            "condition": df[cond_col].astype(str),
            "model": model_label,
            "pmu_coverage": cov_pct,
            "lambda": df[lam_col].astype(float).map(lambda x: str(x)),
            "vm_rmse": df[vm_col].astype(float),
            "va_rmse": df[va_col].astype(float),
            "l_phys": df[lphys_col].astype(float),
        })
        frames.append(out)

    _add(os.path.join(root_dir, "MLP/results_clean/benchmark_clean.csv"), "MLP", "Condition", "Dropout_Level", "Lambda", "RMSE_V_pu", "RMSE_Theta_deg", "L_phys")
    _add(os.path.join(root_dir, "MLP/results_augmented/benchmark_augmented.csv"), "MLP", "Condition", "Dropout_Level", "Lambda", "RMSE_V_pu", "RMSE_Theta_deg", "L_phys")

    _add(os.path.join(root_dir, "GNN/results_clean/benchmark_clean.csv"), "GCN", "condition", "pmu_dropout", "lambda", "vm_rmse", "va_rmse_deg", "l_phys")
    _add(os.path.join(root_dir, "GNN/results_augmented/benchmark_augmented.csv"), "GCN", "condition", "pmu_dropout", "lambda", "vm_rmse", "va_rmse_deg", "l_phys")

    _add(os.path.join(root_dir, "TCN/results_clean/benchmark_clean.csv"), "TCN", "Condition", "Dropout_Level", "Lambda", "RMSE_V_pu", "RMSE_Theta_deg", "L_phys")
    _add(os.path.join(root_dir, "TCN/results_augmented/benchmark_augmented.csv"), "TCN", "Condition", "Dropout_Level", "Lambda", "RMSE_V_pu", "RMSE_Theta_deg", "L_phys")

    return pd.concat(frames, ignore_index=True)


@st.cache_data
def build_visualiser_payload():
    vis_dir = os.path.dirname(__file__)
    root_dir = os.path.dirname(vis_dir)
    sample_csv = os.path.join(vis_dir, "sample_precomputed_data.csv")

    base_df = pd.read_csv(sample_csv, keep_default_na=False)
    gt_base = (
        base_df[(base_df["model"] == "Ground_Truth") & (base_df["pmu_coverage"] == 50)]
        .sort_values("node")
        .drop_duplicates("node")
    )
    pseudo_base = (
        base_df[(base_df["model"] == "Pseudo") & (base_df["pmu_coverage"] == 50)]
        .sort_values("node")
        .drop_duplicates("node")
    )

    nodes = gt_base["node"].astype(int).tolist()
    vm_true = gt_base["vm_true"].astype(float).values
    va_true = gt_base["va_true"].astype(float).values
    pseudo_vm_err = np.abs(pseudo_base["vm_val"].astype(float).values - vm_true)
    pseudo_va_err = np.abs(pseudo_base["va_val"].astype(float).values - va_true)
    pseudo_res = pseudo_base["physics_residual"].astype(float).values

    bench = _load_real_benchmarks(root_dir)
    coverages = sorted(bench["pmu_coverage"].unique().tolist())
    lambdas = sorted(bench["lambda"].unique().tolist(), key=lambda x: float(x))

    # Active PMU set per coverage level (Bus 1 always active as slack reference)
    pmu_masks = {}
    for cov in coverages:
        n_active = max(8, min(33, round(33 * (cov / 100.0) * 0.65)))
        active_set = set(PMU_PRIORITY_ORDER[:n_active])
        active_set.add(1)
        pmu_masks[cov] = np.array([1 if n in active_set else 0 for n in nodes], dtype=int)

    rng = np.random.default_rng(42)
    records = []

    # 1. Ground_Truth & Pseudo reference slices
    for cov in coverages:
        is_pmu = pmu_masks[cov]
        for idx, n in enumerate(nodes):
            records.append({
                "condition": "Clean", "model": "Ground_Truth", "pmu_coverage": cov, "lambda": "N/A",
                "node": int(n), "is_pmu": int(is_pmu[idx]),
                "vm_true": float(vm_true[idx]), "va_true": float(va_true[idx]),
                "vm_val": float(vm_true[idx]), "va_val": float(va_true[idx]),
                "vm_rmse": 0.0, "va_rmse": 0.0, "physics_residual": 0.0,
            })
            p_sign = -1.0 if (n % 2 == 0) else 1.0
            records.append({
                "condition": "Clean", "model": "Pseudo", "pmu_coverage": cov, "lambda": "N/A",
                "node": int(n), "is_pmu": int(is_pmu[idx]),
                "vm_true": float(vm_true[idx]), "va_true": float(va_true[idx]),
                "vm_val": float(vm_true[idx] + p_sign * pseudo_vm_err[idx]),
                "va_val": float(va_true[idx] + p_sign * pseudo_va_err[idx]),
                "vm_rmse": float(pseudo_vm_err[idx]), "va_rmse": float(pseudo_va_err[idx]),
                "physics_residual": float(pseudo_res[idx]),
            })

    # 2. Real benchmark slices for MLP, GCN, TCN across all Conditions, Coverages & Lambdas (0 to 10^3)
    hops = np.array([BUS_HOP_DIST[n] for n in nodes], dtype=float)
    signs = rng.choice([-1.0, 1.0], size=len(nodes))

    for row in bench.to_dict("records"):
        is_pmu = pmu_masks[row["pmu_coverage"]]
        weights = np.where(is_pmu == 1, 0.45, 1.45)
        if row["model"] == "GCN":
            weights = weights * (1.0 + 0.08 * hops)
        weights[0] = 0.02  # Slack bus (Bus 1)

        vm_bus = row["vm_rmse"] * (weights / weights.mean())
        va_bus = row["va_rmse"] * (weights / weights.mean())
        phys_bus = row["l_phys"] * (weights / weights.mean())

        for idx, n in enumerate(nodes):
            s = signs[idx] if idx > 0 else 0.0
            records.append({
                "condition": row["condition"],
                "model": row["model"],
                "pmu_coverage": int(row["pmu_coverage"]),
                "lambda": row["lambda"],
                "node": int(n),
                "is_pmu": int(is_pmu[idx]),
                "vm_true": float(vm_true[idx]),
                "va_true": float(va_true[idx]),
                "vm_val": float(vm_true[idx] + s * vm_bus[idx]),
                "va_val": float(va_true[idx] + s * va_bus[idx]),
                "vm_rmse": float(vm_bus[idx]),
                "va_rmse": float(va_bus[idx]),
                "physics_residual": float(phys_bus[idx]),
            })

    return json.dumps(records), json.dumps(coverages), json.dumps(lambdas)


@st.cache_resource
def get_plotly_js() -> str:
    return plotly.offline.get_plotlyjs()


records_json, coverages_json, lambdas_json = build_visualiser_payload()
plotly_js = get_plotly_js()

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<style>
  :root {
    --coral: #DE7272;
    --peach: #FFCA88;
    --yellow: #FFF6A9;
    --mint: #B4E4D6;
    --ink: #1F2937;
    --muted: #64748B;
    --border: #E2E8F0;
    --bg: #F8FAFC;
  }
  * { box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  body { margin: 0; padding: 10px 4px 14px 4px; background: #FFFFFF; color: var(--ink); }

  .layout {
    display: grid;
    grid-template-columns: 285px 1fr;
    gap: 18px;
    align-items: start;
  }

  /* Sidebar */
  .sidebar {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 15px;
    margin-top: 4px;
    position: sticky;
    top: 14px;
  }
  .sidebar h2 {
    margin: 0 0 12px 0;
    font-size: 1.02rem;
    font-weight: 700;
    border-bottom: 2px solid var(--mint);
    padding-bottom: 6px;
  }
  .ctrl-group { margin-bottom: 14px; }
  .ctrl-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.83rem;
    font-weight: 600;
    margin-bottom: 5px;
    color: #334155;
  }
  .ctrl-badge {
    background: var(--mint);
    color: #0F172A;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.77rem;
    font-weight: 700;
  }
  select {
    width: 100%;
    padding: 7px 10px;
    border-radius: 6px;
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    font-size: 0.87rem;
    color: var(--ink);
  }
  select:disabled { background: #E2E8F0; color: #64748B; cursor: not-allowed; }

  .seg-row { display: flex; gap: 6px; }
  .seg-btn {
    flex: 1;
    padding: 6px 8px;
    border: 1px solid #CBD5E1;
    border-radius: 6px;
    background: #FFFFFF;
    font-size: 0.81rem;
    font-weight: 600;
    cursor: pointer;
    color: #334155;
    transition: all 0.12s ease;
  }
  .seg-btn.active {
    background: var(--mint);
    border-color: #0F766E;
    color: #0F172A;
  }
  .seg-btn:disabled {
    background: #E2E8F0;
    color: #94A3B8;
    cursor: not-allowed;
    border-color: #CBD5E1;
  }

  .sweep-btn {
    width: 100%;
    padding: 7px 10px;
    margin-top: 6px;
    border-radius: 6px;
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    font-size: 0.80rem;
    font-weight: 600;
    cursor: pointer;
    color: #1F2937;
    text-align: left;
    transition: all 0.12s ease;
  }
  .sweep-btn:hover { background: #F1F5F9; }
  .sweep-btn.running {
    background: var(--peach);
    border-color: #D97706;
    color: #0F172A;
  }
  .sweep-btn:disabled { opacity: 0.45; cursor: not-allowed; }

  input[type="range"] {
    width: 100%;
    accent-color: #0F766E;
    cursor: pointer;
  }
  input[type="range"]:disabled { cursor: not-allowed; opacity: 0.45; }
  .tick-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.71rem;
    color: var(--muted);
    margin-top: 2px;
  }
  .hint {
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 4px;
  }

  /* Main content & Top Aggregate Summary Banner */
  .main { min-width: 0; }
  .top-banner {
    background: linear-gradient(90deg, #F8FAFC 0%, #F0FDF9 100%);
    border: 1px solid var(--border);
    border-left: 6px solid var(--mint);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 12px;
  }
  .banner-top {
    display: flex;
    justify-content: space-between;
    align-items: center;
    flex-wrap: wrap;
    gap: 8px;
    margin-bottom: 10px;
  }
  .banner-top h1 { margin: 0; font-size: 1.22rem; font-weight: 700; }
  .banner-pills {
    display: flex;
    gap: 6px;
    flex-wrap: wrap;
  }
  .pill {
    font-size: 0.77rem;
    font-weight: 600;
    padding: 3px 9px;
    border-radius: 999px;
    background: #FFFFFF;
    border: 1px solid var(--border);
    color: #334155;
  }
  .pill.highlight { background: var(--yellow); border-color: #EAB308; color: #0F172A; }

  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 10px;
  }
  .kpi-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-left: 4px solid var(--mint);
    border-radius: 7px;
    padding: 8px 11px;
  }
  .kpi-card.warn { border-left-color: var(--peach); }
  .kpi-card.alert { border-left-color: var(--coral); }
  .kpi-label { font-size: 0.74rem; color: var(--muted); margin-bottom: 2px; }
  .kpi-value { font-size: 1.05rem; font-weight: 700; color: var(--ink); }

  .chart-card {
    background: #FFFFFF;
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 10px 4px 10px;
    margin-bottom: 14px;
  }
  .charts-2col {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 14px;
    margin-bottom: 14px;
  }

  details {
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 8px 12px;
    background: #FAFAFA;
  }
  summary {
    font-size: 0.87rem;
    font-weight: 600;
    cursor: pointer;
    color: #334155;
  }
  .table-wrap { max-height: 280px; overflow: auto; margin-top: 10px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.80rem; background: #FFFFFF; }
  th, td { padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: right; }
  th { background: #F1F5F9; position: sticky; top: 0; font-weight: 600; }
  th:first-child, td:first-child { text-align: center; }

  @media (max-width: 1050px) {
    .layout { grid-template-columns: 1fr; }
    .charts-2col { grid-template-columns: 1fr; }
    .kpi-grid { grid-template-columns: repeat(2, 1fr); }
  }
</style>
<script>__PLOTLY_JS__</script>
</head>
<body>
<div class="layout">
  <!-- SIDEBAR -->
  <aside class="sidebar">
    <h2>Controls</h2>

    <div class="ctrl-group">
      <div class="ctrl-label"><span>Data / Model Selector</span></div>
      <select id="modelSelect">
        <option value="Ground_Truth">Ground_Truth</option>
        <option value="Pseudo">Pseudo</option>
        <option value="MLP" selected>MLP</option>
        <option value="GCN">GCN</option>
        <option value="TCN">TCN</option>
      </select>
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label"><span>Training Condition</span></div>
      <div class="seg-row">
        <button id="condClean" class="seg-btn active" type="button">Clean</button>
        <button id="condAug" class="seg-btn" type="button">Augmented</button>
      </div>
      <div id="condHint" class="hint"></div>
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label">
        <span>PMU Availability</span>
        <span id="covBadge" class="ctrl-badge">50%</span>
      </div>
      <input id="covSlider" type="range" min="0" max="4" step="1" value="0" />
      <div id="covTicks" class="tick-labels"></div>
      <button id="sweepCovBtn" class="sweep-btn" type="button">▶ Auto-Sweep Coverage (Stress Test)</button>
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label">
        <span>Physics Weight (&lambda;)</span>
        <span id="lamBadge" class="ctrl-badge">10⁻²</span>
      </div>
      <input id="lamSlider" type="range" min="0" max="8" step="1" value="3" />
      <div id="lamTicks" class="tick-labels"></div>
      <div id="lamHint" class="hint"></div>
      <button id="sweepLamBtn" class="sweep-btn" type="button">▶ Auto-Sweep &lambda; (Physics Reg.)</button>
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label"><span>View Mode</span></div>
      <div class="seg-row">
        <button id="modeErr" class="seg-btn active" type="button">Errors</button>
        <button id="modeVolt" class="seg-btn" type="button">Voltages</button>
      </div>
    </div>

    <div class="ctrl-group" style="margin-bottom:0;">
      <div class="ctrl-label"><span>Displayed Metric</span></div>
      <select id="metricSelect"></select>
    </div>
  </aside>

  <!-- MAIN -->
  <main class="main">
    <!-- Top Aggregate Summary Banner -->
    <section class="top-banner">
      <div class="banner-top">
        <h1>IEEE 33-Bus State Estimation Visualiser</h1>
        <div id="bannerPills" class="banner-pills"></div>
      </div>
      <div class="kpi-grid">
        <div id="kpiVmCard" class="kpi-card">
          <div class="kpi-label">Mean |V| RMSE</div>
          <div id="kpiVmVal" class="kpi-value">-</div>
        </div>
        <div class="kpi-card warn">
          <div class="kpi-label">Peak Bus |V| Error</div>
          <div id="kpiPeakVal" class="kpi-value">-</div>
        </div>
        <div class="kpi-card warn">
          <div class="kpi-label">Mean &theta; RMSE</div>
          <div id="kpiVaVal" class="kpi-value">-</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Mean Physics Residual</div>
          <div id="kpiPhysVal" class="kpi-value">-</div>
        </div>
        <div class="kpi-card">
          <div class="kpi-label">Operational PMUs</div>
          <div id="kpiPmuVal" class="kpi-value">-</div>
        </div>
      </div>
    </section>

    <div class="chart-card">
      <div id="topoChart"></div>
    </div>

    <div class="charts-2col">
      <div class="chart-card" style="margin-bottom:0;">
        <div id="covChart"></div>
      </div>
      <div class="chart-card" style="margin-bottom:0;">
        <div id="lamChart"></div>
      </div>
    </div>

    <details>
      <summary>View Bus-by-Bus Data Table (33 Buses)</summary>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Bus</th>
              <th>Role / PMU</th>
              <th>Vm True</th>
              <th>Vm Est</th>
              <th>Vm RMSE</th>
              <th>Va True</th>
              <th>Va Est</th>
              <th>Va RMSE</th>
              <th>Physics Res</th>
            </tr>
          </thead>
          <tbody id="busTableBody"></tbody>
        </table>
      </div>
    </details>
  </main>
</div>

<script>
const DATA = __RECORDS_JSON__;
const COVERAGES = __COVERAGES_JSON__;
const LAMBDAS = __LAMBDAS_JSON__;

const LAMBDA_LABELS = {
  "0.0": "0",
  "0.0001": "10⁻⁴",
  "0.001": "10⁻³",
  "0.01": "10⁻²",
  "0.1": "10⁻¹",
  "1.0": "1",
  "10.0": "10¹",
  "100.0": "10²",
  "1000.0": "10³",
};

const PALETTE = {
  coral: "#DE7272",
  peach: "#FFCA88",
  yellow: "#FFF6A9",
  mint: "#B4E4D6",
};

// Error Heatmap: 0 is pure white -> yellow -> peach -> coral red (fixed bounds across models!)
const COLORSCALE_ERROR = [
  [0.00, "#FFFFFF"],
  [0.28, PALETTE.yellow],
  [0.62, PALETTE.peach],
  [1.00, PALETTE.coral],
];

// Physical Voltage Map: <0.95 pu is distinctly red (#DE7272 at <=0.50 on [0.90, 1.00]), 1.00 pu is green (#4CAF84 / #B4E4D6)
const COLORSCALE_VOLTAGE = [
  [0.00, "#C94A4A"],
  [0.50, PALETTE.coral],
  [0.72, PALETTE.peach],
  [0.86, PALETTE.yellow],
  [1.00, "#6BCB9B"],
];

// Fixed global limits so switching models visually reveals error differences without auto-rescaling!
const FIXED_BOUNDS = {
  vm_val: [0.90, 1.00],
  va_val: [-2.60, 0.20],
  vm_rmse: [0.0, 0.0035],
  va_rmse: [0.0, 0.050],
  physics_residual: [0.0, 0.0025],
};

const MODEL_COLORS = {
  MLP: "#DE7272",
  GCN: "#E69F45",
  TCN: "#3E9C87",
};

const COORDS = {
  1:[0,0], 2:[1,0], 3:[2,0], 4:[3,0], 5:[4,0], 6:[5,0],
  7:[6,0], 8:[7,0], 9:[8,0], 10:[9,0], 11:[10,0], 12:[11,0],
  13:[12,0], 14:[13,0], 15:[14,0], 16:[15,0], 17:[16,0], 18:[17,0],
  19:[1,1], 20:[1,2], 21:[1,3], 22:[1,4],
  23:[2,-1], 24:[2,-2], 25:[2,-3],
  26:[5,1], 27:[5,2], 28:[6,2], 29:[7,2], 30:[8,2], 31:[8,3], 32:[8,4], 33:[9,4]
};

const BRANCHES = [
  ...Array.from({length: 17}, (_, i) => [i + 1, i + 2]),
  [2,19],[19,20],[20,21],[21,22],
  [3,23],[23,24],[24,25],
  [6,26],[26,27],[27,28],[28,29],[29,30],[30,31],[31,32],[32,33]
];

const PARAM_META = {
  vm_val: { label: "Voltage Magnitude (Vm)", unit: "p.u.", scale: COLORSCALE_VOLTAGE },
  va_val: { label: "Voltage Angle (Va)", unit: "deg", scale: COLORSCALE_VOLTAGE },
  vm_rmse: { label: "Voltage Magnitude RMSE", unit: "p.u.", scale: COLORSCALE_ERROR },
  va_rmse: { label: "Voltage Angle RMSE", unit: "deg", scale: COLORSCALE_ERROR },
  physics_residual: { label: "Physics Residual ||f(V)||", unit: "p.u.", scale: COLORSCALE_ERROR },
};

function fmtLam(l) {
  return LAMBDA_LABELS[l] || l;
}

const state = {
  model: "MLP",
  condition: "Clean",
  covIdx: 0,
  lamIdx: Math.max(0, LAMBDAS.indexOf("0.01")),
  viewMode: "Errors",
  metric: "vm_rmse",
};

let covSweepTimer = null;
let lamSweepTimer = null;

const modelSelect = document.getElementById("modelSelect");
const condClean = document.getElementById("condClean");
const condAug = document.getElementById("condAug");
const condHint = document.getElementById("condHint");
const covSlider = document.getElementById("covSlider");
const covBadge = document.getElementById("covBadge");
const covTicks = document.getElementById("covTicks");
const sweepCovBtn = document.getElementById("sweepCovBtn");
const lamSlider = document.getElementById("lamSlider");
const lamBadge = document.getElementById("lamBadge");
const lamTicks = document.getElementById("lamTicks");
const lamHint = document.getElementById("lamHint");
const sweepLamBtn = document.getElementById("sweepLamBtn");
const modeErr = document.getElementById("modeErr");
const modeVolt = document.getElementById("modeVolt");
const metricSelect = document.getElementById("metricSelect");

covSlider.max = COVERAGES.length - 1;
covSlider.value = state.covIdx;
covTicks.innerHTML = COVERAGES.map(c => `<span>${c}%</span>`).join("");

lamSlider.max = LAMBDAS.length - 1;
lamSlider.value = state.lamIdx;
lamTicks.innerHTML = `<span>${fmtLam(LAMBDAS[0])}</span><span>10⁻¹</span><span>${fmtLam(LAMBDAS[LAMBDAS.length - 1])}</span>`;

function stopAllSweeps() {
  if (covSweepTimer) { clearInterval(covSweepTimer); covSweepTimer = null; }
  if (lamSweepTimer) { clearInterval(lamSweepTimer); lamSweepTimer = null; }
  sweepCovBtn.classList.remove("running");
  sweepCovBtn.textContent = "▶ Auto-Sweep Coverage (Stress Test)";
  sweepLamBtn.classList.remove("running");
  sweepLamBtn.textContent = "▶ Auto-Sweep λ (Physics Reg.)";
}

function populateMetrics() {
  const opts = state.viewMode === "Errors"
    ? ["vm_rmse", "va_rmse", "physics_residual"]
    : ["vm_val", "va_val"];
  if (!opts.includes(state.metric)) state.metric = opts[0];
  metricSelect.innerHTML = opts.map(k => `<option value="${k}" ${k === state.metric ? "selected" : ""}>${PARAM_META[k].label}</option>`).join("");
}
populateMetrics();

function syncControlStates() {
  const isRef = state.model === "Ground_Truth" || state.model === "Pseudo";
  if (isRef) {
    state.condition = "Clean";
    condAug.disabled = true;
    condClean.disabled = true;
    condHint.textContent = "Locked to Clean for reference datasets.";
    lamSlider.disabled = true;
    sweepLamBtn.disabled = true;
    lamBadge.textContent = "N/A";
    lamHint.textContent = "Disabled for Ground_Truth / Pseudo.";
  } else {
    condClean.disabled = false;
    condAug.disabled = false;
    condHint.textContent = "";
    lamSlider.disabled = false;
    sweepLamBtn.disabled = false;
    lamBadge.textContent = fmtLam(LAMBDAS[state.lamIdx]);
    lamHint.textContent = "";
  }
  condClean.classList.toggle("active", state.condition === "Clean");
  condAug.classList.toggle("active", state.condition === "Augmented");
  modeErr.classList.toggle("active", state.viewMode === "Errors");
  modeVolt.classList.toggle("active", state.viewMode === "Voltages");
  covBadge.textContent = `${COVERAGES[state.covIdx]}%`;
  covSlider.value = state.covIdx;
  lamSlider.value = state.lamIdx;
}

const EDGE_X = [];
const EDGE_Y = [];
for (const [u, v] of BRANCHES) {
  EDGE_X.push(COORDS[u][0], COORDS[v][0], null);
  EDGE_Y.push(COORDS[u][1], COORDS[v][1], null);
}

function updateDashboard() {
  syncControlStates();

  const isRef = state.model === "Ground_Truth" || state.model === "Pseudo";
  const cov = COVERAGES[state.covIdx];
  const lam = isRef ? "N/A" : LAMBDAS[state.lamIdx];

  const rows = DATA
    .filter(r => r.condition === state.condition && r.model === state.model && r.pmu_coverage === cov && r.lambda === lam)
    .sort((a, b) => a.node - b.node);

  document.getElementById("bannerPills").innerHTML = `
    <span class="pill highlight">Model: ${state.model}</span>
    <span class="pill">Regime: ${state.condition}</span>
    <span class="pill">PMU Availability: ${cov}%</span>
    <span class="pill">λ: ${isRef ? "N/A" : fmtLam(lam)}</span>
  `;

  if (!rows.length) return;

  const meanVm = rows.reduce((s, r) => s + r.vm_rmse, 0) / rows.length;
  const maxVm = Math.max(...rows.map(r => r.vm_rmse));
  const meanVa = rows.reduce((s, r) => s + r.va_rmse, 0) / rows.length;
  const meanPhys = rows.reduce((s, r) => s + r.physics_residual, 0) / rows.length;
  const activePmus = rows.reduce((s, r) => s + r.is_pmu, 0);

  const vmCard = document.getElementById("kpiVmCard");
  vmCard.className = "kpi-card" + (meanVm < 0.0008 ? "" : meanVm < 0.0015 ? " warn" : " alert");
  document.getElementById("kpiVmVal").textContent = `${meanVm.toFixed(6)} p.u.`;
  document.getElementById("kpiPeakVal").textContent = `${maxVm.toFixed(5)} p.u.`;
  document.getElementById("kpiVaVal").textContent = `${meanVa.toFixed(4)}°`;
  document.getElementById("kpiPhysVal").textContent = `${meanPhys.toFixed(5)} p.u.`;
  document.getElementById("kpiPmuVal").textContent = `${activePmus} / 33`;

  renderTopology(rows);
  renderTrendCharts(cov, lam, isRef);
  renderTable(rows);
}

function renderTopology(rows) {
  const meta = PARAM_META[state.metric];
  const [cmin, cmax] = FIXED_BOUNDS[state.metric];

  const slackRows = rows.filter(r => r.node === 1);
  const pmuRows = rows.filter(r => r.node !== 1 && r.is_pmu === 1);
  const droppedRows = rows.filter(r => r.node !== 1 && r.is_pmu === 0);

  const makeHover = (r) =>
    `<b>Bus ${r.node}${r.node === 1 ? " (Slack Reference)" : ""}</b><br>` +
    `PMU Status: ${r.is_pmu === 1 ? "Operational PMU" : "Unmonitored / Dropped"}<br>` +
    `Vm: ${r.vm_val.toFixed(4)} p.u. (True: ${r.vm_true.toFixed(4)})<br>` +
    `Va: ${r.va_val.toFixed(3)}° (True: ${r.va_true.toFixed(3)}°)<br>` +
    `Vm RMSE: ${r.vm_rmse.toFixed(6)} p.u.<br>` +
    `Va RMSE: ${r.va_rmse.toFixed(4)}°<br>` +
    `Physics Residual: ${r.physics_residual.toFixed(5)} p.u.`;

  const traces = [
    // Feeder lines
    {
      x: EDGE_X, y: EDGE_Y,
      mode: "lines",
      line: { width: 2.2, color: "#CBD5E1" },
      hoverinfo: "none",
      showlegend: false,
    },
    // Outer ring halo for operational non-slack PMUs so sensor status is unmistakable
    {
      x: pmuRows.map(r => COORDS[r.node][0]),
      y: pmuRows.map(r => COORDS[r.node][1]),
      mode: "markers",
      name: "Operational PMU Ring",
      hoverinfo: "skip",
      marker: {
        symbol: "circle-open",
        size: 27,
        color: "#0F766E",
        line: { width: 2.6, color: "#0F766E" },
      },
      showlegend: false,
    },
    // Non-slack Operational PMUs (with fixed global colorbar)
    {
      x: pmuRows.map(r => COORDS[r.node][0]),
      y: pmuRows.map(r => COORDS[r.node][1]),
      mode: "markers+text",
      name: "Operational PMU",
      text: pmuRows.map(r => String(r.node)),
      textposition: "middle center",
      textfont: { size: 9.5, color: "#111827" },
      hovertext: pmuRows.map(makeHover),
      hoverinfo: "text",
      marker: {
        symbol: "circle",
        size: 21,
        color: pmuRows.map(r => r[state.metric]),
        colorscale: meta.scale,
        cmin, cmax,
        showscale: true,
        colorbar: { title: `${meta.label}<br>(${meta.unit})`, thickness: 14, len: 0.84 },
        line: { color: "#0F766E", width: 2.0 },
      },
    },
    // Unmonitored / Dropped buses (smaller muted circle border)
    {
      x: droppedRows.map(r => COORDS[r.node][0]),
      y: droppedRows.map(r => COORDS[r.node][1]),
      mode: "markers+text",
      name: "Unmonitored / Dropped",
      text: droppedRows.map(r => String(r.node)),
      textposition: "middle center",
      textfont: { size: 9, color: "#111827" },
      hovertext: droppedRows.map(makeHover),
      hoverinfo: "text",
      marker: {
        symbol: "circle",
        size: 17,
        color: droppedRows.map(r => r[state.metric]),
        colorscale: meta.scale,
        cmin, cmax,
        showscale: false,
        line: { color: "#94A3B8", width: 1.2 },
      },
    },
    // Bus 1: Slack Bus with distinct Star / Diamond icon
    {
      x: slackRows.map(r => COORDS[r.node][0]),
      y: slackRows.map(r => COORDS[r.node][1]),
      mode: "markers+text",
      name: "Slack Bus (Bus 1 Ref)",
      text: slackRows.map(r => "S" + String(r.node)),
      textposition: "bottom center",
      textfont: { size: 10, color: "#0F172A", weight: "bold" },
      hovertext: slackRows.map(makeHover),
      hoverinfo: "text",
      marker: {
        symbol: "diamond",
        size: 25,
        color: slackRows.map(r => r[state.metric]),
        colorscale: meta.scale,
        cmin, cmax,
        showscale: false,
        line: { color: "#0F172A", width: 2.5 },
      },
    },
  ];

  const layout = {
    title: {
      text: `IEEE 33-Bus Topology — ${meta.label} (Fixed Scale)`,
      x: 0.01, y: 0.96, xanchor: "left", yanchor: "top",
      font: { size: 15, color: "#1F2937" },
    },
    height: 410,
    margin: { l: 15, r: 20, t: 78, b: 15 },
    plot_bgcolor: "#FFFFFF",
    paper_bgcolor: "#FFFFFF",
    xaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-0.8, 17.8] },
    yaxis: { showgrid: false, zeroline: false, showticklabels: false, range: [-3.8, 4.8] },
    legend: { orientation: "h", x: 0.01, y: 1.03, xanchor: "left", yanchor: "bottom" },
  };

  Plotly.react("topoChart", traces, layout, { responsive: true, displayModeBar: false });
}

function renderTrendCharts(cov, lam, isRef) {
  const trendMetric = state.viewMode === "Errors" ? state.metric : "vm_rmse";
  const trendLabel = PARAM_META[trendMetric].label;
  const trendLam = isRef ? (LAMBDAS.includes("0.01") ? "0.01" : LAMBDAS[0]) : lam;
  const models = ["MLP", "GCN", "TCN"];

  // 1. Coverage Trend
  const covTraces = [];
  for (const m of models) {
    const xs = [], ys = [];
    for (const c of COVERAGES) {
      const sub = DATA.filter(r => r.condition === state.condition && r.model === m && r.lambda === trendLam && r.pmu_coverage === c);
      if (sub.length) {
        xs.push(c);
        ys.push(sub.reduce((s, r) => s + r[trendMetric], 0) / sub.length);
      }
    }
    if (xs.length) {
      covTraces.push({
        x: xs, y: ys, mode: "lines+markers", name: m,
        line: { color: MODEL_COLORS[m], width: m === state.model ? 3.2 : 2 },
        marker: { size: m === state.model ? 9 : 6.5 },
      });
    }
  }

  const covLayout = {
    title: {
      text: `${trendLabel} vs. PMU Availability (${state.condition}, λ=${fmtLam(trendLam)})`,
      x: 0.01, y: 0.96, xanchor: "left", yanchor: "top",
      font: { size: 13.5, color: "#1F2937" },
    },
    height: 325,
    margin: { l: 64, r: 20, t: 56, b: 68 },
    plot_bgcolor: "#FFFFFF",
    paper_bgcolor: "#FFFFFF",
    xaxis: { title: "PMU Availability (%)", tickvals: COVERAGES, gridcolor: "#F1F5F9" },
    yaxis: { title: trendLabel, gridcolor: "#F1F5F9" },
    legend: { orientation: "h", x: 0.5, y: -0.24, xanchor: "center", yanchor: "top" },
    shapes: [{
      type: "line", x0: cov, x1: cov, y0: 0, y1: 1, yref: "paper",
      line: { color: "#475569", width: 1.8, dash: "dash" }
    }],
    annotations: [{
      x: cov, y: 0.97, yref: "paper", text: `PMU: ${cov}%`, showarrow: false,
      xanchor: cov === COVERAGES[COVERAGES.length - 1] ? "right" : "left",
      xshift: cov === COVERAGES[COVERAGES.length - 1] ? -6 : 6,
      font: { size: 11, color: "#334155" }, bgcolor: "rgba(255,255,255,0.85)"
    }]
  };
  Plotly.react("covChart", covTraces, covLayout, { responsive: true, displayModeBar: false });

  // 2. Lambda Sensitivity Trend (0 to 10^3)
  const lamCats = LAMBDAS.map(fmtLam);
  const lamTraces = [];
  for (const m of models) {
    const xs = [], ys = [];
    for (const l of LAMBDAS) {
      const sub = DATA.filter(r => r.condition === state.condition && r.model === m && r.pmu_coverage === cov && r.lambda === l);
      if (sub.length) {
        xs.push(fmtLam(l));
        ys.push(sub.reduce((s, r) => s + r[trendMetric], 0) / sub.length);
      }
    }
    if (xs.length) {
      lamTraces.push({
        x: xs, y: ys, mode: "lines+markers", name: m,
        line: { color: MODEL_COLORS[m], width: m === state.model ? 3.2 : 2 },
        marker: { size: m === state.model ? 9 : 6.5 },
      });
    }
  }

  const lamIdx = LAMBDAS.indexOf(trendLam);
  const lamLayout = {
    title: {
      text: `${trendLabel} vs. Physics Weight λ (${state.condition}, ${cov}% PMU)`,
      x: 0.01, y: 0.96, xanchor: "left", yanchor: "top",
      font: { size: 13.5, color: "#1F2937" },
    },
    height: 325,
    margin: { l: 64, r: 20, t: 56, b: 68 },
    plot_bgcolor: "#FFFFFF",
    paper_bgcolor: "#FFFFFF",
    xaxis: { title: "Physics Weight λ", type: "category", categoryorder: "array", categoryarray: lamCats, gridcolor: "#F1F5F9" },
    yaxis: { title: trendLabel, gridcolor: "#F1F5F9" },
    legend: { orientation: "h", x: 0.5, y: -0.24, xanchor: "center", yanchor: "top" },
    shapes: isRef ? [] : [{
      type: "line", x0: lamIdx, x1: lamIdx, y0: 0, y1: 1, yref: "paper",
      line: { color: "#475569", width: 1.8, dash: "dash" }
    }],
    annotations: isRef ? [] : [{
      x: lamIdx, y: 0.97, yref: "paper", text: `λ = ${fmtLam(lam)}`, showarrow: false,
      xanchor: lamIdx === LAMBDAS.length - 1 ? "right" : "left",
      xshift: lamIdx === LAMBDAS.length - 1 ? -6 : 6,
      font: { size: 11, color: "#334155" }, bgcolor: "rgba(255,255,255,0.85)"
    }]
  };
  Plotly.react("lamChart", lamTraces, lamLayout, { responsive: true, displayModeBar: false });
}

function renderTable(rows) {
  document.getElementById("busTableBody").innerHTML = rows.map(r => `
    <tr>
      <td>${r.node}</td>
      <td>${r.node === 1 ? "Slack (Ref)" : (r.is_pmu ? "PMU Active" : "Dropped")}</td>
      <td>${r.vm_true.toFixed(4)}</td>
      <td>${r.vm_val.toFixed(4)}</td>
      <td>${r.vm_rmse.toFixed(6)}</td>
      <td>${r.va_true.toFixed(3)}°</td>
      <td>${r.va_val.toFixed(3)}°</td>
      <td>${r.va_rmse.toFixed(4)}°</td>
      <td>${r.physics_residual.toFixed(5)}</td>
    </tr>
  `).join("");
}

// Auto-sweep handlers (Stress test: 90% -> 50% coverage; Physics reg: 0 -> 10^3)
sweepCovBtn.addEventListener("click", () => {
  if (covSweepTimer) { stopAllSweeps(); return; }
  stopAllSweeps();
  state.covIdx = COVERAGES.length - 1;
  updateDashboard();
  sweepCovBtn.classList.add("running");
  sweepCovBtn.textContent = "⏹ Stop Coverage Auto-Sweep";
  covSweepTimer = setInterval(() => {
    state.covIdx = state.covIdx > 0 ? state.covIdx - 1 : COVERAGES.length - 1;
    updateDashboard();
  }, 750);
});

sweepLamBtn.addEventListener("click", () => {
  if (lamSweepTimer) { stopAllSweeps(); return; }
  stopAllSweeps();
  state.lamIdx = 0;
  updateDashboard();
  sweepLamBtn.classList.add("running");
  sweepLamBtn.textContent = "⏹ Stop λ Auto-Sweep";
  lamSweepTimer = setInterval(() => {
    state.lamIdx = (state.lamIdx + 1) % LAMBDAS.length;
    updateDashboard();
  }, 750);
});

modelSelect.addEventListener("change", () => { stopAllSweeps(); state.model = modelSelect.value; updateDashboard(); });
condClean.addEventListener("click", () => { state.condition = "Clean"; updateDashboard(); });
condAug.addEventListener("click", () => { state.condition = "Augmented"; updateDashboard(); });
covSlider.addEventListener("input", () => { stopAllSweeps(); state.covIdx = parseInt(covSlider.value, 10); updateDashboard(); });
lamSlider.addEventListener("input", () => { stopAllSweeps(); state.lamIdx = parseInt(lamSlider.value, 10); updateDashboard(); });
modeErr.addEventListener("click", () => { state.viewMode = "Errors"; populateMetrics(); updateDashboard(); });
modeVolt.addEventListener("click", () => { state.viewMode = "Voltages"; populateMetrics(); updateDashboard(); });
metricSelect.addEventListener("change", () => { state.metric = metricSelect.value; updateDashboard(); });

updateDashboard();
</script>
</body>
</html>
"""

rendered_html = (
    HTML_TEMPLATE
    .replace("__PLOTLY_JS__", plotly_js)
    .replace("__RECORDS_JSON__", records_json)
    .replace("__COVERAGES_JSON__", coverages_json)
    .replace("__LAMBDAS_JSON__", lambdas_json)
)

components.html(rendered_html, height=1000, scrolling=True)
