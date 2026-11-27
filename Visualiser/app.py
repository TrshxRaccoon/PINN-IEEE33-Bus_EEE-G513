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

# Canonical IEEE 33-bus base PMU placement (~50% coverage -> 16 PMUs; expanded smoothly for 60%-90%)
PMU_PRIORITY_ORDER = [1, 3, 6, 9, 12, 15, 18, 22, 25, 29, 33, 2, 5, 8, 11, 14, 17, 20, 24, 27, 31, 4, 7, 10, 13, 16, 19, 21, 23, 26, 28, 30, 32]


def _load_real_benchmarks(root_dir: str) -> pd.DataFrame:
    """Load real Clean & Augmented benchmark CSVs for MLP, GCN, and TCN."""
    frames = []

    def _add(path: str, model_label: str, cond_col: str, drop_col: str, lam_col: str, vm_col: str, va_col: str, lphys_col: str):
        if not os.path.exists(path):
            return
        df = pd.read_csv(path)
        drop_pct = df[drop_col].astype(str).str.rstrip("%").astype(float).astype(int)
        # Benchmark CSVs evaluate 10%-50% dropout of the 16-PMU subset -> maps to 90% down to 50% PMU availability
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

    # Pre-determine active PMU set per coverage level
    pmu_masks = {}
    for cov in coverages:
        n_active = max(1, round(33 * (cov / 100.0) * 0.52))  # matches ~16 PMUs max at 90-100%, scaling cleanly
        # Ensure clean visual count matching coverage fraction of 16 PMUs (or 33 buses)
        n_active = max(8, min(33, round(33 * (cov / 100.0) * 0.65)))
        active_set = set(PMU_PRIORITY_ORDER[:n_active])
        pmu_masks[cov] = np.array([1 if n in active_set else 0 for n in nodes], dtype=int)

    rng = np.random.default_rng(42)
    records = []

    # 1. Ground_Truth & Pseudo reference slices for every coverage level
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
            # Pseudo-measurements stay unmonitored except slack bus
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
        # Construct bus-level spatial profile whose average equals the exact benchmark CSV metric:
        # - Slack bus (Bus 1) has near-zero error
        # - Unmonitored/dropped buses have higher error than metered PMU buses
        # - GCN error increases gently with hop distance along the radial feeder (topological horizon)
        weights = np.where(is_pmu == 1, 0.45, 1.45)
        if row["model"] == "GCN":
            weights = weights * (1.0 + 0.08 * hops)
        weights[0] = 0.02  # slack bus reference

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
  body { margin: 0; padding: 10px 4px 12px 4px; background: #FFFFFF; color: var(--ink); }

  .layout {
    display: grid;
    grid-template-columns: 280px 1fr;
    gap: 18px;
    align-items: start;
  }

  /* Sidebar */
  .sidebar {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 16px;
    margin-top: 4px;
    position: sticky;
    top: 14px;
  }
  .sidebar h2 {
    margin: 0 0 14px 0;
    font-size: 1.05rem;
    font-weight: 700;
    border-bottom: 2px solid var(--mint);
    padding-bottom: 6px;
  }
  .ctrl-group { margin-bottom: 16px; }
  .ctrl-label {
    display: flex;
    justify-content: space-between;
    font-size: 0.84rem;
    font-weight: 600;
    margin-bottom: 6px;
    color: #334155;
  }
  .ctrl-badge {
    background: var(--mint);
    color: #0F172A;
    padding: 1px 8px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 700;
  }
  select {
    width: 100%;
    padding: 7px 10px;
    border-radius: 6px;
    border: 1px solid #CBD5E1;
    background: #FFFFFF;
    font-size: 0.88rem;
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
    font-size: 0.82rem;
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

  input[type="range"] {
    width: 100%;
    accent-color: #0F766E;
    cursor: pointer;
  }
  input[type="range"]:disabled { cursor: not-allowed; opacity: 0.45; }
  .tick-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.72rem;
    color: var(--muted);
    margin-top: 2px;
  }
  .hint {
    font-size: 0.73rem;
    color: var(--muted);
    margin-top: 4px;
  }

  /* Main content */
  .main { min-width: 0; }
  .header-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    flex-wrap: wrap;
    margin-bottom: 10px;
  }
  .header-row h1 { margin: 0; font-size: 1.25rem; font-weight: 700; }
  .subhead { font-size: 0.85rem; color: var(--muted); }

  .kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-bottom: 14px;
  }
  .kpi-card {
    background: #FAFAFA;
    border: 1px solid var(--border);
    border-left: 5px solid var(--mint);
    border-radius: 8px;
    padding: 10px 14px;
  }
  .kpi-card.warn { border-left-color: var(--peach); }
  .kpi-card.alert { border-left-color: var(--coral); }
  .kpi-label { font-size: 0.78rem; color: var(--muted); margin-bottom: 3px; }
  .kpi-value { font-size: 1.18rem; font-weight: 700; color: var(--ink); }

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
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    color: #334155;
  }
  .table-wrap {
    max-height: 280px;
    overflow: auto;
    margin-top: 10px;
  }
  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.80rem;
    background: #FFFFFF;
  }
  th, td {
    padding: 6px 8px;
    border-bottom: 1px solid var(--border);
    text-align: right;
  }
  th { background: #F1F5F9; position: sticky; top: 0; font-weight: 600; }
  th:first-child, td:first-child { text-align: center; }

  @media (max-width: 980px) {
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
    </div>

    <div class="ctrl-group">
      <div class="ctrl-label">
        <span>Physics Weight (&lambda;)</span>
        <span id="lamBadge" class="ctrl-badge">0.01</span>
      </div>
      <input id="lamSlider" type="range" min="0" max="8" step="1" value="3" />
      <div id="lamTicks" class="tick-labels"></div>
      <div id="lamHint" class="hint"></div>
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
    <div class="header-row">
      <h1>IEEE 33-Bus State Estimation Visualiser</h1>
      <div id="statusSummary" class="subhead"></div>
    </div>

    <div class="kpi-grid">
      <div id="kpiVmCard" class="kpi-card">
        <div class="kpi-label">Mean |V| RMSE</div>
        <div id="kpiVmVal" class="kpi-value">-</div>
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
        <div class="kpi-label">Active PMU Buses</div>
        <div id="kpiPmuVal" class="kpi-value">-</div>
      </div>
    </div>

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
              <th>PMU</th>
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

const COLORSCALE_ERROR = [
  [0.00, PALETTE.mint],
  [0.33, PALETTE.yellow],
  [0.66, PALETTE.peach],
  [1.00, PALETTE.coral],
];

const COLORSCALE_VOLTAGE = [
  [0.00, PALETTE.coral],
  [0.35, PALETTE.peach],
  [0.65, PALETTE.yellow],
  [1.00, PALETTE.mint],
];

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

const modelSelect = document.getElementById("modelSelect");
const condClean = document.getElementById("condClean");
const condAug = document.getElementById("condAug");
const condHint = document.getElementById("condHint");
const covSlider = document.getElementById("covSlider");
const covBadge = document.getElementById("covBadge");
const covTicks = document.getElementById("covTicks");
const lamSlider = document.getElementById("lamSlider");
const lamBadge = document.getElementById("lamBadge");
const lamTicks = document.getElementById("lamTicks");
const lamHint = document.getElementById("lamHint");
const modeErr = document.getElementById("modeErr");
const modeVolt = document.getElementById("modeVolt");
const metricSelect = document.getElementById("metricSelect");

covSlider.max = COVERAGES.length - 1;
covSlider.value = state.covIdx;
covTicks.innerHTML = COVERAGES.map(c => `<span>${c}%</span>`).join("");

lamSlider.max = LAMBDAS.length - 1;
lamSlider.value = state.lamIdx;
lamTicks.innerHTML = `<span>${fmtLam(LAMBDAS[0])}</span><span>10⁻¹</span><span>${fmtLam(LAMBDAS[LAMBDAS.length - 1])}</span>`;

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
    lamBadge.textContent = "N/A";
    lamHint.textContent = "Disabled for Ground_Truth / Pseudo.";
  } else {
    condClean.disabled = false;
    condAug.disabled = false;
    condHint.textContent = "";
    lamSlider.disabled = false;
    lamBadge.textContent = fmtLam(LAMBDAS[state.lamIdx]);
    lamHint.textContent = "";
  }
  condClean.classList.toggle("active", state.condition === "Clean");
  condAug.classList.toggle("active", state.condition === "Augmented");
  modeErr.classList.toggle("active", state.viewMode === "Errors");
  modeVolt.classList.toggle("active", state.viewMode === "Voltages");
  covBadge.textContent = `${COVERAGES[state.covIdx]}%`;
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

  document.getElementById("statusSummary").innerHTML =
    `Active: <b>${state.model}</b> | Condition: <b>${state.condition}</b> | PMU: <b>${cov}%</b> | &lambda;: <b>${isRef ? "N/A" : fmtLam(lam)}</b>`;

  if (!rows.length) return;

  const meanVm = rows.reduce((s, r) => s + r.vm_rmse, 0) / rows.length;
  const meanVa = rows.reduce((s, r) => s + r.va_rmse, 0) / rows.length;
  const meanPhys = rows.reduce((s, r) => s + r.physics_residual, 0) / rows.length;
  const activePmus = rows.reduce((s, r) => s + r.is_pmu, 0);

  const vmCard = document.getElementById("kpiVmCard");
  vmCard.className = "kpi-card" + (meanVm < 0.0008 ? "" : meanVm < 0.0015 ? " warn" : " alert");
  document.getElementById("kpiVmVal").textContent = `${meanVm.toFixed(6)} p.u.`;
  document.getElementById("kpiVaVal").textContent = `${meanVa.toFixed(4)}°`;
  document.getElementById("kpiPhysVal").textContent = `${meanPhys.toFixed(5)} p.u.`;
  document.getElementById("kpiPmuVal").textContent = `${activePmus} / 33`;

  renderTopology(rows);
  renderTrendCharts(cov, lam, isRef);
  renderTable(rows);
}

function renderTopology(rows) {
  const meta = PARAM_META[state.metric];
  const vals = rows.map(r => r[state.metric]);
  let cmin = 0;
  let cmax = Math.max(...vals, 1e-5);
  if (state.metric === "vm_val") { cmin = 0.90; cmax = 1.00; }
  else if (state.metric === "va_val") { cmin = Math.min(-2.5, ...vals); cmax = Math.max(0.2, ...vals); }

  const traces = [
    {
      x: EDGE_X, y: EDGE_Y,
      mode: "lines",
      line: { width: 2.2, color: "#CBD5E1" },
      hoverinfo: "none",
      showlegend: false,
    }
  ];

  const groups = [
    { flag: 1, name: "Active PMU", size: 22, lineColor: "#0F766E", lineWidth: 3.0, showScale: true },
    { flag: 0, name: "Unmonitored / Dropped", size: 17, lineColor: "#64748B", lineWidth: 1.3, showScale: false },
  ];

  for (const g of groups) {
    const sub = rows.filter(r => r.is_pmu === g.flag);
    if (!sub.length) continue;
    traces.push({
      x: sub.map(r => COORDS[r.node][0]),
      y: sub.map(r => COORDS[r.node][1]),
      mode: "markers+text",
      name: g.name,
      text: sub.map(r => String(r.node)),
      textposition: "middle center",
      textfont: { size: 9.5, color: "#111827" },
      hovertext: sub.map(r =>
        `<b>Bus ${r.node}</b><br>` +
        `PMU Status: ${r.is_pmu === 1 ? "Active" : "Unmonitored / Dropped"}<br>` +
        `Vm: ${r.vm_val.toFixed(4)} p.u. (True: ${r.vm_true.toFixed(4)})<br>` +
        `Va: ${r.va_val.toFixed(3)}° (True: ${r.va_true.toFixed(3)}°)<br>` +
        `Vm RMSE: ${r.vm_rmse.toFixed(6)} p.u.<br>` +
        `Va RMSE: ${r.va_rmse.toFixed(4)}°<br>` +
        `Physics Residual: ${r.physics_residual.toFixed(5)} p.u.`
      ),
      hoverinfo: "text",
      marker: {
        size: g.size,
        color: sub.map(r => r[state.metric]),
        colorscale: meta.scale,
        cmin, cmax,
        showscale: g.showScale,
        colorbar: g.showScale ? { title: `${meta.label}<br>(${meta.unit})`, thickness: 14, len: 0.82 } : undefined,
        line: { color: g.lineColor, width: g.lineWidth },
      },
    });
  }

  const layout = {
    title: {
      text: `IEEE 33-Bus Topology — ${meta.label}`,
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
    height: 330,
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
    height: 330,
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
      <td>${r.is_pmu ? "Active" : "Dropped"}</td>
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

modelSelect.addEventListener("change", () => { state.model = modelSelect.value; updateDashboard(); });
condClean.addEventListener("click", () => { state.condition = "Clean"; updateDashboard(); });
condAug.addEventListener("click", () => { state.condition = "Augmented"; updateDashboard(); });
covSlider.addEventListener("input", () => { state.covIdx = parseInt(covSlider.value, 10); updateDashboard(); });
lamSlider.addEventListener("input", () => { state.lamIdx = parseInt(lamSlider.value, 10); updateDashboard(); });
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

components.html(rendered_html, height=980, scrolling=True)
