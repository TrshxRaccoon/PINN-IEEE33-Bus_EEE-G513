# Demonstration & Model Comparison Suite

This folder provides an interactive demonstration tool for inspecting and comparing the state estimation performance of the **Baseline MLP ($\lambda=0$)** against the **Physics-Informed MLP ($\lambda > 0$)** on the IEEE 33-bus distribution grid under sensor dropout.

---

## What the Demo Does

1. **Loads Both Models**: Directly loads trained `.pt` model weights for side-by-side inference.
2. **Injects Controlled Dropout**: Evaluates on real test snapshots across any dropout level ($10\%$ to $50\%$).
3. **Telemetry Tracking**: Displays which specific PMU buses had active telemetry vs. which suffered communication loss.
4. **Bus-by-Bus Comparison**: Prints the exact ground truth, baseline prediction, and physics-informed prediction for all 33 buses, highlighting where the physics regularizer corrected voltage drift.
5. **Generates Visual Profiles**: Optionally generates high-resolution comparison graphs across the feeder.

---

## Quick Start / Usage

Run the demo script from the project root:

### 1. Default Quick Demo (50% Dropout on Clean Models)
```powershell
python demo/compare_models.py --condition Clean --drop 50 --snapshot 0
```

### 2. Compare on Augmented Models
```powershell
python demo/compare_models.py --condition Augmented --drop 50 --snapshot 0
```

### 3. Generate a Visual Bus Comparison Plot
Add the `--plot` flag to generate and save `snapshot_0_drop50_comparison.png`:
```powershell
python demo/compare_models.py --condition Clean --drop 50 --snapshot 0 --plot
```

### 4. Custom Model Paths
You can compare any two `.pt` checkpoints directly:
```powershell
python demo/compare_models.py --baseline results_clean/model_Clean_lam0.0_seed42.pt --pinn results_clean/best_model_Clean_seed42.pt --drop 50
```

---

## Available Command Line Options

| Argument | Default | Choices / Format | Description |
| :--- | :--- | :--- | :--- |
| `--condition` | `Clean` | `Clean`, `Augmented` | Which training condition models to load. |
| `--drop` | `50` | `10`, `20`, `30`, `40`, `50` | PMU dropout level percentage. |
| `--snapshot` | `0` | Integer (`0` to `999`) | Index of the test snapshot to inspect. |
| `--baseline` | Auto | Filepath (`.pt`) | Path to the baseline model checkpoint. |
| `--pinn` | Auto | Filepath (`.pt`) | Path to the physics-informed model checkpoint. |
| `--plot` | Disabled | Flag | Generates a 2-panel voltage profile and error plot. |
