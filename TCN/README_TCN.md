# Physics-Informed Temporal Convolutional Network (PI-TCN) for Power System State Estimation

> **Target System**: IEEE 33-Bus Radial Distribution Feeder (`case33bw`, $12.66\text{ kV}$, $10\text{ MVA}$ base)  
> **Target Problem**: State Estimation under Progressive PMU Telemetry Dropouts ($10\%$ to $50\%$)  
> **Precision Standard**: Full Double Precision (`torch.float64` / `torch.complex128`)

---

## 1. Executive Summary & Research Scope

In modern active distribution networks, phasor measurement units (PMUs) provide high-resolution, synchronized voltage measurements ($|V|, \theta$). However, communication latency, packet jitter, and telemetry dropouts regularly induce missing data patterns, compromising state awareness. 

Conventional unconstrained deep neural networks (e.g., standard MLPs or purely data-driven temporal estimators) fail catastrophically when telemetry is interrupted: they overfit historical patterns and predict electrically impossible system states (e.g., severe over-voltages, reverse power flows, or massive Kirchhoff current law violations).

This subproject implements a **Physics-Informed Temporal Convolutional Network (PI-TCN)** that incorporates non-linear AC power-flow physics directly into the temporal state estimation loop:

$$\mathbf{S}_{\text{calc}} = \mathbf{V} \odot (\mathbf{Y}_{\text{bus}} \mathbf{V})^*$$

By penalizing physical residuals between calculated complex power injections and pseudo-measurement forecasts as a soft regularizer, the network bounds estimation error and enforces grid consistency even when up to $50\%$ of PMU channels are abruptly dropped.

```
       Temporal Telemetry Window (T Steps)
   [ PMU |V| * mask (16) ]   ──┐
   [ PMU  θ  * mask (16) ]   ──┼─► [ Dilated Causal Conv1D ] ──► [ Residual Block ] ──► h_T ──► [ Linear Head ] ──► State Prediction:
   [ Active Power P_ps (33)] ──┤        (d = 1, 2, 4, 8)            (Tanh, float64)           (66 Outputs)         [ |V| (33), θ (33) ]
   [ React. Power Q_ps (33)] ──┤                                                                                          │
   [ Binary PMU Mask  (16) ] ──┘                                                                                          ▼
                                                                                                                   [ AC Physics Loss ]
                                                                                                           L_phys = ||S_calc - S_pseudo||²
```

---

## 2. Temporal Convolutional Network Architecture

Recurrent neural networks (RNNs, LSTMs, GRUs) suffer from sequential bottlenecks, vanishing gradients over long horizons, and non-deterministic memory decay. In contrast, the **Temporal Convolutional Network (TCN)** architecture (Bai, Kolter, & Koltun, 2018) provides:
1. **Strict Causality**: Convolutions are strictly unidirectional. Predictions at time $t$ depend solely on past and present measurements $\{\tau \le t\}$ with **zero future leakage**.
2. **Exponentially Expanding Receptive Field**: Dilated convolutions allow the network to capture long-term diurnal cycles and load dynamics with fewer layers and parameters.
3. **Massive Parallelism**: Unlike recurrent models, sequences are processed concurrently in 1D convolutional operations during training and inference.
4. **Stable Gradient Flow**: Residual skip connections ensure robust gradient backpropagation across all dilation stages.

### Dilated Causal Convolutions & Causal Chomp

For a 1D sequence input $\mathbf{x} \in \mathbb{R}^T$ and a convolution filter $f: \{0, \dots, k-1\} \to \mathbb{R}$, the dilated convolution operation with dilation factor $d$ is:

$$y(t) = (\mathbf{x} *_d f)(t) = \sum_{i=0}^{k-1} f(i) \cdot \mathbf{x}(t - d \cdot i)$$

To enforce strict causality:
- Standard convolutions are padded on both sides by $(k - 1) \cdot d$.
- A custom `Chomp1d((k-1)*d)` layer slices off the right-hand trailing padding:
  
  $$\mathbf{x}_{\text{causal}} = \mathbf{x}[:, :, :-(k-1)d]$$
  
  This mathematically guarantees that no future information leaks into the hidden states.

### Receptive Field Calculation

In a $K$-level TCN with kernel size $k$ and dilations $d = 2^0, 2^1, \dots, 2^{L-1}$, each residual block contains two dilated convolutional stages. The theoretical receptive field (in time steps) is:

$$\text{RF} = 1 + \sum_{l=0}^{L-1} 2 \cdot (k - 1) \cdot 2^l = 1 + 2(k - 1)(2^L - 1)$$

For our default 4-layer architecture with $k=3$ and channels $[64, 64, 128, 128]$:

$$\text{RF} = 1 + 2 \cdot (3 - 1) \cdot (2^4 - 1) = 1 + 4 \times 15 = 61 \text{ time steps}$$

With an hourly sampling rate ($\Delta t = 1.0\text{ h}$), a 61-step receptive field easily spans over 2.5 full diurnal load cycles, capturing deep day-to-day load rhythms.

### Residual Temporal Block Architecture

Each `TemporalBlock` consists of:
1. `Conv1d(in_ch, out_ch, kernel_size=3, dilation=d, padding=2*d, dtype=torch.float64)`
2. `Chomp1d(2*d)`
3. Smooth activation: `nn.Tanh()`
4. `Dropout(p=0.1)` (disabled during test evaluation)
5. `Conv1d(out_ch, out_ch, kernel_size=3, dilation=d, padding=2*d, dtype=torch.float64)`
6. `Chomp1d(2*d)`
7. Smooth activation: `nn.Tanh()`
8. `Dropout(p=0.1)`
9. Residual Connection: $1 \times 1$ `Conv1d` projection if $\text{in\_ch} \neq \text{out\_ch}$, else `nn.Identity()`
10. Post-addition activation: `nn.Tanh()`

### Numerical Stability: Double Precision & Smooth Activations
- **`torch.float64` / `torch.complex128`**: Admittance matrix entries reach magnitudes of $10^2$ to $10^3\text{ pu}$. Subtracting nearly identical complex voltage drops causes fatal cancellation in single-precision `float32`. Double precision is strictly enforced across all weights, activations, and loss computations.
- **$\tanh$ Activation**: Non-smooth piecewise linear activations (e.g., standard ReLU) possess discontinuous first derivatives and zero second derivatives, disrupting physics-informed backpropagation through non-linear power-flow residuals. Smooth $\tanh$ ensures well-behaved gradients.

---

## 3. Physical Formulation: AC Power Flow Regularizer

### IEEE 33-Bus System Specifications
- **Base Power**: $S_{\text{base}} = 10\text{ MVA}$
- **Nominal Voltage**: $V_{\text{base}} = 12.66\text{ kV}$
- **Topology**: 33 buses (Bus 0 is the slack substation), 32 radial branches.
- **Complex Bus Admittance Matrix**: $\mathbf{Y}_{\text{bus}} \in \mathbb{C}^{33 \times 33}$, derived from line series impedances ($R + jX$) and shunt admittances.

### AC Power Injection Residuals
Complex bus voltages are parameterized in polar coordinates:

$$\mathbf{V} = |V| \odot \exp(j \theta) \in \mathbb{C}^{33}$$

The physical complex power injected at each bus is governed by Ohm's and Kirchhoff's circuit laws:

$$\mathbf{S}_{\text{calc}} = \mathbf{V} \odot (\mathbf{Y}_{\text{bus}} \mathbf{V})^*$$

Decomposed into real active power $\mathbf{P}$ and reactive power $\mathbf{Q}$:

$$\mathbf{P}_{\text{calc}} = \operatorname{Re}(\mathbf{S}_{\text{calc}}), \quad \mathbf{Q}_{\text{calc}} = \operatorname{Im}(\mathbf{S}_{\text{calc}})$$

The physical mismatch relative to forecasted pseudo-injections $(\mathbf{P}_{\text{pseudo}}, \mathbf{Q}_{\text{pseudo}})$ is:

$$r_P = \mathbf{P}_{\text{calc}} - \mathbf{P}_{\text{pseudo}}, \quad r_Q = \mathbf{Q}_{\text{calc}} - \mathbf{Q}_{\text{pseudo}}$$

### Slack Bus Exclusion & Physical Restoration
- **Exclusion**: The slack bus (Bus 0) represents the infinite grid substation. Its active and reactive power injections fluctuate dynamically to balance total feeder line losses and cannot be fixed to a pseudo-measurement. Thus, **Bus 0 is excluded from both the supervised loss $\mathcal{L}_{\text{data}}$ and the physics loss $\mathcal{L}_{\text{phys}}$**.
- **Restoration**: During de-standardization (`to_physical`), slack bus values are rigidly pinned to their physical constants:

$$|V|_0 = 1.0\text{ pu}, \quad \theta_0 = 0.0\text{ rad}$$

### Loss Function Formulation
The physics regularizer loss is evaluated strictly over the 32 non-slack buses ($i \in \{1, \dots, 32\}$):

$$\mathcal{L}_{\text{phys}} = \frac{1}{32} \sum_{i=1}^{32} \left( r_{P,i}^2 + r_{Q,i}^2 \right)$$

The supervised data-fidelity loss is the MSE over the 64 standardized non-slack state targets:

$$\mathcal{L}_{\text{data}} = \frac{1}{64} \sum_{k \in \text{non-slack}} (\hat{y}_k - y_k)^2$$

The combined optimization objective is:

$$\mathcal{L} = \mathcal{L}_{\text{data}} + \lambda(t) \cdot \mathcal{L}_{\text{phys}}$$

### 20% Epoch $\lambda$ Warm-Up Schedule
Untrained convolutional filters initially produce chaotic voltage predictions, yielding gigantic physical residuals that destabilize early training. To guarantee convergence:
- **Epochs 0% to 10%**: $\lambda = 0.0$ (warm up data-fitting first).
- **Epochs 10% to 20%**: $\lambda$ ramps linearly from $0.0$ to target $\lambda$.
- **Epochs 20% to 100%**: $\lambda = \text{target } \lambda$.

---

## 4. Chronological Sequential Dataset Pipeline

Unlike static snapshot models that treat consecutive power flow samples as independent draws, distribution grids exhibit strong temporal autocorrelation driven by diurnal human activity.

### Autoregressive Diurnal Load Process
1. **24-Hour Diurnal Profile**: Base residential curve with morning (08:00) and evening (19:00) peaks:
   
   $$\text{daily\_profile}(h) = 0.55 + 0.25 \exp\left(-\frac{1}{2}\left(\frac{h - 8}{2}\right)^2\right) + 0.45 \exp\left(-\frac{1}{2}\left(\frac{h - 19}{2.5}\right)^2\right)$$

2. **Markovian AR(1) Load Fluctuation**: To simulate realistic load drift across consecutive hours:
   
   $$\xi_{t,i} = \rho \cdot \xi_{t-1,i} + (1 - \rho) \cdot \mu + \epsilon_t, \quad \rho = 0.90, \; \mu = 1.0, \; \sigma = 0.05$$
   
   Clipped to $[0.5, 1.5]$ to prevent pathological feeder divergence.

### 114 Input Features per Time Step
At every time step $t$, the input feature vector $\mathbf{x}_t \in \mathbb{R}^{114}$ follows an invariant block structure:
1. **Block 1 (16 features)**: Standardized PMU $|V| \odot \text{mask}$
2. **Block 2 (16 features)**: Standardized PMU $\theta \odot \text{mask}$
3. **Block 3 (33 features)**: Standardized active power pseudo-measurements $P_{\text{pseudo}}$
4. **Block 4 (33 features)**: Standardized reactive power pseudo-measurements $Q_{\text{pseudo}}$
5. **Block 5 (16 features)**: Raw binary PMU availability mask ($1.0 = \text{active}, 0.0 = \text{dropped}$)

> [!IMPORTANT]
> **Standardize First, Mask Second**: PMU readings are z-scored using training mean and standard deviation *before* multiplying by the mask:
> $$x_{\text{pmu}} = \left(\frac{V_{\text{raw}} - \mu_{\text{train}}}{\sigma_{\text{train}}}\right) \odot \text{mask}$$
> When a PMU drops telemetry ($\text{mask} = 0$), the feature value becomes $0.0$, which is the exact training mean (neutral). Zeroing raw values before scaling would erroneously inject artificial $0\text{ V}$ ground faults!

### Sliding Window Tensor Structure
- **Input Tensor**: Sliding window of length $T = 12$ consecutive hours: shape $(B, 114, T)$ in `torch.float64`.
- **Target Vector**: Ground truth state vector $[|V|_{0..32}, \theta_{0..32}]$ at the **current/latest time step** $T - 1$: shape $(B, 66)$.

---

## 5. Expanded $\lambda$ Regularization Sweep

To rigorously map the Pareto frontier between empirical telemetry fitting and physical law satisfaction, `train_tcn.py` executes an expanded 9-point $\lambda$ sweep:

$$\lambda \in \left[ 0.0, \; 10^{-4}, \; 10^{-3}, \; 10^{-2}, \; 10^{-1}, \; 10^0, \; 10^1, \; 10^2, \; 10^3 \right]$$

- **$\lambda = 0.0$**: Pure Data-Driven Baseline (unconstrained empirical fit).
- **$\lambda \in [10^{-4}, 10^{-1}]$**: Weak-to-moderate physics regularization.
- **$\lambda = 1.0$**: Balanced data-physics weighting.
- **$\lambda \in [10^1, 10^3]$**: High-physics enforcement regime, forcing strict satisfaction of Ohm's and Kirchhoff's laws even under severe telemetry starvation.

---

## 6. Clean vs. Augmented Training Regimes

1. **Clean Training**:
   - Trained on 100% complete PMU telemetry (`mask = 1.0` for all training samples).
   - Evaluated on test sets with 10%, 20%, 30%, 40%, and 50% PMU dropouts.
   - Tests whether physical constraints prevent catastrophic model drift when exposed to unexpected telemetry failure.
2. **Augmented Training (`--aug`)**:
   - Random dropout levels $p \sim \mathcal{U}(0, 0.5)$ are drawn fresh per sample on-the-fly during training.
   - Eliminates train/test distribution shift, representing the true robust benchmark.

---

## 7. Step-by-Step Operations & PowerShell Runbook

Below are copy-pasteable PowerShell terminal commands for the complete end-to-end lifecycle.

### Step 1: Install Dependencies
```powershell
pip install torch pandapower numpy pandas matplotlib
```

### Step 2: Generate Chronological Sequential Dataset
Generates continuous 24-hour diurnal trajectories for the IEEE 33-bus system:
```powershell
python make_sequential_dataset.py --out data --systems ieee33 --n_train 8000 --n_val 1000 --n_test 1000 --step_hours 1.0 --seed 42
```
*Creates `data/ieee33/train.npz`, `val.npz`, `test_drop10.npz` through `test_drop50.npz`, and `meta.npz`.*

### Step 3: Run Contract & Causality Assertions
Executes an automated verification suite verifying dataset shapes, scaler isolation, physical residual tolerances, and **strict causal temporal invariance**:
```powershell
python check_spec_contract.py --data data/ieee33 --seq_len 12
```

### Step 4: Train Clean Models (Expanded Sweep)
Trains Baseline ($\lambda=0$) and PI-TCN models across all 9 $\lambda$ values on 100% complete data:
```powershell
python train_tcn.py --data data/ieee33 --epochs 100 --seq_len 12 --hidden 64 64 128 128 --kernel_size 3 --act tanh --lams 0.0 1e-4 1e-3 1e-2 1e-1 1.0 10.0 100.0 1000.0 --out results_clean
```
*Saves all model checkpoints `model_Clean_lam*_seed42.pt`, the best checkpoint `best_model_Clean_seed42.pt`, and `benchmark_clean.csv`.*

### Step 5: Train Augmented Models (Dropout Augmentation)
Trains models under dynamic on-the-fly PMU dropouts ($\mathcal{U}(0, 0.5)$):
```powershell
python train_tcn.py --data data/ieee33 --epochs 100 --seq_len 12 --hidden 64 64 128 128 --kernel_size 3 --act tanh --lams 0.0 1e-4 1e-3 1e-2 1e-1 1.0 10.0 100.0 1000.0 --aug --out results_augmented
```
*Saves all augmented checkpoints and `benchmark_augmented.csv`.*

### Step 6: Generate Publication-Quality Plots
Reads benchmark CSVs and produces high-resolution (300 DPI) curves comparing Baseline vs. Best PI-TCN:
```powershell
python plot_results.py --clean_csv results_clean/benchmark_clean.csv --aug_csv results_augmented/benchmark_augmented.csv --out_dir plots
```
*Exports `plots/tcn_rmse_vs_coverage_clean.png` and `plots/tcn_rmse_vs_coverage_augmented.png`.*

### Step 7: Run Interactive Model Comparison Tool
Evaluate Baseline vs. PI-TCN side-by-side on any sequential sliding window:

```powershell
# A. Inspect Clean models on Window 0 with 50% PMU dropout
python demo_compare.py --data_dir data/ieee33 --condition Clean --drop 50 --seq_idx 0 --seq_len 12

# B. Inspect Augmented models
python demo_compare.py --data_dir data/ieee33 --condition Augmented --drop 50 --seq_idx 0 --seq_len 12

# C. Generate and save the 2-panel comparison chart
python demo_compare.py --data_dir data/ieee33 --condition Clean --drop 50 --seq_idx 0 --seq_len 12 --plot
```
*Prints PMU telemetry health status, terminal comparison table across all 33 buses, summary evaluation card, and saves `snapshot_comparison.png`.*

---

## 8. Directory File Manifest

```
TCN/
├── make_sequential_dataset.py   # Chronological diurnal power-flow simulator & AR(1) dynamics
├── tcn_model.py                 # Double-precision Causal Dilated 1D TCN & AC physics functions
├── check_spec_contract.py       # 6-check unit test suite (including strict causality test)
├── train_tcn.py                 # Training engine with sliding window, warm-up, and λ sweep
├── plot_results.py              # Publication-quality 300 DPI plotting suite
├── demo_compare.py              # Interactive terminal table & visual snapshot comparison
├── README.md                    # Master engineering and operational documentation
├── data/                        # Generated chronological datasets
│   └── ieee33/
│       ├── train.npz            # 8,000 clean training snapshots (mask = 1.0)
│       ├── val.npz              # 1,000 validation snapshots (random 10-50% dropout)
│       ├── test_drop10.npz      # 1,000 chronological test snapshots (10% PMU dropout)
│       ├── test_drop20.npz      # 1,000 identical snapshots (20% PMU dropout)
│       ├── test_drop30.npz      # 1,000 identical snapshots (30% PMU dropout)
│       ├── test_drop40.npz      # 1,000 identical snapshots (40% PMU dropout)
│       ├── test_drop50.npz      # 1,000 identical snapshots (50% PMU dropout)
│       └── meta.npz             # Y_bus admittance matrix, slack index, PMU positions
├── plots/                       # Publication figures (300 DPI)
│   ├── tcn_rmse_vs_coverage_clean.png
│   └── tcn_rmse_vs_coverage_augmented.png
├── results_clean/               # Checkpoints and CSV for Clean training
└── results_augmented/           # Checkpoints and CSV for Augmented training
```
