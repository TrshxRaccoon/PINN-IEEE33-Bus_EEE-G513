#!/usr/bin/env python3
"""
make_sequential_dataset.py -- Chronological Sequential Dataset Generator for TCN State Estimation
Physics-informed state estimation for distribution grids under PMU dropout.

TARGET SYSTEM:
    IEEE 33-bus radial distribution feeder (pandapower `case33bw`, 12.66 kV, 10 MVA base).

KEY SEQUENTIAL CHARACTERISTICS:
    1. Chronological Time Evolution:
       Time of day advances chronologically: h_t = (h_0 + t * delta_t) mod 24.0
    2. Smooth Diurnal Load Profile:
       Residential curve: base + morning peak + evening peak via Gaussian bumps:
       daily_profile(hour) = 0.55 + 0.25 * exp(-0.5 * ((hour - 8)/2)^2) + 0.45 * exp(-0.5 * ((hour - 19)/2.5)^2)
    3. Autoregressive AR(1) Load Variation Process:
       per_load_t = rho * per_load_{t-1} + (1 - rho) * mu + epsilon_t
       with rho = 0.90, mu = 1.0, sigma = 0.05, clipped to [0.5, 1.5].
       Provides realistic, continuous diurnal load curves where consecutive time steps are smoothly correlated.
    4. Measurement Model:
       - 16 PMUs placed across non-slack buses (~50% coverage).
       - Additive Gaussian PMU noise: sigma_vm = 0.001 pu, sigma_va = 0.05 deg.
       - Multiplicative load forecast pseudo-measurements: sigma_pseudo = 0.10 on P, Q injections (slack set to 0.0).
    5. Dropout Masks:
       - train.npz: complete telemetry (mask = 1.0)
       - val.npz: dropout levels randomly drawn per snapshot from {10, 20, 30, 40, 50}%
       - test_drop10.npz to test_drop50.npz: paired test files with identical chronological states, fixed dropout masks
       - meta.npz: Ybus admittance matrix, slack bus index, PMU bus indices, feeder parameters.

USAGE:
    python make_sequential_dataset.py --out data --systems ieee33
    python make_sequential_dataset.py --n_train 300 --n_val 100 --n_test 100 --out data_small
"""

import argparse
import os
from typing import Dict, List, Sequence, Tuple, Union

import numpy as np
import pandapower as pp
import pandapower.networks as pn


# ----------------------------------------------------------------------------------
# 1. BUILD THE NETWORKS
# ----------------------------------------------------------------------------------
def build_net(system: str) -> pp.pandapowerNet:
    """Build and return pandapower network for the specified system."""
    if system.lower() == "ieee33":
        return pn.case33bw()  # Baran-Wu 33-bus radial distribution network
    raise ValueError(f"Unknown system: {system}. Supported systems: ['ieee33']")


# ----------------------------------------------------------------------------------
# 2. DIURNAL LOAD PROFILE & AR(1) DYNAMICS
# ----------------------------------------------------------------------------------
def daily_profile(hour: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Smooth residential-style 24-hour diurnal curve:
    base load + morning peak (~08:00) + evening peak (~19:00).
    Output is a multiplier on nominal feeder load (~0.55 at night, ~1.0 at 19:00).
    """
    base = 0.55
    morning = 0.25 * np.exp(-0.5 * ((hour - 8.0) / 2.0) ** 2)
    evening = 0.45 * np.exp(-0.5 * ((hour - 19.0) / 2.5) ** 2)
    return base + morning + evening


# ----------------------------------------------------------------------------------
# 3. ADMITTANCE MATRIX CALCULATION
# ----------------------------------------------------------------------------------
def get_ybus(net: pp.pandapowerNet) -> np.ndarray:
    """
    Extract the complex bus admittance matrix Y_bus (per-unit) in pandapower bus index order.
    Physical circuit law: I_inj = Y @ V.
    """
    pp.runpp(net, calculate_voltage_angles=True)
    Yppc = net._ppc["internal"]["Ybus"].toarray()
    lut = net._pd2ppc_lookups["bus"][net.bus.index.values]  # pandapower idx -> internal idx
    return Yppc[np.ix_(lut, lut)]


# ----------------------------------------------------------------------------------
# 4. CHRONOLOGICAL SEQUENTIAL SIMULATION
# ----------------------------------------------------------------------------------
def simulate_sequential_snapshots(
    net: pp.pandapowerNet,
    n: int,
    rng: np.random.Generator,
    p0: np.ndarray,
    q0: np.ndarray,
    step_hours: float = 1.0,
    h0: float = 0.0,
    rho: float = 0.90,
    mu: float = 1.0,
    sigma: float = 0.05,
) -> Dict[str, np.ndarray]:
    """
    Solve n chronological, consecutive AC power-flow steps with an AR(1) load variation process.

    Parameters:
      net: pandapower network
      n: number of time steps to generate
      rng: numpy random generator
      p0: nominal active load powers (MW)
      q0: nominal reactive load powers (MVar)
      step_hours: time increment per step Delta t (hours)
      h0: initial hour of day
      rho: AR(1) persistence coefficient (default 0.90)
      mu: AR(1) long-term mean (default 1.0)
      sigma: AR(1) innovation noise standard deviation (default 0.05)

    Returns:
      Dictionary containing:
        vm: true voltage magnitudes (|V| in pu), shape (n, n_bus)
        va: true voltage phase angles (theta in rad), shape (n, n_bus)
        pinj: true active power injections (pu), shape (n, n_bus)
        qinj: true reactive power injections (pu), shape (n, n_bus)
        hour: chronological hour of day mod 24, shape (n,)
    """
    nb = len(net.bus)
    nl = len(net.load)

    vm = np.zeros((n, nb), dtype=np.float64)
    va = np.zeros((n, nb), dtype=np.float64)
    pinj = np.zeros((n, nb), dtype=np.float64)
    qinj = np.zeros((n, nb), dtype=np.float64)
    hour = np.zeros(n, dtype=np.float64)

    # Initialize AR(1) load factor state at mu = 1.0
    per_load_prev = np.full(nl, mu, dtype=np.float64)

    k = 0
    fails = 0
    while k < n:
        # Chronological hour of the day: h_t = (h_0 + k * Delta t) mod 24.0
        h = (h0 + k * step_hours) % 24.0

        # AR(1) Markovian update: per_load_t = rho * per_load_{t-1} + (1 - rho) * mu + epsilon_t
        eps = rng.normal(0.0, sigma, nl)
        per_load = np.clip(rho * per_load_prev + (1.0 - rho) * mu + eps, 0.5, 1.5)

        # Scale by smooth diurnal profile
        scale = daily_profile(h) * per_load
        net.load["p_mw"] = p0 * scale
        net.load["q_mvar"] = q0 * scale * np.clip(rng.normal(1.0, 0.02, nl), 0.9, 1.1)

        try:
            pp.runpp(net, calculate_voltage_angles=True)
        except pp.LoadflowNotConverged:
            fails += 1
            if fails > 5 * n + 100:
                raise RuntimeError("AC power flow repeatedly diverged - check network parameters and load scaling.")
            # Retry current time step with newly sampled innovation noise from per_load_prev
            continue

        # Extract converged AC state
        r = net.res_bus.loc[net.bus.index]
        vm[k] = r["vm_pu"].values
        va[k] = np.deg2rad(r["va_degree"].values)

        # pandapower convention: p_mw is consumption (+); injection convention is generation (+) -> minus sign
        pinj[k] = -r["p_mw"].values / net.sn_mva
        qinj[k] = -r["q_mvar"].values / net.sn_mva
        hour[k] = h

        # Transition forward
        per_load_prev = per_load
        k += 1

        if k % 500 == 0 or k == n:
            print(f"    simulated {k}/{n} sequential steps (t={k}, hour={h:.2f})", flush=True)

    return dict(vm=vm, va=va, pinj=pinj, qinj=qinj, hour=hour)


# ----------------------------------------------------------------------------------
# 5. MEASUREMENT MODEL
# ----------------------------------------------------------------------------------
def make_measurements(
    sim: Dict[str, np.ndarray],
    pmu_idx: np.ndarray,
    slack: int,
    rng: np.random.Generator,
    sig_vm: float,
    sig_va_deg: float,
    sig_pseudo: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Build noisy PMU readings (|V|, theta at PMU buses) and noisy pseudo-measurements
    of P/Q injections at all buses.

    - PMU noise: additive Gaussian (sig_vm in pu, sig_va_deg in degrees).
    - Pseudo-measurements: multiplicative Gaussian forecast error (e.g. 10%).
      Multiplicative error ensures zero-injection buses stay 0.
      Slack bus injection is unobserved -> set to 0.0.
    """
    n = sim["vm"].shape[0]
    k = len(pmu_idx)

    # Additive PMU noise
    pm_vm = sim["vm"][:, pmu_idx] + rng.normal(0.0, sig_vm, (n, k))
    pm_va = sim["va"][:, pmu_idx] + rng.normal(0.0, np.deg2rad(sig_va_deg), (n, k))

    # Multiplicative load forecast pseudo-measurements
    p_ps = sim["pinj"] * (1.0 + rng.normal(0.0, sig_pseudo, sim["pinj"].shape))
    q_ps = sim["qinj"] * (1.0 + rng.normal(0.0, sig_pseudo, sim["qinj"].shape))

    # Slack bus injection is unknown / unconstrained
    p_ps[:, slack] = 0.0
    q_ps[:, slack] = 0.0

    return (
        pm_vm.astype(np.float32),
        pm_va.astype(np.float32),
        p_ps.astype(np.float32),
        q_ps.astype(np.float32),
    )


def make_mask(
    rng: np.random.Generator,
    n: int,
    k: int,
    drop_frac: Union[float, np.ndarray, Sequence[float]],
) -> np.ndarray:
    """
    Generate PMU dropout mask.
    In each snapshot, exactly round(drop_frac * k) PMUs are masked out (0.0),
    chosen uniformly at random.
    Returns:
      mask of shape (n, k), where 1.0 = available, 0.0 = dropped.
    """
    drop_frac_arr = np.broadcast_to(np.asarray(drop_frac, dtype=float), (n,))
    n_drop = np.rint(drop_frac_arr * k).astype(int)
    ranks = rng.random((n, k)).argsort(1).argsort(1)
    return (ranks >= n_drop[:, None]).astype(np.float32)


def pack(
    sim: Dict[str, np.ndarray],
    meas: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    mask: np.ndarray,
    drop_frac: Union[float, np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Assemble the dataset dictionary saved to .npz file.
    Dropped telemetry channels are zeroed in pmu_vm and pmu_va.
    """
    pm_vm, pm_va, p_ps, q_ps = meas
    return dict(
        pmu_vm=pm_vm * mask,
        pmu_va=pm_va * mask,
        mask=mask,
        pmu_vm_full=pm_vm,
        pmu_va_full=pm_va,
        p_pseudo=p_ps,
        q_pseudo=q_ps,
        vm_true=sim["vm"].astype(np.float32),
        va_true=sim["va"].astype(np.float32),
        p_true=sim["pinj"].astype(np.float32),
        q_true=sim["qinj"].astype(np.float32),
        drop_frac=np.broadcast_to(drop_frac, (len(mask),)).astype(np.float32),
        hour=sim["hour"].astype(np.float32),
    )


# ----------------------------------------------------------------------------------
# 6. PHYSICAL SANITY CHECK
# ----------------------------------------------------------------------------------
def check_physics(Y: np.ndarray, sim: Dict[str, np.ndarray], slack: int) -> float:
    """
    Verify AC power flow equation holds:
      S_calc = V * conj(V @ Y.T)
      err = |S_calc - (p_inj + 1j * q_inj)|
    Max error across non-slack buses should be < 1e-6 pu.
    """
    V = sim["vm"] * np.exp(1j * sim["va"])  # Polar to complex, shape (n, nb)
    S = V * np.conj(V @ Y.T)                # S_i = V_i * conj((Y V)_i)
    err = np.abs(S - (sim["pinj"] + 1j * sim["qinj"]))
    err[:, slack] = 0.0                     # Exclude slack bus
    return float(err.max())


# ----------------------------------------------------------------------------------
# 7. MAIN PIPELINE
# ----------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Generate chronological sequential AC power-flow datasets with AR(1) dynamics for TCN."
    )
    parser.add_argument("--out", default="data", help="Output directory path (default: 'data')")
    parser.add_argument("--systems", nargs="+", default=["ieee33"], help="List of systems (default: ['ieee33'])")
    parser.add_argument("--n_train", type=int, default=8000, help="Number of training steps (default: 8000)")
    parser.add_argument("--n_val", type=int, default=1000, help="Number of validation steps (default: 1000)")
    parser.add_argument("--n_test", type=int, default=1000, help="Number of test steps (default: 1000)")
    parser.add_argument("--step_hours", type=float, default=1.0, help="Time step Delta t in hours (default: 1.0)")
    parser.add_argument("--h0", type=float, default=0.0, help="Initial hour of day (default: 0.0)")
    parser.add_argument("--pmu_frac", type=float, default=0.5, help="Fraction of non-slack buses with PMU (default: 0.5)")
    parser.add_argument("--sig_vm", type=float, default=0.001, help="PMU |V| noise std in pu (default: 0.001)")
    parser.add_argument("--sig_va", type=float, default=0.05, help="PMU angle noise std in deg (default: 0.05)")
    parser.add_argument("--sig_pseudo", type=float, default=0.10, help="Load forecast relative error std (default: 0.10)")
    parser.add_argument("--rho", type=float, default=0.90, help="AR(1) persistence coefficient (default: 0.90)")
    parser.add_argument("--mu", type=float, default=1.0, help="AR(1) long-term mean (default: 1.0)")
    parser.add_argument("--sigma", type=float, default=0.05, help="AR(1) innovation std (default: 0.05)")
    parser.add_argument(
        "--levels",
        nargs="+",
        type=float,
        default=[0.1, 0.2, 0.3, 0.4, 0.5],
        help="Dropout fractions (default: [0.1, 0.2, 0.3, 0.4, 0.5])",
    )
    parser.add_argument("--seed", type=int, default=0, help="Master random seed (default: 0)")
    args = parser.parse_args()

    for system in args.systems:
        print(f"\n=======================================================")
        print(f" Generating Sequential Dataset for: {system}")
        print(f"=======================================================")
        net = build_net(system)
        nb = len(net.bus)
        slack = int(np.where(net.bus.index == net.ext_grid.bus.iloc[0])[0][0])
        p0 = net.load["p_mw"].values.copy()
        q0 = net.load["q_mvar"].values.copy()
        Y = get_ybus(net)

        # Strategic PMU placement across non-slack candidate buses (50% coverage)
        place_rng = np.random.default_rng(args.seed)
        candidate_buses = np.array([i for i in range(nb) if i != slack])
        k_pmu = int(round(args.pmu_frac * len(candidate_buses)))
        pmu_idx = np.sort(place_rng.choice(candidate_buses, k_pmu, replace=False))

        print(f"System buses: {nb} | Slack bus index: {slack}")
        print(f"PMU count: {k_pmu} ({args.pmu_frac:.0%}) installed at buses: {pmu_idx.tolist()}")
        print(f"AR(1) Parameters: rho={args.rho}, mu={args.mu}, sigma={args.sigma}, step={args.step_hours}h")

        outdir = os.path.abspath(os.path.join(args.out, system))
        os.makedirs(outdir, exist_ok=True)
        print(f"Output directory: {outdir}")

        # Independent RNG streams per split to ensure clean data partitioning
        rng = {
            s: np.random.default_rng(args.seed + 1000 * (i + 1))
            for i, s in enumerate(["train", "val", "test"])
        }

        def sim_and_measure(split_name: str, n_steps: int):
            print(f"\n  [{split_name.upper()}] Simulating {n_steps} sequential steps...")
            sim_data = simulate_sequential_snapshots(
                net=net,
                n=n_steps,
                rng=rng[split_name],
                p0=p0,
                q0=q0,
                step_hours=args.step_hours,
                h0=args.h0,
                rho=args.rho,
                mu=args.mu,
                sigma=args.sigma,
            )
            max_err = check_physics(Y, sim_data, slack)
            print(f"  [{split_name.upper()}] AC Power Flow check max |S_calc - S_inj|: {max_err:.2e} pu")
            if max_err > 1e-5:
                print(f"  WARNING: Physical residual error {max_err:.2e} exceeds typical threshold 1e-6.")

            meas_data = make_measurements(
                sim=sim_data,
                pmu_idx=pmu_idx,
                slack=slack,
                rng=rng[split_name],
                sig_vm=args.sig_vm,
                sig_va_deg=args.sig_va,
                sig_pseudo=args.sig_pseudo,
            )
            return sim_data, meas_data

        # ------------------------------------------------------------------
        # A. TRAIN SPLIT: Full telemetry (mask = all 1.0)
        # ------------------------------------------------------------------
        train_sim, train_meas = sim_and_measure("train", args.n_train)
        train_mask = np.ones((args.n_train, k_pmu), dtype=np.float32)
        train_pkg = pack(train_sim, train_meas, train_mask, 0.0)
        train_path = os.path.join(outdir, "train.npz")
        np.savez_compressed(train_path, **train_pkg)
        print(f"  -> Saved {train_path} ({args.n_train} steps)")

        # ------------------------------------------------------------------
        # B. VAL SPLIT: Random dropout levels per snapshot from {10..50}%
        # ------------------------------------------------------------------
        val_sim, val_meas = sim_and_measure("val", args.n_val)
        val_lvls = rng["val"].choice(args.levels, args.n_val)
        val_mask = make_mask(rng["val"], args.n_val, k_pmu, val_lvls)
        val_pkg = pack(val_sim, val_meas, val_mask, val_lvls)
        val_path = os.path.join(outdir, "val.npz")
        np.savez_compressed(val_path, **val_pkg)
        print(f"  -> Saved {val_path} ({args.n_val} steps)")

        # ------------------------------------------------------------------
        # C. TEST SPLITS: Paired test files (Identical state trajectory, varying dropout)
        # ------------------------------------------------------------------
        test_sim, test_meas = sim_and_measure("test", args.n_test)
        for lvl in args.levels:
            lvl_pct = int(round(lvl * 100))
            test_mask = make_mask(rng["test"], args.n_test, k_pmu, lvl)
            test_pkg = pack(test_sim, test_meas, test_mask, lvl)
            test_path = os.path.join(outdir, f"test_drop{lvl_pct}.npz")
            np.savez_compressed(test_path, **test_pkg)
            print(f"  -> Saved {test_path} ({args.n_test} steps, {lvl_pct}% dropout)")

        # ------------------------------------------------------------------
        # D. META INFORMATION: Network Ybus, topology, and parameters
        # ------------------------------------------------------------------
        meta_path = os.path.join(outdir, "meta.npz")
        np.savez(
            meta_path,
            Ybus_real=Y.real.astype(np.float64),
            Ybus_imag=Y.imag.astype(np.float64),
            slack=slack,
            pmu_idx=pmu_idx,
            n_bus=nb,
            base_mva=float(net.sn_mva),
            sig_vm=float(args.sig_vm),
            sig_va_deg=float(args.sig_va),
            sig_pseudo=float(args.sig_pseudo),
            step_hours=float(args.step_hours),
            rho=float(args.rho),
            mu=float(args.mu),
            sigma=float(args.sigma),
        )
        print(f"  -> Saved {meta_path}")
        print(f"\nSequential dataset generation completed successfully at: {outdir}\n")


if __name__ == "__main__":
    main()
