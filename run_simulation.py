"""
run_simulation.py — Monte Carlo simulation of Model 1 and Model 2.

Usage
-----
    python run_simulation.py --gamma 0.3
    python run_simulation.py --gamma 8.0

Outputs  (saved to  data/<case_label>/)
-------
- model1_gamma<γ>.npz   : Model 1 full raw data  (r_evolution, ...)
- model2_gamma<γ>.npz   : Model 2 full raw data  (r_evolution, S_evolution, ...)
- summary_gamma<γ>.npz  : Lightweight summary sufficient for ALL figures
                          (statistics, histogram data, covariances for IF)

The *summary* file is what goes on Zenodo (~tens of MB).
The *full raw* files are optional / local-only (10–20 GB each).

References
----------
Shu et al. (2025), GRL;  McGraw & Liu (2006), GRL.
"""

import argparse
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
from tqdm.auto import tqdm
import os
from datetime import datetime

from config import (
    N_c, rho_w, kT, ND_eff, UM3_TO_CM3,
    N_droplets as N, initial_L, z0,
    RANDOM_SEED, HIST_RMAX, HIST_NBINS,
    compute_S_mean, compute_DZ, compute_tau,
)

# ============================================================
# Worker functions (top-level for pickling / threading)
# ============================================================
def _compute_r2_model1(chunk, v_depl, sqrt_2DZ, dt):
    """Model 1: explicit Euler–Maruyama step for z = r^2."""
    dW = np.random.normal(0.0, 1.0, size=chunk.shape) * np.sqrt(dt)
    return chunk**2 + v_depl * dt + sqrt_2DZ * dW


def _compute_r2_model2(r_chunk, dz_chunk):
    """Model 2: deterministic z-update using pre-computed dz."""
    return r_chunk**2 + dz_chunk


# ============================================================
# Boundary handling  (conserves N_c and L)
# ============================================================
def _apply_boundary(r2, initial_volume):
    """
    Enforce physical constraints:
      1. No negative z (= r^2).
      2. Conserve total liquid water volume.
    Returns updated r array.
    """
    negative_mask = r2 <= 0
    num_negative = int(np.sum(negative_mask))
    positive_r = np.sqrt(r2[~negative_mask])
    current_volume = np.sum(positive_r**3)
    lost_volume = initial_volume - current_volume

    if num_negative == 0:
        r = np.sqrt(r2)
        r *= np.cbrt(initial_volume / current_volume)
    elif lost_volume <= 0:
        r2 = np.where(negative_mask, -r2, r2)
        r = np.sqrt(r2)
        r *= np.cbrt(initial_volume / np.sum(r**3))
    else:
        r_max = lost_volume * 2 / num_negative
        r2[negative_mask] = (r_max * np.random.rand(num_negative)) ** (2.0/3.0)
        r = np.sqrt(r2)
        r *= np.cbrt(initial_volume / np.sum(r**3))
    return r


# ============================================================
# Per-step statistics
# ============================================================
def _step_stats(r):
    """Return dict of per-step summary statistics from radius array."""
    mean_r = np.mean(r)
    std_r  = np.std(r)
    r2_arr = r**2
    r3_arr = r**3
    second_mom = np.mean(r2_arr)
    third_mom  = np.mean(r3_arr)
    eps = std_r / mean_r if mean_r > 0 else 0.0
    # Effective radius ratio β = <r^3>/<r^2> / <r^3>^{1/3}
    beta = (third_mom / second_mom) / (third_mom ** (1.0/3.0)) if second_mom > 0 else 1.0
    return dict(mean_r=mean_r, std_r=std_r, eps=eps, beta=beta,
                second_mom=second_mom, third_mom=third_mom)


# ============================================================
# Histogram builder
# ============================================================
def _build_histogram(r, radius_bins, N_total):
    """
    Build n(r) histogram in physical units [cm^{-3} μm^{-1}].
    n(r) = N_c * (counts / N_total) / Δr
    """
    counts, _ = np.histogram(r, bins=radius_bins, density=False)
    dr = np.diff(radius_bins)
    nr = N_c * (counts / N_total) / dr    # [cm^{-3} μm^{-1}]
    return nr


# ============================================================
# Model 1 simulation
# ============================================================
def run_model1(gamma, sigmaS, save_raw=True, save_dir='data'):
    """Run Model 1 and return (or save) results."""
    print(f"\n{'='*60}")
    print(f"  MODEL 1   γ = {gamma}  σ_S = {sigmaS}")
    print(f"{'='*60}")

    np.random.seed(RANDOM_SEED)

    # Derived parameters
    S_mean = compute_S_mean(sigmaS, gamma)
    DZ     = compute_DZ(sigmaS, gamma)
    tau    = compute_tau(sigmaS, gamma)

    dt     = 0.001 * tau
    n_steps = 15_000
    total_time = n_steps * dt
    time_axis  = np.linspace(0, total_time, n_steps)

    v_depl   = kT * (S_mean - 1.0)
    sqrt_2DZ = np.sqrt(2.0 * DZ)

    # Initial conditions
    r = np.full(N, np.sqrt(z0), dtype=np.float64)
    initial_volume = np.sum(r**3)

    # Storage: raw (optional) and summary (always)
    if save_raw:
        r_evolution = np.zeros((N, n_steps), dtype=np.float32)

    # Summary statistics
    mean_r_arr  = np.zeros(n_steps)
    std_r_arr   = np.zeros(n_steps)
    eps_arr     = np.zeros(n_steps)
    beta_arr    = np.zeros(n_steps)
    L_arr       = np.zeros(n_steps)

    # Histogram storage
    radius_bins = np.linspace(0, HIST_RMAX, HIST_NBINS)
    hist_data   = np.zeros((HIST_NBINS - 1, n_steps), dtype=np.float32)

    # Thread pool
    num_workers = multiprocessing.cpu_count()
    executor = ThreadPoolExecutor(max_workers=num_workers)

    for step in tqdm(range(n_steps), desc='Model 1', unit='step'):
        # Store raw
        if save_raw:
            r_evolution[:, step] = r.astype(np.float32)

        # Summary
        L_arr[step] = ND_eff * (4.0 * np.pi / 3.0) * np.sum(r**3) / N
        stats = _step_stats(r)
        mean_r_arr[step] = stats['mean_r']
        std_r_arr[step]  = stats['std_r']
        eps_arr[step]    = stats['eps']
        beta_arr[step]   = stats['beta']

        # Histogram
        hist_data[:, step] = _build_histogram(r, radius_bins, N)

        # Euler–Maruyama step
        chunks = np.array_split(r, num_workers)
        futures = [executor.submit(_compute_r2_model1, ch, v_depl, sqrt_2DZ, dt)
                   for ch in chunks]
        r2 = np.concatenate([f.result() for f in futures])

        # Boundary
        r = _apply_boundary(r2, initial_volume)

    executor.shutdown()

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)

    # Summary file (lightweight — for Zenodo)
    summary = dict(
        time_axis=time_axis,
        mean_r=mean_r_arr, std_r=std_r_arr,
        eps=eps_arr, beta=beta_arr, L=L_arr,
        hist_data=hist_data, radius_bins=radius_bins,
        # Parameters
        gamma=gamma, sigmaS=sigmaS, N=N,
        N_c=N_c, rho_w=rho_w, kT=kT, ND_eff=ND_eff,
        initial_L=initial_L, S_mean=S_mean,
        tau=tau, dt=dt, n_steps=n_steps,
        total_time=total_time,
    )

    # Raw file (large — local only)
    if save_raw:
        raw_path = os.path.join(save_dir, f"model1_gamma{gamma}_raw.npz")
        np.savez_compressed(raw_path,
                            r_evolution=r_evolution,
                            **{k: v for k, v in summary.items()
                               if k not in ('hist_data', 'radius_bins')})
        print(f"  Raw data → {raw_path}  "
              f"({os.path.getsize(raw_path)/(1024**2):.1f} MB)")

    return summary


# ============================================================
# Model 2 simulation
# ============================================================
def run_model2(gamma, sigmaS, save_raw=True, save_dir='data'):
    """Run Model 2 and return (or save) results."""
    print(f"\n{'='*60}")
    print(f"  MODEL 2   γ = {gamma}  σ_S = {sigmaS}")
    print(f"{'='*60}")

    np.random.seed(RANDOM_SEED)

    # Derived parameters
    S_mean = compute_S_mean(sigmaS, gamma)
    DZ     = compute_DZ(sigmaS, gamma)
    tau    = compute_tau(sigmaS, gamma)

    dt     = 0.001 * tau
    n_steps = 15_000
    total_time = n_steps * dt
    time_axis  = np.linspace(0, total_time, n_steps)

    # Initial conditions
    r = np.full(N, np.sqrt(z0), dtype=np.float64)
    S = np.full(N, S_mean, dtype=np.float64)
    initial_volume = np.sum(r**3)

    # Storage: raw (optional) and summary (always)
    if save_raw:
        r_evolution = np.zeros((N, n_steps), dtype=np.float32)
        S_evolution = np.zeros((N, n_steps), dtype=np.float32)

    # Summary statistics
    mean_r_arr  = np.zeros(n_steps)
    std_r_arr   = np.zeros(n_steps)
    eps_arr     = np.zeros(n_steps)
    beta_arr    = np.zeros(n_steps)
    L_arr       = np.zeros(n_steps)

    # Histogram storage
    radius_bins = np.linspace(0, HIST_RMAX, HIST_NBINS)
    hist_data   = np.zeros((HIST_NBINS - 1, n_steps), dtype=np.float32)

    # ── Covariance arrays for information-flow estimation ──
    # At each time step, store ensemble (co)variances needed by Eqs. 8 & 9.
    C_zz   = np.zeros(n_steps)    # Var(z)
    C_SS   = np.zeros(n_steps)    # Var(S)
    C_zS   = np.zeros(n_steps)    # Cov(z, S)
    # For time-derivative covariances we need consecutive steps;
    # these are computed in a second pass or on-the-fly.
    C_S_dz = np.zeros(n_steps - 1)   # Cov(S_t, Δz_t/dt)
    C_z_dS = np.zeros(n_steps - 1)   # Cov(z_t, ΔS_t/dt)
    C_S_dS = np.zeros(n_steps - 1)   # Cov(S_t, ΔS_t/dt)

    # Correlation coefficient ρ_{zS}
    rho_zS = np.zeros(n_steps)

    # Thread pool
    num_workers = multiprocessing.cpu_count()
    executor = ThreadPoolExecutor(max_workers=num_workers)

    # Arrays to hold previous-step values for derivative covariances
    z_prev = r**2              # z = r^2
    S_prev = S.copy()

    for step in tqdm(range(n_steps), desc='Model 2', unit='step'):
        z_current = r**2

        # Store raw
        if save_raw:
            r_evolution[:, step] = r.astype(np.float32)
            S_evolution[:, step] = S.astype(np.float32)

        # Summary statistics
        L_arr[step] = ND_eff * (4.0 * np.pi / 3.0) * np.sum(r**3) / N
        stats = _step_stats(r)
        mean_r_arr[step] = stats['mean_r']
        std_r_arr[step]  = stats['std_r']
        eps_arr[step]    = stats['eps']
        beta_arr[step]   = stats['beta']

        # Histogram
        hist_data[:, step] = _build_histogram(r, radius_bins, N)

        # Ensemble (co)variances at this step
        z_mean = np.mean(z_current)
        S_mean_ens = np.mean(S)
        dz = z_current - z_mean
        dS_ens = S - S_mean_ens
        C_zz[step] = np.mean(dz**2)
        C_SS[step] = np.mean(dS_ens**2)
        C_zS[step] = np.mean(dz * dS_ens)
        denom = np.sqrt(C_zz[step] * C_SS[step])
        rho_zS[step] = C_zS[step] / denom if denom > 0 else 0.0

        # Time-derivative covariances (from step-1 → step)
        if step > 0:
            dz_dt = (z_current - z_prev) / dt
            dS_dt = (S - S_prev) / dt
            # Use previous-step S and z for covariances
            z_prev_mean = np.mean(z_prev)
            S_prev_mean = np.mean(S_prev)
            dz_prev = z_prev - z_prev_mean
            dS_prev = S_prev - S_prev_mean
            C_S_dz[step - 1] = np.mean(dS_prev * (dz_dt - np.mean(dz_dt)))
            C_z_dS[step - 1] = np.mean(dz_prev * (dS_dt - np.mean(dS_dt)))
            C_S_dS[step - 1] = np.mean(dS_prev * (dS_dt - np.mean(dS_dt)))

        # Save previous-step values
        z_prev = z_current.copy()
        S_prev = S.copy()

        # ── Model 2 update scheme ──
        # 1) Implicit Euler–Maruyama for S  (Eq. S7a)
        noise = sigmaS * np.sqrt(2.0 * gamma * dt) * np.random.normal(size=N)
        S = (S + gamma * compute_S_mean(sigmaS, gamma) * dt + noise) / (1.0 + gamma * dt)

        # 2) Deterministic dz for each droplet  (Eq. S7b)
        dz_step = kT * (S - 1.0) * dt

        # 3) Parallel r^2 update
        r_chunks  = np.array_split(r, num_workers)
        dz_chunks = np.array_split(dz_step, num_workers)
        futures = [executor.submit(_compute_r2_model2, rc, dzc)
                   for rc, dzc in zip(r_chunks, dz_chunks)]
        r2 = np.concatenate([f.result() for f in futures])

        # 4) Boundary
        r = _apply_boundary(r2, initial_volume)

    executor.shutdown()

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(save_dir, exist_ok=True)

    summary = dict(
        time_axis=time_axis,
        mean_r=mean_r_arr, std_r=std_r_arr,
        eps=eps_arr, beta=beta_arr, L=L_arr,
        hist_data=hist_data, radius_bins=radius_bins,
        # Covariance arrays for information flow
        C_zz=C_zz, C_SS=C_SS, C_zS=C_zS,
        C_S_dz=C_S_dz, C_z_dS=C_z_dS, C_S_dS=C_S_dS,
        rho_zS=rho_zS,
        # Parameters
        gamma=gamma, sigmaS=sigmaS, N=N,
        N_c=N_c, rho_w=rho_w, kT=kT, ND_eff=ND_eff,
        initial_L=initial_L,
        S_mean=compute_S_mean(sigmaS, gamma),
        tau=tau, dt=dt, n_steps=n_steps,
        total_time=total_time,
    )

    # Raw file (large — local only)
    if save_raw:
        raw_path = os.path.join(save_dir, f"model2_gamma{gamma}_raw.npz")
        np.savez_compressed(raw_path,
                            r_evolution=r_evolution,
                            S_evolution=S_evolution,
                            **{k: v for k, v in summary.items()
                               if k not in ('hist_data', 'radius_bins')})
        print(f"  Raw data → {raw_path}  "
              f"({os.path.getsize(raw_path)/(1024**2):.1f} MB)")

    return summary


# ============================================================
# Main entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Run stochastic condensation simulations (Model 1 & 2).')
    parser.add_argument('--gamma', type=float, required=True,
                        help='Correlation rate γ [s^{-1}]')
    parser.add_argument('--sigmaS', type=float, default=0.01,
                        help='Std dev of S fluctuation (default: 0.01)')
    parser.add_argument('--no-raw', action='store_true',
                        help='Skip saving large raw data files')
    parser.add_argument('--outdir', type=str, default='data',
                        help='Output directory (default: data/)')
    args = parser.parse_args()

    gamma  = args.gamma
    sigmaS = args.sigmaS
    save_raw = not args.no_raw
    save_dir = os.path.join(args.outdir, f"gamma{gamma}")

    # Run both models
    summary_m1 = run_model1(gamma, sigmaS, save_raw=save_raw, save_dir=save_dir)
    summary_m2 = run_model2(gamma, sigmaS, save_raw=save_raw, save_dir=save_dir)

    # Save combined summary (lightweight — for Zenodo)
    summary_path = os.path.join(save_dir, f"summary_gamma{gamma}.npz")
    combined = {}
    for prefix, d in [('m1_', summary_m1), ('m2_', summary_m2)]:
        for k, v in d.items():
            combined[prefix + k] = v
    np.savez_compressed(summary_path, **combined)
    print(f"\n  Summary → {summary_path}  "
          f"({os.path.getsize(summary_path)/(1024**2):.1f} MB)")
    print("\nDone.")


if __name__ == '__main__':
    main()
