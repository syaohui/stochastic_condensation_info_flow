"""
run_simulation.py — Monte Carlo simulation of Model 1 and Model 2.

Usage
-----
    python run_simulation.py --gamma 0.1
    python run_simulation.py --gamma 10.0
    python run_simulation.py              # runs both gammas, all seeds, 10 parallel jobs

Outputs  (saved to  data/<case_label>/)
-------
- summary_gamma<γ>_seed<s>.npz : Lightweight summary sufficient for ALL figures
                                  (statistics, histogram data, covariances for IF)

The *summary* file is what goes on Zenodo.

References
----------
Shu et al. (2026), GRL;  McGraw & Liu (2006), GRL.
"""

import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from tqdm.auto import tqdm
import os

from config import (
    N_c, rho_w, kT, ND_eff, UM3_TO_CM3,
    N_droplets as N, initial_L, z0,
    SEEDS, DT_FACTOR, N_STEPS, HIST_RMAX, HIST_NBINS,
    compute_S_mean, compute_DZ, compute_tau,
)


# ============================================================
# Boundary handling  (conserves N_c and L)
# ============================================================
def _apply_boundary(r2, initial_volume):
    """
    Enforce physical constraints:
      1. Droplets with z = r^2 <= 0 have fully evaporated.  They are reborn
         with volume drawn uniformly on (0, V0), where V0 = <V> is the mean
         droplet volume fixed by the conserved liquid water content and
         droplet number concentration.  In the r^3 units used here,
         V0 = initial_volume / N.  The rule involves no time step and no
         ensemble size.
      2. Conserve total liquid water volume by a global scaling.
      3. Droplet number is conserved exactly (one crosser in, one rebirth out).
    Returns updated r array.
    """
    n_total = r2.size
    negative_mask = r2 <= 0
    num_negative = int(np.sum(negative_mask))

    r = np.empty(n_total, dtype=np.float64)
    r[~negative_mask] = np.sqrt(r2[~negative_mask])

    if num_negative > 0:
        v_max = initial_volume / n_total      # = V0, dt- and N-free
        r[negative_mask] = np.cbrt(v_max * np.random.rand(num_negative))

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
    nr = N_c * (counts / N_total) / dr
    return nr


# ============================================================
# Model 1 simulation
# ============================================================
def run_model1(gamma, sigmaS, seed, save_dir='data'):
    """Run Model 1 for one seed and return summary."""
    np.random.seed(seed)

    S_mean = compute_S_mean(sigmaS, gamma)
    DZ     = compute_DZ(sigmaS, gamma)
    tau    = compute_tau(sigmaS, gamma)

    dt      = DT_FACTOR * tau
    n_steps = N_STEPS
    total_time = n_steps * dt
    time_axis  = np.linspace(0, total_time, n_steps)

    v_depl   = kT * (S_mean - 1.0)
    sqrt_2DZ = np.sqrt(2.0 * DZ)
    sqrt_dt  = np.sqrt(dt)

    r = np.full(N, np.sqrt(z0), dtype=np.float64)
    initial_volume = np.sum(r**3)

    mean_r_arr     = np.zeros(n_steps)
    std_r_arr      = np.zeros(n_steps)
    eps_arr        = np.zeros(n_steps)
    beta_arr       = np.zeros(n_steps)
    L_arr          = np.zeros(n_steps)
    dL_over_L_arr  = np.zeros(n_steps)
    n_neg_frac_arr = np.zeros(n_steps)

    radius_bins = np.linspace(0, HIST_RMAX, HIST_NBINS)
    hist_data   = np.zeros((HIST_NBINS - 1, n_steps), dtype=np.float32)

    for step in range(n_steps):
        L_arr[step] = ND_eff * (4.0 * np.pi / 3.0) * np.sum(r**3) / N
        stats = _step_stats(r)
        mean_r_arr[step] = stats['mean_r']
        std_r_arr[step]  = stats['std_r']
        eps_arr[step]    = stats['eps']
        beta_arr[step]   = stats['beta']
        hist_data[:, step] = _build_histogram(r, radius_bins, N)

        r2 = r**2 + v_depl * dt + sqrt_2DZ * np.random.normal(0.0, 1.0, size=N) * sqrt_dt

        neg_mask = r2 <= 0
        n_neg_frac_arr[step] = np.sum(neg_mask) / N
        pos_vol  = np.sum(np.sqrt(r2[~neg_mask])**3)
        dL_over_L_arr[step]  = (initial_volume - pos_vol) / initial_volume

        r = _apply_boundary(r2, initial_volume)

    summary = dict(
        time_axis=time_axis,
        mean_r=mean_r_arr, std_r=std_r_arr,
        eps=eps_arr, beta=beta_arr, L=L_arr,
        dL_over_L=dL_over_L_arr, n_neg_frac=n_neg_frac_arr,
        hist_data=hist_data, radius_bins=radius_bins,
        gamma=gamma, sigmaS=sigmaS, N=N,
        N_c=N_c, rho_w=rho_w, kT=kT, ND_eff=ND_eff,
        initial_L=initial_L, S_mean=S_mean,
        tau=tau, dt=dt, n_steps=n_steps,
        total_time=total_time,
    )
    return summary


# ============================================================
# Model 2 simulation
# ============================================================
def run_model2(gamma, sigmaS, seed, save_dir='data'):
    """Run Model 2 for one seed and return summary."""
    np.random.seed(seed)

    S_mean = compute_S_mean(sigmaS, gamma)
    DZ     = compute_DZ(sigmaS, gamma)
    tau    = compute_tau(sigmaS, gamma)

    dt      = DT_FACTOR * tau
    n_steps = N_STEPS
    M_STEP  = 100
    n_if    = n_steps // M_STEP
    total_time = n_steps * dt
    time_axis  = np.linspace(0, total_time, n_steps)

    r = np.full(N, np.sqrt(z0), dtype=np.float64)
    S = np.full(N, S_mean, dtype=np.float64)
    initial_volume = np.sum(r**3)

    mean_r_arr     = np.zeros(n_steps)
    std_r_arr      = np.zeros(n_steps)
    eps_arr        = np.zeros(n_steps)
    beta_arr       = np.zeros(n_steps)
    L_arr          = np.zeros(n_steps)
    dL_over_L_arr  = np.zeros(n_steps)
    n_neg_frac_arr = np.zeros(n_steps)

    radius_bins = np.linspace(0, HIST_RMAX, HIST_NBINS)
    hist_data   = np.zeros((HIST_NBINS - 1, n_steps), dtype=np.float32)

    # IF covariance arrays — sampled every M_STEP steps only
    C_zz   = np.zeros(n_if)
    C_SS   = np.zeros(n_if)
    C_zS   = np.zeros(n_if)
    rho_zS = np.zeros(n_if)
    C_S_dz = np.zeros(n_if - 1)
    C_z_dS = np.zeros(n_if - 1)
    C_S_dS = np.zeros(n_if - 1)

    z_sample_prev = (r**2).copy()
    S_sample_prev = S.copy()

    for step in range(n_steps):
        z_current = r**2

        L_arr[step] = ND_eff * (4.0 * np.pi / 3.0) * np.sum(r**3) / N
        stats = _step_stats(r)
        mean_r_arr[step] = stats['mean_r']
        std_r_arr[step]  = stats['std_r']
        eps_arr[step]    = stats['eps']
        beta_arr[step]   = stats['beta']
        hist_data[:, step] = _build_histogram(r, radius_bins, N)

        # ── IF covariances: sampled every M_STEP steps ──
        if step % M_STEP == 0:
            k         = step // M_STEP
            z_mean_k  = np.mean(z_current)
            S_mean_k  = np.mean(S)
            dz_ens    = z_current - z_mean_k
            dS_ens    = S - S_mean_k
            C_zz[k]   = np.mean(dz_ens**2)
            C_SS[k]   = np.mean(dS_ens**2)
            C_zS[k]   = np.mean(dz_ens * dS_ens)
            denom     = np.sqrt(C_zz[k] * C_SS[k])
            rho_zS[k] = C_zS[k] / denom if denom > 0 else 0.0

            if k > 0:
                m_dt  = M_STEP * dt   # physical window — stable as dt→0
                dz_dt = (z_current    - z_sample_prev) / m_dt
                dS_dt = (S            - S_sample_prev) / m_dt
                dz_sp = z_sample_prev - np.mean(z_sample_prev)
                dS_sp = S_sample_prev - np.mean(S_sample_prev)
                C_S_dz[k-1] = np.mean(dS_sp * (dz_dt - np.mean(dz_dt)))
                C_z_dS[k-1] = np.mean(dz_sp * (dS_dt - np.mean(dS_dt)))
                C_S_dS[k-1] = np.mean(dS_sp * (dS_dt - np.mean(dS_dt)))

            z_sample_prev = z_current.copy()
            S_sample_prev = S.copy()

        # ── S update (implicit Euler) ──
        noise = sigmaS * np.sqrt(2.0 * gamma * dt) * np.random.normal(size=N)
        S = (S + gamma * compute_S_mean(sigmaS, gamma) * dt + noise) / (1.0 + gamma * dt)

        # ── z update ──
        dz_step = kT * (S - 1.0) * dt
        r2 = r**2 + dz_step

        neg_mask = r2 <= 0
        n_neg_frac_arr[step] = np.sum(neg_mask) / N
        pos_vol  = np.sum(np.sqrt(r2[~neg_mask])**3)
        dL_over_L_arr[step]  = (initial_volume - pos_vol) / initial_volume

        r = _apply_boundary(r2, initial_volume)

    # rho_zS_full: interpolate sampled rho_zS back to full time axis for plotting
    rho_zS_full = np.repeat(rho_zS, M_STEP)[:n_steps]

    summary = dict(
        time_axis=time_axis,
        mean_r=mean_r_arr, std_r=std_r_arr,
        eps=eps_arr, beta=beta_arr, L=L_arr,
        dL_over_L=dL_over_L_arr, n_neg_frac=n_neg_frac_arr,
        hist_data=hist_data, radius_bins=radius_bins,
        C_zz=C_zz, C_SS=C_SS, C_zS=C_zS, rho_zS=rho_zS,
        rho_zS_full=rho_zS_full,
        C_S_dz=C_S_dz, C_z_dS=C_z_dS, C_S_dS=C_S_dS,
        M_STEP=M_STEP,
        gamma=gamma, sigmaS=sigmaS, N=N,
        N_c=N_c, rho_w=rho_w, kT=kT, ND_eff=ND_eff,
        initial_L=initial_L,
        S_mean=compute_S_mean(sigmaS, gamma),
        tau=tau, dt=dt, n_steps=n_steps,
        total_time=total_time,
    )
    return summary


# ============================================================
# Worker: one (gamma, seed) combination
# M1 and M2 run in parallel via ThreadPoolExecutor (2 cores per job)
# ============================================================
def _run_one_job(args):
    """Top-level worker — runs Model 1 and Model 2 in parallel for one (gamma, seed) pair."""
    gamma, sigmaS, seed, save_dir, position = args

    job_save_dir = os.path.join(save_dir, f"gamma{gamma}")
    os.makedirs(job_save_dir, exist_ok=True)

    desc = f"γ={gamma} s={seed}"

    with tqdm(total=2, desc=desc, position=position,
              leave=False, ncols=80) as pbar:
        pbar.set_postfix(status='M1+M2 running')
        with ThreadPoolExecutor(max_workers=2) as thread_pool:
            f1 = thread_pool.submit(run_model1, gamma, sigmaS, seed, job_save_dir)
            f2 = thread_pool.submit(run_model2, gamma, sigmaS, seed, job_save_dir)
            summary_m1 = f1.result()
            pbar.update(1)
            summary_m2 = f2.result()
            pbar.update(1)

    # keys used for information flow estimation — must stay float64
    IF_KEYS = {'C_zz', 'C_SS', 'C_zS', 'rho_zS', 'C_S_dz', 'C_z_dS', 'C_S_dS'}

    combined = {}
    for prefix, d in [('m1_', summary_m1), ('m2_', summary_m2)]:
        for k, v in d.items():
            if isinstance(v, np.ndarray) and np.issubdtype(v.dtype, np.floating):
                if k in IF_KEYS:
                    combined[prefix + k] = v.astype(np.float64)
                else:
                    combined[prefix + k] = v.astype(np.float32)
            else:
                combined[prefix + k] = v

    out_path = os.path.join(job_save_dir, f"summary_gamma{gamma}_seed{seed}.npz")
    np.savez_compressed(out_path, **combined)
    size_mb = os.path.getsize(out_path) / (1024**2)
    return gamma, seed, out_path, size_mb


# ============================================================
# Main entry point
# ============================================================
def main():
    parser = argparse.ArgumentParser(
        description='Run stochastic condensation simulations (Model 1 & 2).')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Single γ to run (default: both 0.1 and 10.0)')
    parser.add_argument('--sigmaS', type=float, default=0.01,
                        help='Std dev of S fluctuation (default: 0.01)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Single seed to run (default: all seeds from config)')
    parser.add_argument('--outdir', type=str, default='data',
                        help='Output directory (default: data/)')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: number of jobs)')
    args = parser.parse_args()

    sigmaS = args.sigmaS

    # Determine which gammas and seeds to run
    gammas_to_run = [args.gamma] if args.gamma is not None else [0.1, 10.0]
    seeds_to_run  = [args.seed]  if args.seed  is not None else list(SEEDS)

    # Build all (gamma, seed) job combinations — position 0 reserved for outer bar
    all_jobs = [
        (gamma, sigmaS, seed, args.outdir, i + 1)
        for i, (gamma, seed) in enumerate(
            (g, s)
            for g in gammas_to_run
            for s in seeds_to_run
        )
    ]

    n_jobs    = len(all_jobs)
    n_workers = args.workers if args.workers is not None else n_jobs

    print(f"\nTotal jobs : {n_jobs}  "
          f"({len(gammas_to_run)} gamma(s) × {len(seeds_to_run)} seed(s))")
    print(f"Workers    : {n_workers}")
    print(f"Jobs       : {[(g, s) for g, _, s, _, _ in all_jobs]}\n")

    os.makedirs(args.outdir, exist_ok=True)

    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_one_job, job): job
                   for job in all_jobs}

        with tqdm(total=n_jobs, desc='Total', position=0,
                  ncols=80) as pbar:
            for future in as_completed(futures):
                gamma, seed, out_path, size_mb = future.result()
                tqdm.write(f"  ✓ gamma={gamma}  seed={seed}  "
                           f"→ {out_path}  ({size_mb:.1f} MB)")
                pbar.update(1)

    print("\nDone.")


if __name__ == '__main__':
    main()
