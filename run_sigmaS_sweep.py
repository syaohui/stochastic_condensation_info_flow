"""
run_sigmaS_sweep.py — Sweep σ_S at fixed γ (Fig. 3 of the paper).

For each σ_S, runs both Model 1 and Model 2 for N_STEPS steps and
records the mean-radius and std-radius histories.  Relative dispersion
is NOT stored; it is recovered as sigma_r / <r> at plot time.  The
steady-state values are computed in the plotting script.

By default the sweep is run for BOTH γ = 0.1 and γ = 10.0 s^-1 (one
output file each); pass --gamma to run a single value.

Parallelism strategy
--------------------
Parallelized over ALL (γ, σ_S) combinations in a single ProcessPoolExecutor,
so the default two-γ sweep dispatches 2 × 20 = 40 independent tasks at once
rather than one γ at a time.  Each (σ_S, seed) pair uses
np.random.default_rng(seed) — a local Generator object that never touches the
global numpy random state, so results are bit-for-bit identical regardless of
worker dispatch order.

Usage
-----
    python run_sigmaS_sweep.py
    python run_sigmaS_sweep.py --gamma 0.1
    python run_sigmaS_sweep.py --outdir data/sweeps --workers 40

Output
------
    sigmaS_sweep_gamma{gamma}.npz  —  sigmaS_values, gamma, and for each
    σ_S (tag e.g. "s0p010"):
        s<tag>_MR1, s<tag>_MR2   mean radius <r> [um]  (n_seeds, N_STEPS)
        s<tag>_SR1, s<tag>_SR2   std radius sigma_r    (n_seeds, N_STEPS)
        s<tag>_time              time axis [s]         (N_STEPS,)
    Model 1 = white-noise limit, Model 2 = finite-memory OU.
    Histories are float32; all computation is float64.
"""

import argparse
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm.auto import tqdm
import os

from config import (
    N_c, rho_w, kT, ND_eff, UM3_TO_CM3,
    initial_L, z0, SEEDS, DT_FACTOR, N_STEPS,
    compute_S_mean, compute_DZ, compute_tau,
)


# ============================================================
# RNG-aware helpers  (accept rng instead of using global state)
# ============================================================

def _compute_r2_m1(chunk, v_depl, sqrt_2DZ, dt, rng):
    """Model 1: explicit Euler–Maruyama step for z = r^2."""
    dW = rng.standard_normal(size=chunk.shape) * np.sqrt(dt)
    return chunk**2 + v_depl * dt + sqrt_2DZ * dW


def _apply_boundary(r2, initial_volume, rng):
    """
    Enforce physical constraints:
      1. Droplets with z = r^2 <= 0 have fully evaporated.  They are reborn
         with volume drawn uniformly on (0, V0), where V0 = <V> is the mean
         droplet volume fixed by the conserved liquid water content and
         droplet number concentration.  In the r^3 units used here,
         V0 = initial_volume / N.  The rule involves no time step and no
         ensemble size.
      2. Conserve total liquid water volume by a global rescaling.
      3. Droplet number is conserved exactly (one crosser in, one rebirth out).
    Returns updated r array.
    """
    n_total = r2.size
    negative_mask = r2 <= 0
    num_neg = int(np.sum(negative_mask))

    r = np.empty(n_total, dtype=np.float64)
    r[~negative_mask] = np.sqrt(r2[~negative_mask])

    if num_neg > 0:
        v_max = initial_volume / n_total      # = V0, dt- and N-free
        r[negative_mask] = np.cbrt(v_max * rng.random(num_neg))

    r *= np.cbrt(initial_volume / np.sum(r**3))
    return r


# ============================================================
# Per-(σ_S, seed) worker  — no global random state
# ============================================================

def run_one_sigmaS(sigmaS, gamma, seed, N=100_000, steps=N_STEPS):
    """Run Model 1 & Model 2 for one (σ_S, seed) pair at fixed γ.

    Returns, for each model, the per-step histories of mean radius and
    std radius, plus the time axis:
        MR1, SR1, MR2, SR2, time_axis
    Relative dispersion is not stored (recover it as SR/MR at plot time).
    """
    r0 = np.sqrt(z0)
    S_mean = compute_S_mean(sigmaS, gamma)
    DZ     = compute_DZ(sigmaS, gamma)
    tau    = compute_tau(sigmaS, gamma)
    dt     = DT_FACTOR * tau

    v_depl   = kT * (S_mean - 1.0)
    sqrt_2DZ = np.sqrt(2.0 * DZ)
    time_axis = np.arange(steps, dtype=np.float64) * dt

    # ── Model 1 ──
    rng1 = np.random.default_rng(seed)
    r1 = np.full(N, r0, dtype=np.float64)
    vol_init = np.sum(r1**3)
    MR1 = np.empty(steps, dtype=np.float64)
    SR1 = np.empty(steps, dtype=np.float64)

    for step in range(steps):
        r1_sq = _compute_r2_m1(r1, v_depl, sqrt_2DZ, dt, rng1)
        r1 = _apply_boundary(r1_sq, vol_init, rng1)
        MR1[step] = np.mean(r1)
        SR1[step] = np.std(r1)

    # ── Model 2  (same seed → same starting sequence as Model 1) ──
    rng2 = np.random.default_rng(seed)
    r2 = np.full(N, r0, dtype=np.float64)
    S  = np.full(N, S_mean, dtype=np.float64)
    vol_init2 = np.sum(r2**3)
    MR2 = np.empty(steps, dtype=np.float64)
    SR2 = np.empty(steps, dtype=np.float64)

    for step in range(steps):
        # Implicit Euler–Maruyama for S
        noise = sigmaS * np.sqrt(2.0 * gamma * dt) * rng2.standard_normal(size=N)
        S = (S + gamma * S_mean * dt + noise) / (1.0 + gamma * dt)
        # Deterministic dz
        dz = kT * (S - 1.0) * dt
        r2_sq = r2**2 + dz
        r2 = _apply_boundary(r2_sq, vol_init2, rng2)
        MR2[step] = np.mean(r2)
        SR2[step] = np.std(r2)

    return MR1, SR1, MR2, SR2, time_axis


# ============================================================
# Per-σ_S task  (submitted to ProcessPoolExecutor)
# ============================================================

def _process_one_sigmaS(args):
    """Worker entrypoint: run all seeds for one σ_S and return packed
    results."""
    sigmaS, gamma, seeds, steps = args
    n_seeds = len(seeds)
    MR1_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    SR1_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    MR2_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    SR2_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    t_axis = None

    for i, seed in enumerate(seeds):
        MR1, SR1, MR2, SR2, t = run_one_sigmaS(sigmaS, gamma, seed,
                                               steps=steps)
        MR1_seeds[i] = MR1
        SR1_seeds[i] = SR1
        MR2_seeds[i] = MR2
        SR2_seeds[i] = SR2
        t_axis = t

    return sigmaS, MR1_seeds, SR1_seeds, MR2_seeds, SR2_seeds, t_axis


# ============================================================
# Write one sweep file at fixed γ
# ============================================================

def _write_sweep(gamma, sigmaS_values, gathered, outdir):
    """Assemble in sorted σ_S order and save the npz for one γ."""
    results = dict(sigmaS_values=sigmaS_values, gamma=gamma)

    for sigmaS in sigmaS_values:
        tag = f"s{sigmaS:.3f}".replace('.', 'p')
        MR1_seeds, SR1_seeds, MR2_seeds, SR2_seeds, t_axis = gathered[sigmaS]
        results[f"{tag}_MR1"]  = MR1_seeds.astype(np.float32)
        results[f"{tag}_MR2"]  = MR2_seeds.astype(np.float32)
        results[f"{tag}_SR1"]  = SR1_seeds.astype(np.float32)
        results[f"{tag}_SR2"]  = SR2_seeds.astype(np.float32)
        results[f"{tag}_time"] = t_axis.astype(np.float32)
        print(f"  γ={gamma}  σ_S={sigmaS:.3f}  "
              f"<r>1={MR1_seeds.mean(axis=0)[-1]:.4f}  "
              f"σr1={SR1_seeds.mean(axis=0)[-1]:.4f}  "
              f"<r>2={MR2_seeds.mean(axis=0)[-1]:.4f}  "
              f"σr2={SR2_seeds.mean(axis=0)[-1]:.4f}")

    out_path = os.path.join(outdir, f"sigmaS_sweep_gamma{gamma}.npz")
    np.savez_compressed(out_path, **results)
    print(f"\nSaved → {out_path}  "
          f"({os.path.getsize(out_path) / (1024**2):.1f} MB)")


# ============================================================
# All sweeps — one pool over every (γ, σ_S) combination
# ============================================================

def run_sweep(gammas, sigmaS_values, outdir, workers):
    tasks = [(sigmaS, gamma, SEEDS, N_STEPS)
             for gamma in gammas
             for sigmaS in sigmaS_values]

    print(f"Launching σ_S sweep for γ={list(gammas)}  "
          f"({len(tasks)} tasks = {len(gammas)} γ × {len(sigmaS_values)} σ_S"
          f" × {len(SEEDS)} seeds  |  {workers} workers)")

    gathered = {gamma: {} for gamma in gammas}

    with ProcessPoolExecutor(max_workers=workers) as executor:
        # task[1] is γ — used to bin each result into the right output file
        futures = {executor.submit(_process_one_sigmaS, task): task[1]
                   for task in tasks}

        with tqdm(total=len(tasks), desc='sigmaS sweep (all γ)') as pbar:
            for future in as_completed(futures):
                gamma = futures[future]
                sigmaS, MR1_seeds, SR1_seeds, MR2_seeds, SR2_seeds, t_axis = \
                    future.result()
                gathered[gamma][sigmaS] = (MR1_seeds, SR1_seeds,
                                           MR2_seeds, SR2_seeds, t_axis)
                pbar.update(1)
                pbar.set_postfix(
                    gamma=f'{gamma}',
                    sigmaS=f'{sigmaS:.3f}',
                    r1=f'{MR1_seeds.mean(axis=0)[-1]:.4f}',
                    r2=f'{MR2_seeds.mean(axis=0)[-1]:.4f}',
                )

    for gamma in gammas:
        _write_sweep(gamma, sigmaS_values, gathered[gamma], outdir)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='sigmaS sweep (Fig. 3)')
    parser.add_argument('--outdir',  type=str, default='data/sweeps')
    parser.add_argument('--gamma',   type=float, default=None,
                        help='Run a single γ (default: both 0.1 and 10.0)')
    parser.add_argument('--workers', type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    # σ_S from 0.001 to 0.020 in steps of 0.001  (20 values).
    sigmaS_values = np.round(np.arange(0.001, 0.0201, 0.001), 3)
    os.makedirs(args.outdir, exist_ok=True)

    gammas = [args.gamma] if args.gamma is not None else [0.1, 10.0]
    run_sweep(gammas, sigmaS_values, args.outdir, args.workers)


if __name__ == '__main__':
    main()
