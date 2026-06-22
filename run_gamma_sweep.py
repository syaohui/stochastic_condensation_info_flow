"""
run_gamma_sweep.py — Sweep γ from 0.1 to 10.0 (Fig. 2 of the paper).

For each γ, runs both Model 1 and Model 2 for N_STEPS steps and records
the relative-dispersion history.  The steady-state RD is computed in
the plotting script.

Parallelism strategy
--------------------
Parallelized over γ values using ProcessPoolExecutor. Each (γ, seed)
pair uses np.random.default_rng(seed) — a local Generator object that
never touches the global numpy random state, so results are bit-for-bit
identical regardless of worker dispatch order.

Usage
-----
    python run_gamma_sweep.py
    python run_gamma_sweep.py --outdir data/sweeps
    python run_gamma_sweep.py --workers 80

Output
------
    gamma_sweep_RD.npz  —  contains gamma_values, and for each γ:
        g<tag>_RD1, g<tag>_RD2, g<tag>_time  (tag = e.g. "0p10")
        all computation is performed in float64 for full precision.
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
      1. No negative z (= r^2).
      2. Conserve total liquid water volume.
    Returns updated r array.
    """
    negative_mask = r2 <= 0
    num_neg = int(np.sum(negative_mask))
    pos_r = np.sqrt(r2[~negative_mask])
    curr_vol = np.sum(pos_r**3)
    lost = initial_volume - curr_vol

    if num_neg == 0:
        r = np.sqrt(r2)
        r *= np.cbrt(initial_volume / curr_vol)
    elif lost <= 0:
        r2 = np.where(negative_mask, -r2, r2)
        r = np.sqrt(r2)
        r *= np.cbrt(initial_volume / np.sum(r**3))
    else:
        rmax = lost * 2 / num_neg
        r2[negative_mask] = (rmax * rng.random(num_neg)) ** (2.0 / 3.0)
        r = np.sqrt(r2)
        r *= np.cbrt(initial_volume / np.sum(r**3))
    return r


# ============================================================
# Per-(γ, seed) worker  — no global random state
# ============================================================

def run_one_gamma(gamma, seed, N=100_000, sigmaS=0.01, steps=N_STEPS):
    """Run Model 1 & Model 2 for one (γ, seed) pair."""
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
    RD1 = np.empty(steps, dtype=np.float64)

    for step in range(steps):
        r1_sq = _compute_r2_m1(r1, v_depl, sqrt_2DZ, dt, rng1)
        r1 = _apply_boundary(r1_sq, vol_init, rng1)
        RD1[step] = np.std(r1) / np.mean(r1)

    # ── Model 2  (same seed → same starting sequence as Model 1) ──
    rng2 = np.random.default_rng(seed)
    r2 = np.full(N, r0, dtype=np.float64)
    S  = np.full(N, S_mean, dtype=np.float64)
    vol_init2 = np.sum(r2**3)
    RD2 = np.empty(steps, dtype=np.float64)

    for step in range(steps):
        # Implicit Euler–Maruyama for S
        noise = sigmaS * np.sqrt(2.0 * gamma * dt) * rng2.standard_normal(size=N)
        S = (S + gamma * S_mean * dt + noise) / (1.0 + gamma * dt)
        # Deterministic dz
        dz = kT * (S - 1.0) * dt
        r2_sq = r2**2 + dz
        r2 = _apply_boundary(r2_sq, vol_init2, rng2)
        RD2[step] = np.std(r2) / np.mean(r2)

    return RD1, RD2, time_axis


# ============================================================
# Per-γ task  (submitted to ProcessPoolExecutor)
# ============================================================

def _process_one_gamma(args):
    """Worker entrypoint: run all seeds for one γ and return packed results."""
    gamma, seeds, steps = args
    n_seeds = len(seeds)
    RD1_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    RD2_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    t_axis = None

    for i, seed in enumerate(seeds):
        RD1, RD2, t = run_one_gamma(gamma, seed, steps=steps)
        RD1_seeds[i] = RD1
        RD2_seeds[i] = RD2
        t_axis = t

    return gamma, RD1_seeds, RD2_seeds, t_axis


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Gamma sweep (Fig. 2)')
    parser.add_argument('--outdir',  type=str, default='data/sweeps')
    parser.add_argument('--workers', type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    gamma_values = np.round(np.arange(8.1, 10.01, 0.1), 2)   # ← starts at 0.1
    os.makedirs(args.outdir, exist_ok=True)

    tasks = [(gamma, SEEDS, N_STEPS) for gamma in gamma_values]

    results = dict(gamma_values=gamma_values)

    print(f"Launching γ sweep  ({len(gamma_values)} values × {len(SEEDS)} seeds"
          f"  |  {args.workers} workers)")

    gathered = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one_gamma, task): task[0]
                   for task in tasks}

        with tqdm(total=len(gamma_values), desc='Gamma sweep') as pbar:
            for future in as_completed(futures):
                gamma, RD1_seeds, RD2_seeds, t_axis = future.result()
                gathered[gamma] = (RD1_seeds, RD2_seeds, t_axis)
                pbar.update(1)
                pbar.set_postfix(
                    gamma=f'{gamma:.2f}',
                    RD1=f'{RD1_seeds.mean(axis=0)[-1]:.5f}',
                    RD2=f'{RD2_seeds.mean(axis=0)[-1]:.5f}',
                )

    # Assemble in sorted gamma order
    for gamma in gamma_values:
        tag = f"g{gamma:.2f}".replace('.', 'p')
        RD1_seeds, RD2_seeds, t_axis = gathered[gamma]
        results[f"{tag}_RD1"]  = RD1_seeds.astype(np.float32)
        results[f"{tag}_RD2"]  = RD2_seeds.astype(np.float32)
        results[f"{tag}_time"] = t_axis.astype(np.float32)
        print(f"  γ={gamma:.2f}  "
              f"RD1_mean={RD1_seeds.mean(axis=0)[-1]:.5f}  "
              f"RD2_mean={RD2_seeds.mean(axis=0)[-1]:.5f}")

    out_path = os.path.join(args.outdir, "gamma_sweep_RD.npz")
    np.savez_compressed(out_path, **results)
    print(f"\nSaved → {out_path}  ({os.path.getsize(out_path) / (1024**2):.1f} MB)")


if __name__ == '__main__':
    main()