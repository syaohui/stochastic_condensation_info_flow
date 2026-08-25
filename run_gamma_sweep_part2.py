"""
run_gamma_sweep_part2.py — Sweep γ from 5.1 to 10.0 (Fig. 2 of the paper, part 2).

For each γ, runs both Model 1 and Model 2 for N_STEPS steps and records
the mean-radius and std-radius histories.  Relative dispersion is NOT
stored; it is recovered as sigma_r / <r> at plot time.  The steady-state
values are computed in the plotting script.

Boundary condition
------------------
Droplets that evaporate (z = r^2 <= 0) are reborn with volume drawn
uniformly on (0, V0), where V0 = L / (N_c rho_w) is the mean droplet
volume fixed by the conserved liquid water content and number
concentration.  The rule contains no time step and no ensemble size, so
its content is independent of dt and N.  Total liquid water is restored
afterwards by a global rescaling, and droplet number is conserved
exactly.

Parallelism strategy
--------------------
Parallelized over γ values using ProcessPoolExecutor. Each (γ, seed)
pair uses np.random.default_rng(seed) — a local Generator object that
never touches the global numpy random state, so results are bit-for-bit
identical regardless of worker dispatch order.

Usage
-----
    python run_gamma_sweep_part2.py
    python run_gamma_sweep_part2.py --outdir data/sweeps
    python run_gamma_sweep_part2.py --workers 90

Output
------
    gamma_sweep_RD_part2.npz  —  gamma_values, and for each γ (tag e.g. "g0p10"):
        g<tag>_MR1, g<tag>_MR2   mean radius <r> [um]  (n_seeds, N_STEPS)
        g<tag>_SR1, g<tag>_SR2   std radius sigma_r    (n_seeds, N_STEPS)
        g<tag>_time              time axis [s]         (N_STEPS,)
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

# γ range for this part of the sweep.
GAMMA_MIN, GAMMA_MAX, GAMMA_STEP = 5.1, 10.0, 0.1
PART_TAG = "part2"


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
    N = r2.size
    negative_mask = r2 <= 0
    num_neg = int(np.sum(negative_mask))

    r = np.empty(N, dtype=np.float64)
    r[~negative_mask] = np.sqrt(r2[~negative_mask])

    if num_neg > 0:
        v_max = initial_volume / N          # = V0, dt- and N-free
        r[negative_mask] = np.cbrt(v_max * rng.random(num_neg))

    r *= np.cbrt(initial_volume / np.sum(r**3))
    return r


# ============================================================
# Per-(γ, seed) worker  — no global random state
# ============================================================

def run_one_gamma(gamma, seed, N=100_000, sigmaS=0.01, steps=N_STEPS):
    """Run Model 1 & Model 2 for one (γ, seed) pair.

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
# Per-γ task  (submitted to ProcessPoolExecutor)
# ============================================================

def _process_one_gamma(args):
    """Worker entrypoint: run all seeds for one γ and return packed results."""
    gamma, seeds, steps = args
    n_seeds = len(seeds)
    MR1_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    SR1_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    MR2_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    SR2_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    t_axis = None

    for i, seed in enumerate(seeds):
        MR1, SR1, MR2, SR2, t = run_one_gamma(gamma, seed, steps=steps)
        MR1_seeds[i] = MR1
        SR1_seeds[i] = SR1
        MR2_seeds[i] = MR2
        SR2_seeds[i] = SR2
        t_axis = t

    return gamma, MR1_seeds, SR1_seeds, MR2_seeds, SR2_seeds, t_axis


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=f'Gamma sweep {PART_TAG} '
                    f'(γ = {GAMMA_MIN} … {GAMMA_MAX})')
    parser.add_argument('--outdir',  type=str, default='data/sweeps')
    parser.add_argument('--workers', type=int,
                        default=max(1, multiprocessing.cpu_count() - 1),
                        help='Number of parallel worker processes')
    args = parser.parse_args()

    gamma_values = np.round(
        np.arange(GAMMA_MIN, GAMMA_MAX + GAMMA_STEP / 2.0, GAMMA_STEP), 2)
    os.makedirs(args.outdir, exist_ok=True)

    tasks = [(gamma, SEEDS, N_STEPS) for gamma in gamma_values]

    results = dict(gamma_values=gamma_values)

    print(f"Launching γ sweep {PART_TAG}  ({len(gamma_values)} values × "
          f"{len(SEEDS)} seeds  |  {args.workers} workers)")

    gathered = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one_gamma, task): task[0]
                   for task in tasks}

        with tqdm(total=len(gamma_values),
                  desc=f'Gamma sweep {PART_TAG}') as pbar:
            for future in as_completed(futures):
                gamma, MR1_seeds, SR1_seeds, MR2_seeds, SR2_seeds, t_axis = \
                    future.result()
                gathered[gamma] = (MR1_seeds, SR1_seeds,
                                   MR2_seeds, SR2_seeds, t_axis)
                pbar.update(1)
                pbar.set_postfix(
                    gamma=f'{gamma:.2f}',
                    r1=f'{MR1_seeds.mean(axis=0)[-1]:.4f}',
                    r2=f'{MR2_seeds.mean(axis=0)[-1]:.4f}',
                )

    # Assemble in sorted gamma order
    for gamma in gamma_values:
        tag = f"g{gamma:.2f}".replace('.', 'p')
        MR1_seeds, SR1_seeds, MR2_seeds, SR2_seeds, t_axis = gathered[gamma]
        results[f"{tag}_MR1"]  = MR1_seeds.astype(np.float32)
        results[f"{tag}_MR2"]  = MR2_seeds.astype(np.float32)
        results[f"{tag}_SR1"]  = SR1_seeds.astype(np.float32)
        results[f"{tag}_SR2"]  = SR2_seeds.astype(np.float32)
        results[f"{tag}_time"] = t_axis.astype(np.float32)
        print(f"  γ={gamma:.2f}  "
              f"<r>1={MR1_seeds.mean(axis=0)[-1]:.4f}  "
              f"σr1={SR1_seeds.mean(axis=0)[-1]:.4f}  "
              f"<r>2={MR2_seeds.mean(axis=0)[-1]:.4f}  "
              f"σr2={SR2_seeds.mean(axis=0)[-1]:.4f}")

    out_path = os.path.join(args.outdir, f"gamma_sweep_RD_{PART_TAG}.npz")
    np.savez_compressed(out_path, **results)
    print(f"\nSaved → {out_path}  ({os.path.getsize(out_path) / (1024**2):.1f} MB)")


if __name__ == '__main__':
    main()
