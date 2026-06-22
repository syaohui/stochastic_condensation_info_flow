"""
run_sigmaS_sweep.py — Sweep σ_S from 0.001 to 0.02 (Fig. 3 of the paper).

Runs at two γ values (0.1 and 10.0) to produce both panels of Fig. 3.
Also records steady-state start times for Fig. S4.

Parallelism strategy
--------------------
Parallelized over σ_S values using ProcessPoolExecutor. Each (σ_S, seed)
pair uses np.random.default_rng(seed) — a local Generator object that
never touches the global numpy random state, so results are bit-for-bit
identical regardless of worker dispatch order.

Usage
-----
    python run_sigmaS_sweep.py
    python run_sigmaS_sweep.py --gamma 0.1
    python run_sigmaS_sweep.py --gamma 10.0
    python run_sigmaS_sweep.py --workers 40

Output
------
    sigmaS_sweep_gamma<γ>.npz
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


def _compute_r2_m1(chunk, v_depl, sqrt_2DZ, dt, rng):
    dW = rng.standard_normal(size=chunk.shape) * np.sqrt(dt)
    return chunk**2 + v_depl * dt + sqrt_2DZ * dW


def _apply_boundary(r2, initial_volume, rng):
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
        r2[negative_mask] = (rmax * rng.random(num_neg)) ** (2.0/3.0)
        r = np.sqrt(r2)
        r *= np.cbrt(initial_volume / np.sum(r**3))
    return r, num_neg, lost / initial_volume if initial_volume > 0 else 0.0


def run_one_sigmaS(gamma, sigmaS, seed, N=100_000, steps=N_STEPS):
    """Run Model 1 & Model 2 for one σ_S value, return RD histories + diagnostics."""
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
    RD1            = np.empty(steps, dtype=np.float64)
    dL_over_L_m1   = np.zeros(steps, dtype=np.float64)
    n_neg_frac_m1  = np.zeros(steps, dtype=np.float64)

    for step in range(steps):
        r1_sq = _compute_r2_m1(r1, v_depl, sqrt_2DZ, dt, rng1)
        # diagnostics before BC
        neg_mask = r1_sq <= 0
        n_neg_frac_m1[step] = np.sum(neg_mask) / N
        pos_vol = np.sum(np.sqrt(r1_sq[~neg_mask])**3)
        dL_over_L_m1[step] = (vol_init - pos_vol) / vol_init
        r1, _, _ = _apply_boundary(r1_sq, vol_init, rng1)
        RD1[step] = np.std(r1) / np.mean(r1)

    # ── Model 2 ──
    rng2 = np.random.default_rng(seed)
    r2 = np.full(N, r0, dtype=np.float64)
    S  = np.full(N, S_mean, dtype=np.float64)
    vol_init2 = np.sum(r2**3)
    RD2            = np.empty(steps, dtype=np.float64)
    dL_over_L_m2   = np.zeros(steps, dtype=np.float64)
    n_neg_frac_m2  = np.zeros(steps, dtype=np.float64)

    for step in range(steps):
        noise = sigmaS * np.sqrt(2.0 * gamma * dt) * rng2.standard_normal(size=N)
        S = (S + gamma * S_mean * dt + noise) / (1.0 + gamma * dt)
        dz = kT * (S - 1.0) * dt
        r2_sq = r2**2 + dz
        # diagnostics before BC
        neg_mask = r2_sq <= 0
        n_neg_frac_m2[step] = np.sum(neg_mask) / N
        pos_vol = np.sum(np.sqrt(r2_sq[~neg_mask])**3)
        dL_over_L_m2[step] = (vol_init2 - pos_vol) / vol_init2
        r2, _, _ = _apply_boundary(r2_sq, vol_init2, rng2)
        RD2[step] = np.std(r2) / np.mean(r2)

    return (RD1, RD2, time_axis,
            dL_over_L_m1, n_neg_frac_m1,
            dL_over_L_m2, n_neg_frac_m2)


def _process_one_sigmaS(args):
    """Worker entrypoint: run all seeds for one σ_S and return packed results."""
    gamma, sigmaS, seeds, steps = args
    n_seeds = len(seeds)
    RD1_seeds          = np.empty((n_seeds, steps), dtype=np.float64)
    RD2_seeds          = np.empty((n_seeds, steps), dtype=np.float64)
    dL_over_L_m1_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    n_neg_frac_m1_seeds= np.empty((n_seeds, steps), dtype=np.float64)
    dL_over_L_m2_seeds = np.empty((n_seeds, steps), dtype=np.float64)
    n_neg_frac_m2_seeds= np.empty((n_seeds, steps), dtype=np.float64)
    t_axis = None

    for i, seed in enumerate(seeds):
        (RD1, RD2, t,
         dL1, nn1,
         dL2, nn2) = run_one_sigmaS(gamma, sigmaS, seed, steps=steps)
        RD1_seeds[i]           = RD1
        RD2_seeds[i]           = RD2
        dL_over_L_m1_seeds[i]  = dL1
        n_neg_frac_m1_seeds[i] = nn1
        dL_over_L_m2_seeds[i]  = dL2
        n_neg_frac_m2_seeds[i] = nn2
        t_axis = t

    return (sigmaS, RD1_seeds, RD2_seeds, t_axis,
            dL_over_L_m1_seeds, n_neg_frac_m1_seeds,
            dL_over_L_m2_seeds, n_neg_frac_m2_seeds)


def main():
    parser = argparse.ArgumentParser(description='σ_S sweep (Fig. 3)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Run only this γ (default: both 0.1 and 10.0)')
    parser.add_argument('--outdir', type=str, default='data/sweeps')
    parser.add_argument('--workers', type=int,
                        default=40,
                        help='Number of parallel worker processes (default: 40)')
    args = parser.parse_args()

    gammas = [args.gamma] if args.gamma is not None else [0.1, 10.0]
    sigmaS_values = np.round(np.arange(0.001, 0.0201, 0.001), 4)
    os.makedirs(args.outdir, exist_ok=True)

    # Run both gammas in parallel by submitting all tasks across both gammas
    # to the same pool — 40 workers shared across all (gamma, sigmaS) combos
    all_tasks = [(gamma, sigmaS, SEEDS, N_STEPS)
                 for gamma in gammas
                 for sigmaS in sigmaS_values]

    gathered = {}   # (gamma, sigmaS) -> results tuple

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_process_one_sigmaS, task): (task[0], task[1])
                   for task in all_tasks}

        with tqdm(total=len(all_tasks), desc='All tasks') as pbar:
            for future in as_completed(futures):
                gamma_key, sigmaS_key = futures[future]
                result = future.result()
                gathered[(gamma_key, sigmaS_key)] = result
                sigmaS_res = result[0]
                RD1_seeds  = result[1]
                RD2_seeds  = result[2]
                pbar.update(1)
                pbar.set_postfix(
                    gamma=f'{gamma_key}',
                    sigmaS=f'{sigmaS_res:.3f}',
                    RD1=f'{RD1_seeds.mean(axis=0)[-1]:.5f}',
                    RD2=f'{RD2_seeds.mean(axis=0)[-1]:.5f}',
                )

    # Save results per gamma
    for gamma in gammas:
        print(f"\n{'='*60}")
        print(f"  Saving σ_S sweep at γ = {gamma}")
        print(f"{'='*60}")

        results = dict(
            sigmaS_values=sigmaS_values,
            parameters=dict(gamma=gamma, N=100_000, kT=kT,
                            N_c=N_c, rho_w=rho_w, steps=N_STEPS),
        )

        for sigmaS in sigmaS_values:
            tag = f"s{sigmaS:.3f}".replace('.', 'p')
            (_, RD1_seeds, RD2_seeds, t_axis,
             dL1_seeds, nn1_seeds,
             dL2_seeds, nn2_seeds) = gathered[(gamma, sigmaS)]

            results[f"{tag}_RD1"]          = RD1_seeds.astype(np.float32)
            results[f"{tag}_RD2"]          = RD2_seeds.astype(np.float32)
            results[f"{tag}_time"]         = t_axis.astype(np.float32)
            results[f"{tag}_dL_over_L_m1"] = dL1_seeds.astype(np.float32)
            results[f"{tag}_n_neg_frac_m1"]= nn1_seeds.astype(np.float32)
            results[f"{tag}_dL_over_L_m2"] = dL2_seeds.astype(np.float32)
            results[f"{tag}_n_neg_frac_m2"]= nn2_seeds.astype(np.float32)

            print(f"  σ_S={sigmaS:.3f}  "
                  f"RD1_mean={RD1_seeds.mean(axis=0)[-1]:.5f}  "
                  f"RD2_mean={RD2_seeds.mean(axis=0)[-1]:.5f}  "
                  f"dL/L_m1={dL1_seeds.mean():.2e}  "
                  f"n_neg_m1={nn1_seeds.mean():.2e}  "
                  f"dL/L_m2={dL2_seeds.mean():.2e}  "
                  f"n_neg_m2={nn2_seeds.mean():.2e}")

        out_path = os.path.join(args.outdir, f"sigmaS_sweep_gamma{gamma}.npz")
        np.savez_compressed(out_path, **results)
        print(f"\nSaved → {out_path}  ({os.path.getsize(out_path)/(1024**2):.1f} MB)")


if __name__ == '__main__':
    main()
