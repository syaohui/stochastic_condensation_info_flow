"""
run_sigmaS_sweep.py — Sweep σ_S from 0.001 to 0.02 (Fig. 4 of the paper).

Runs at two γ values (0.3 and 8.0) to produce both panels of Fig. 4.
Also records steady-state start times for Fig. S4.

Usage
-----
    python run_sigmaS_sweep.py
    python run_sigmaS_sweep.py --gamma 0.3
    python run_sigmaS_sweep.py --gamma 8.0

Output
------
    sigmaS_sweep_gamma<γ>.npz
"""

import argparse
import numpy as np
from tqdm.auto import tqdm
import os

from config import (
    N_c, rho_w, kT, ND_eff, UM3_TO_CM3,
    initial_L, z0, RANDOM_SEED,
    compute_S_mean, compute_DZ, compute_tau,
)


def _compute_r2_m1(chunk, v_depl, sqrt_2DZ, dt):
    dW = np.random.normal(0.0, 1.0, size=chunk.shape) * np.sqrt(dt)
    return chunk**2 + v_depl * dt + sqrt_2DZ * dW


def _apply_boundary(r2, initial_volume):
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
        r2[negative_mask] = (rmax * np.random.rand(num_neg)) ** (2.0/3.0)
        r = np.sqrt(r2)
        r *= np.cbrt(initial_volume / np.sum(r**3))
    return r


def run_one_sigmaS(gamma, sigmaS, N=100_000, steps=15_000):
    """Run Model 1 & Model 2 for one σ_S value, return RD histories + time axis."""
    r0 = np.sqrt(z0)
    S_mean = compute_S_mean(sigmaS, gamma)
    DZ     = compute_DZ(sigmaS, gamma)
    tau    = compute_tau(sigmaS, gamma)
    dt     = 0.001 * tau

    v_depl   = kT * (S_mean - 1.0)
    sqrt_2DZ = np.sqrt(2.0 * DZ)
    time_axis = np.arange(steps, dtype=np.float64) * dt

    # ── Model 1 ──
    np.random.seed(RANDOM_SEED)
    r1 = np.full(N, r0, dtype=np.float64)
    vol_init = np.sum(r1**3)
    RD1 = np.empty(steps, dtype=np.float64)

    for step in range(steps):
        r1_sq = _compute_r2_m1(r1, v_depl, sqrt_2DZ, dt)
        r1 = _apply_boundary(r1_sq, vol_init)
        RD1[step] = np.std(r1) / np.mean(r1)

    # ── Model 2 ──
    np.random.seed(RANDOM_SEED)
    r2 = np.full(N, r0, dtype=np.float64)
    S  = np.full(N, S_mean, dtype=np.float64)
    vol_init2 = np.sum(r2**3)
    RD2 = np.empty(steps, dtype=np.float64)

    for step in range(steps):
        noise = sigmaS * np.sqrt(2.0 * gamma * dt) * np.random.normal(size=N)
        S = (S + gamma * S_mean * dt + noise) / (1.0 + gamma * dt)
        dz = kT * (S - 1.0) * dt
        r2_sq = r2**2 + dz
        r2 = _apply_boundary(r2_sq, vol_init2)
        RD2[step] = np.std(r2) / np.mean(r2)

    return RD1, RD2, time_axis


def main():
    parser = argparse.ArgumentParser(description='σ_S sweep (Fig. 4)')
    parser.add_argument('--gamma', type=float, default=None,
                        help='Run only this γ (default: both 0.3 and 8.0)')
    parser.add_argument('--outdir', type=str, default='data/sweeps')
    args = parser.parse_args()

    gammas = [args.gamma] if args.gamma is not None else [0.3, 8.0]
    sigmaS_values = np.round(np.arange(0.001, 0.0201, 0.001), 4)
    os.makedirs(args.outdir, exist_ok=True)

    for gamma in gammas:
        print(f"\n{'='*60}")
        print(f"  σ_S sweep at γ = {gamma}")
        print(f"{'='*60}")

        results = dict(
            sigmaS_values=sigmaS_values,
            parameters=dict(gamma=gamma, N=100_000, kT=kT,
                            N_c=N_c, rho_w=rho_w, steps=15_000),
        )

        for sigmaS in tqdm(sigmaS_values, desc=f'γ={gamma}'):
            tag = f"s{sigmaS:.3f}".replace('.', 'p')
            RD1, RD2, t_axis = run_one_sigmaS(gamma, sigmaS)
            results[f"{tag}_RD1"]  = RD1
            results[f"{tag}_RD2"]  = RD2
            results[f"{tag}_time"] = t_axis
            print(f"  σ_S={sigmaS:.3f}  RD1_final={RD1[-1]:.5f}  "
                  f"RD2_final={RD2[-1]:.5f}")

        out_path = os.path.join(args.outdir, f"sigmaS_sweep_gamma{gamma}.npz")
        np.savez_compressed(out_path, **results)
        print(f"\nSaved → {out_path}  ({os.path.getsize(out_path)/(1024**2):.1f} MB)")


if __name__ == '__main__':
    main()
