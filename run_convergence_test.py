"""
run_convergence_test.py — Replicate Reviewer 2's Figure 1 for the new boundary
condition (at volume ~ U(0, V0), followed by LWC scaling).

Purpose
-------
Demonstrate that the reported steady-state relative dispersion is independent
of BOTH the time step and the ensemble size, to within the seed variability,
which is what Request 2(a) of the review asks for.  Two panels, matching the
reviewer's figure:

    Panel (a)   eps vs Delta_t / tau      at N = 100,000
    Panel (b)   eps vs N                  at Delta_t = 5e-6 tau

Three series in each panel, as in the reviewer's figure:
    Model 1, gamma = 10.0             (its normalized dynamics is gamma-independent)
    Model 2, gamma = 0.1
    Model 2, gamma = 10.0

Each configuration is run with SEEDS_DEFAULT independent seeds, so the seed
scatter of the steady-state value can be quoted alongside the convergence.

Steady-state definition
-----------------------

eps_ss      stored here, averaged over the fixed window
            [AVG_START_TAU, TOTAL_TAU] = [7.5 tau, 15 tau].  Kept as a
            reference value only.  It is not what Figure S8 plots.


Storage
-------
eps is recorded at EVERY time step.  Run lengths therefore differ between
time steps (15,000 to 30,000,000 samples), so each run is stored as its own
array rather than in one rectangular block:

    eps_<i>       (n_steps_i,) float32   per-step relative dispersion
    taxis_<tag>   (n_steps,)   float32   t / tau, one per distinct Delta_t/tau

The time axis is exactly (n + 1) * dt_factor, so it is stored once per
distinct time step rather than once per run.

Expect roughly 3.9 GB raw and 2.6 GB compressed for the default grid with
5 seeds.  Almost all of it comes from the two finest time steps.

Cost
----
n_steps = 15 / dt_factor, so the finest time steps dominate.  At N = 100,000
one step is roughly 2.3 ms, hence:

    dt_factor = 1e-5   ->  1.5e6 steps   ~ 1 h
    dt_factor = 5e-6   ->  3.0e6 steps   ~ 2 h
    dt_factor = 1e-6   ->  1.5e7 steps   ~ 9.5 h
    dt_factor = 5e-7   ->  3.0e7 steps   ~ 19 h

Tasks are dispatched longest-first.  With 5 seeds there are 150 tasks, of
which the 15 heaviest set the wall time.  Given >= 30 workers those all start
immediately and the wall time is that of a single 5e-7 run, about 19 h.
Adding seeds therefore costs core-hours but not wall time.

Usage
-----
    python run_convergence_test.py
    python run_convergence_test.py --outdir data/convergence --workers 90
    python run_convergence_test.py --dt-factors 1e-3 1e-4 1e-5 5e-6
    python run_convergence_test.py --seeds 2 5 8 17 21

Output
------
    convergence_test.npz
        eps_<i>       per-run per-step relative dispersion (float32)
        taxis_<tag>   t / tau for each distinct dt_factor (float32)
        eps_ss        (n_runs,)   mean of eps over [7.5 tau, 15 tau]
        model_id      (n_runs,)   1 or 2
        gamma         (n_runs,)
        N_droplets    (n_runs,)
        dt_factor     (n_runs,)
        seed          (n_runs,)
        n_steps       (n_runs,)
        axis_name     (n_runs,)   name of the matching taxis_<tag> array
        in_panel_a    (n_runs,) bool
        in_panel_b    (n_runs,) bool
        sigmaS, seeds, total_tau, avg_start_tau
"""

import argparse
import os
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
from tqdm.auto import tqdm

from config import (
    N_c, rho_w, kT, ND_eff, UM3_TO_CM3,
    initial_L, z0,
    compute_S_mean, compute_DZ, compute_tau,
)

# ── Reviewer's settings ──────────────────────────────────────
SEEDS_DEFAULT = [2, 5, 8, 17, 21]
SIGMA_S       = 0.01
TOTAL_TAU     = 15.0        # run length in units of tau
AVG_START_TAU = 7.5         # eps averaged over [AVG_START_TAU, TOTAL_TAU]

# Panel (a): time-step refinement at fixed N
DT_FACTORS_DEFAULT = [1e-3, 1e-4, 1e-5, 5e-6, 1e-6, 5e-7]
N_PANEL_A = 100_000

# Panel (b): ensemble-size scan at fixed time step
N_VALUES_DEFAULT = [10_000, 20_000, 50_000, 100_000, 200_000]
DT_FACTOR_PANEL_B = 5e-6

# The three series, as (model_id, gamma).  Model 1's normalized dynamics is
# gamma-independent (drift/diffusion both scale as 1/gamma and tau as gamma,
# so the dynamics in units of z0 and tau carries no gamma), and its cost is
# gamma-independent too since n_steps = TOTAL_TAU / dt_factor.
SERIES = [(1, 10.0), (2, 0.1), (2, 10.0)]


def _dt_tag(dt_factor):
    """Filesystem-safe tag for one dt_factor, e.g. 5e-06 -> '5em06'."""
    return f"{dt_factor:.0e}".replace('-', 'm').replace('+', 'p').replace('.', 'p')


# ============================================================
# Boundary condition (same rule as the production scripts)
# ============================================================

def _apply_boundary(r2, initial_volume, rng):
    """
    Droplets with z = r^2 <= 0 have fully evaporated.  They are reborn with
    volume drawn uniformly on (0, V0), where V0 = <V> is the mean droplet
    volume fixed by the conserved liquid water content and droplet number
    concentration.  In the r^3 units used here, V0 = initial_volume / N.  The
    rule involves no time step and no ensemble size.  Liquid water is then
    restored by a global rescaling; droplet number is conserved exactly.
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
# One run: fixed (model, gamma, N, dt_factor, seed)
# ============================================================

def _simulate(model_id, gamma, N, dt_factor, seed, sigmaS=SIGMA_S,
              total_tau=TOTAL_TAU):
    """Return the per-step eps history (n_steps,) float32 for one run."""
    S_mean = compute_S_mean(sigmaS, gamma)
    DZ     = compute_DZ(sigmaS, gamma)
    tau    = compute_tau(sigmaS, gamma)
    dt     = dt_factor * tau

    n_steps = int(round(total_tau / dt_factor))

    v_depl   = kT * (S_mean - 1.0)
    sqrt_2DZ = np.sqrt(2.0 * DZ)
    sqrt_dt  = np.sqrt(dt)

    rng = np.random.default_rng(seed)
    r = np.full(N, np.sqrt(z0), dtype=np.float64)
    initial_volume = np.sum(r**3)
    if model_id == 2:
        S = np.full(N, S_mean, dtype=np.float64)

    # float32 keeps worker memory at 120 MB for the longest run
    eps_rec = np.empty(n_steps, dtype=np.float32)

    for step in range(n_steps):
        if model_id == 1:
            dW = rng.standard_normal(size=N) * sqrt_dt
            r2 = r**2 + v_depl * dt + sqrt_2DZ * dW
        else:
            noise = sigmaS * np.sqrt(2.0 * gamma * dt) * rng.standard_normal(size=N)
            S = (S + gamma * S_mean * dt + noise) / (1.0 + gamma * dt)
            r2 = r**2 + kT * (S - 1.0) * dt

        r = _apply_boundary(r2, initial_volume, rng)

        mean_r = r.mean()
        eps_rec[step] = r.std() / mean_r if mean_r > 0 else 0.0

    return eps_rec


def _run_one_task(args):
    """Worker entrypoint for one (model, gamma, N, dt_factor, seed) run."""
    model_id, gamma, N, dt_factor, seed = args
    eps_rec = _simulate(model_id, gamma, N, dt_factor, seed)

    # t/tau for step n is exactly (n + 1) * dt_factor
    n_steps    = eps_rec.size
    start_step = int(np.ceil(AVG_START_TAU / dt_factor)) - 1
    start_step = max(0, min(start_step, n_steps - 1))
    eps_ss = float(eps_rec[start_step:].mean())

    return (model_id, gamma, N, dt_factor, seed), eps_rec, eps_ss


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Convergence test in Delta_t and N (Reviewer 2, Fig. 1)")
    parser.add_argument('--outdir', type=str, default='data/convergence')
    parser.add_argument('--workers', type=int, default=None,
                        help='Default: one worker per task (full parallelism)')
    parser.add_argument('--dt-factors', type=float, nargs='+',
                        default=DT_FACTORS_DEFAULT,
                        help='Delta_t / tau values for panel (a)')
    parser.add_argument('--n-values', type=int, nargs='+',
                        default=N_VALUES_DEFAULT,
                        help='Ensemble sizes for panel (b)')
    parser.add_argument('--seeds', type=int, nargs='+',
                        default=SEEDS_DEFAULT,
                        help='Random seeds; every configuration is run with each')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Build the task set.  Panels overlap at (N = N_PANEL_A,
    # dt_factor = DT_FACTOR_PANEL_B); the dict collapses that duplicate.
    panel_a_keys, panel_b_keys = set(), set()
    tasks = {}

    for model_id, gamma in SERIES:
        for seed in args.seeds:
            for dtf in args.dt_factors:
                key = (model_id, gamma, N_PANEL_A, dtf, seed)
                tasks[key] = None
                panel_a_keys.add(key)
            for N in args.n_values:
                key = (model_id, gamma, N, DT_FACTOR_PANEL_B, seed)
                tasks[key] = None
                panel_b_keys.add(key)

    task_list = list(tasks.keys())
    steps = {k: int(round(TOTAL_TAU / k[3])) for k in task_list}
    work  = {k: steps[k] * k[2] for k in task_list}
    heaviest = max(task_list, key=lambda k: work[k])

    n_workers = args.workers if args.workers is not None else len(task_list)

    total_samples = sum(steps.values())
    print(f"\nConvergence test  (seeds {args.seeds}, sigma_S {SIGMA_S})")
    print(f"  tasks    : {len(task_list)}  "
          f"({len(panel_a_keys)} panel a + {len(panel_b_keys)} panel b, "
          f"overlap collapsed)")
    print(f"  workers  : {n_workers}")
    print(f"  heaviest : model {heaviest[0]}, gamma {heaviest[1]}, "
          f"N {heaviest[2]}, dt/tau {heaviest[3]:.0e}, seed {heaviest[4]}  "
          f"-> {steps[heaviest]:,} steps")
    print(f"  eps window: [{AVG_START_TAU} tau, {TOTAL_TAU} tau]")
    print(f"  per-step samples stored: {total_samples:,}  "
          f"(~{total_samples * 4 / 1e9:.2f} GB raw)\n")

    # Longest tasks first so they are not left straggling at the end.
    task_list.sort(key=lambda k: work[k], reverse=True)

    results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_run_one_task, t): t for t in task_list}
        with tqdm(total=len(task_list), desc='convergence') as pbar:
            for future in as_completed(futures):
                key, eps_rec, eps_ss = future.result()
                results[key] = (eps_rec, eps_ss)
                pbar.update(1)
                pbar.set_postfix(
                    m=key[0], g=f'{key[1]}', N=key[2],
                    dtf=f'{key[3]:.0e}', s=key[4], eps=f'{eps_ss:.4f}')

    # ── Pack ──────────────────────────────────────────────────
    ordered = sorted(results.keys())
    n_runs  = len(ordered)

    out = {}

    # one time axis per distinct dt_factor (t/tau = (n + 1) * dt_factor)
    for dtf in sorted(set(k[3] for k in ordered)):
        n = int(round(TOTAL_TAU / dtf))
        out[f"taxis_{_dt_tag(dtf)}"] = (
            (np.arange(n, dtype=np.float64) + 1.0) * dtf).astype(np.float32)

    eps_ss_arr = np.empty(n_runs, dtype=np.float64)
    model_id   = np.empty(n_runs, dtype=np.int32)
    gamma_arr  = np.empty(n_runs, dtype=np.float64)
    N_arr      = np.empty(n_runs, dtype=np.int64)
    dtf_arr    = np.empty(n_runs, dtype=np.float64)
    seed_arr   = np.empty(n_runs, dtype=np.int64)
    nstep_arr  = np.empty(n_runs, dtype=np.int64)
    axis_name  = np.empty(n_runs, dtype=object)
    in_a       = np.zeros(n_runs, dtype=bool)
    in_b       = np.zeros(n_runs, dtype=bool)

    for i, key in enumerate(ordered):
        eps_rec, eps_ss = results[key]
        out[f"eps_{i}"] = eps_rec
        eps_ss_arr[i] = eps_ss
        (model_id[i], gamma_arr[i], N_arr[i],
         dtf_arr[i], seed_arr[i]) = key
        nstep_arr[i] = eps_rec.size
        axis_name[i] = f"taxis_{_dt_tag(key[3])}"
        in_a[i] = key in panel_a_keys
        in_b[i] = key in panel_b_keys

    out.update(
        eps_ss=eps_ss_arr, model_id=model_id, gamma=gamma_arr,
        N_droplets=N_arr, dt_factor=dtf_arr, seed=seed_arr,
        n_steps=nstep_arr, axis_name=axis_name.astype(str),
        in_panel_a=in_a, in_panel_b=in_b,
        sigmaS=SIGMA_S, seeds=np.asarray(args.seeds),
        total_tau=TOTAL_TAU, avg_start_tau=AVG_START_TAU,
    )

    out_path = os.path.join(args.outdir, 'convergence_test.npz')
    np.savez_compressed(out_path, **out)

    # ── Console summary: mean over seeds, with seed scatter ──
    def _stats(model_id_s, gamma_s, N, dtf):
        vals = [results[(model_id_s, gamma_s, N, dtf, s)][1]
                for s in args.seeds]
        v = np.asarray(vals)
        return v.mean(), v.std(ddof=1) if v.size > 1 else 0.0

    print("\nPanel (a)   eps vs dt/tau   at N =", N_PANEL_A,
          "   (mean +/- seed s.d.)")
    for model_id_s, gamma_s in SERIES:
        label = "Model 1" if model_id_s == 1 else f"Model 2, gamma={gamma_s}"
        row = []
        for dtf in sorted(args.dt_factors, reverse=True):
            m, s = _stats(model_id_s, gamma_s, N_PANEL_A, dtf)
            row.append(f"{dtf:.0e}:{m:.4f}+-{s:.4f}")
        print(f"  {label:22s} " + "  ".join(row))

    print(f"\nPanel (b)   eps vs N   at dt/tau = {DT_FACTOR_PANEL_B:.0e}"
          "   (mean +/- seed s.d.)")
    for model_id_s, gamma_s in SERIES:
        label = "Model 1" if model_id_s == 1 else f"Model 2, gamma={gamma_s}"
        row = []
        for N in sorted(args.n_values):
            m, s = _stats(model_id_s, gamma_s, N, DT_FACTOR_PANEL_B)
            row.append(f"{N}:{m:.4f}+-{s:.4f}")
        print(f"  {label:22s} " + "  ".join(row))

    print(f"\nSaved → {out_path}  "
          f"({os.path.getsize(out_path) / (1024**3):.2f} GB)")


if __name__ == '__main__':
    main()
