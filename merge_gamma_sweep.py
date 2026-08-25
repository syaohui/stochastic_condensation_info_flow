"""
merge_gamma_sweep.py — Merge the part 1 and part 2 γ-sweep outputs.

Part 1 covers γ = 0.1 … 5.0, part 2 covers γ = 5.1 … 10.0, both with an
interval of 0.1.  This script concatenates their gamma_values arrays and
copies every per-γ data array across, producing a single file with the
same layout as the original unsplit sweep, so downstream plotting scripts
need no changes.

Usage
-----
    python merge_gamma_sweep.py
    python merge_gamma_sweep.py --indir data/sweeps --outdir data/sweeps

Output
------
    gamma_sweep_RD.npz  —  gamma_values (100 values, sorted), plus for each γ
        g<tag>_MR1, g<tag>_MR2, g<tag>_SR1, g<tag>_SR2, g<tag>_time
"""

import argparse
import os
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Merge γ sweep part 1 + part 2')
    parser.add_argument('--indir',  type=str, default='data/sweeps')
    parser.add_argument('--outdir', type=str, default='data/sweeps')
    parser.add_argument('--part1',  type=str, default='gamma_sweep_RD_part1.npz')
    parser.add_argument('--part2',  type=str, default='gamma_sweep_RD_part2.npz')
    parser.add_argument('--out',    type=str, default='gamma_sweep_RD.npz')
    args = parser.parse_args()

    p1 = os.path.join(args.indir, args.part1)
    p2 = os.path.join(args.indir, args.part2)
    for p in (p1, p2):
        if not os.path.exists(p):
            raise FileNotFoundError(f"missing input: {p}")

    d1 = np.load(p1)
    d2 = np.load(p2)

    g1 = d1['gamma_values']
    g2 = d2['gamma_values']

    overlap = np.intersect1d(g1, g2)
    if overlap.size:
        raise ValueError(f"γ values appear in both parts: {overlap}")

    gamma_values = np.round(np.sort(np.concatenate([g1, g2])), 2)

    merged = dict(gamma_values=gamma_values)
    for d in (d1, d2):
        for key in d.files:
            if key == 'gamma_values':
                continue
            if key in merged:
                raise ValueError(f"duplicate key across parts: {key}")
            merged[key] = d[key]

    # Sanity check: every γ must have its full set of arrays.
    suffixes = ('MR1', 'MR2', 'SR1', 'SR2', 'time')
    for gamma in gamma_values:
        tag = f"g{gamma:.2f}".replace('.', 'p')
        for suf in suffixes:
            key = f"{tag}_{suf}"
            if key not in merged:
                raise KeyError(f"missing array {key} after merge")

    os.makedirs(args.outdir, exist_ok=True)
    out_path = os.path.join(args.outdir, args.out)
    np.savez_compressed(out_path, **merged)

    print(f"part 1 : {g1.size} γ values  ({g1.min():.2f} … {g1.max():.2f})")
    print(f"part 2 : {g2.size} γ values  ({g2.min():.2f} … {g2.max():.2f})")
    print(f"merged : {gamma_values.size} γ values  "
          f"({gamma_values.min():.2f} … {gamma_values.max():.2f})")
    print(f"\nSaved → {out_path}  "
          f"({os.path.getsize(out_path) / (1024**2):.1f} MB)")


if __name__ == '__main__':
    main()
