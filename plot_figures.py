"""
plot_figures.py — Generate all figures for the paper from summary data.

Usage
-----
python plot_figures.py --datadir data --outdir figures
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import os
from glob import glob

from config import (
    N_c, rho_w, kT, ND_eff, z0,
    compute_S_mean, compute_DZ, compute_tau, HIST_RMAX,
    DT_FACTOR,
)
from information_flow import (
    estimate_T_S_to_z, estimate_T_z_to_S,
    T_S_to_z_analytical, T_S_to_z_limit_gamma_inf,
    T_S_to_z_limit_gamma_zero,
)

def find_steady_state_start(history, window=100, rel_tol=1e-3):
    cum_mean = np.cumsum(history) / (np.arange(len(history)) + 1)
    for i in range(window, len(history) - window):
        rel_change = abs(cum_mean[i + window] - cum_mean[i]) / (abs(cum_mean[i]) + 1e-12)
        if rel_change < rel_tol:
            return i
    return int(0.85 * len(history))

def load_summary(datadir, gamma):
    """Load all seed files for a given gamma and return element-wise mean."""
    pattern = os.path.join(datadir, f"gamma{gamma}", f"summary_gamma{gamma}_seed*.npz")
    files = sorted(glob(pattern))
    if not files:
        raise FileNotFoundError(f"No seed files found: {pattern}")
    m1, m2 = {}, {}
    for i, path in enumerate(files):
        data = np.load(path, allow_pickle=True)
        for key in data.files:
            if key.startswith('m1_'):
                k = key[3:]
                arr = data[key]
                if i == 0:
                    m1[k] = arr.copy().astype(np.float64) if arr.ndim >= 1 else arr
                elif arr.ndim >= 1:
                    m1[k] = m1[k] + arr.astype(np.float64)
            elif key.startswith('m2_'):
                k = key[3:]
                arr = data[key]
                if i == 0:
                    m2[k] = arr.copy().astype(np.float64) if arr.ndim >= 1 else arr
                elif arr.ndim >= 1:
                    m2[k] = m2[k] + arr.astype(np.float64)
    n = len(files)
    for k in m1:
        if isinstance(m1[k], np.ndarray) and m1[k].ndim >= 1:
            m1[k] /= n
    for k in m2:
        if isinstance(m2[k], np.ndarray) and m2[k].ndim >= 1:
            m2[k] /= n
    return m1, m2

# ── Figure S4: Heatmap, 4tau, Time [s] ──
def plot_figS4(datadir, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for row, gamma in enumerate([0.1, 10.0]):
        m1, m2 = load_summary(datadir, gamma)
        for col, (mlabel, d) in enumerate([('Model 1', m1), ('Model 2', m2)]):
            ax = axes[row, col]
            hist = d['hist_data']; tau = float(d['tau']); ta = d['time_axis']
            s4 = np.searchsorted(ta, 4*tau)
            hp = hist[:, :s4]; mask = np.where(hp > 0, 1, np.nan)
            im = ax.imshow(hp*mask, cmap='viridis', aspect='auto', origin='lower',
                           extent=[0, 4*tau, 0, HIST_RMAX], vmin=0, vmax=11)
            ax.text(0.02, 0.98, f'{mlabel} $\\gamma={gamma}$ $s^{{-1}}$',
                    transform=ax.transAxes, fontsize=12, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.tick_params(axis='both', which='both', direction='in',
                           labelsize=12, length=6, width=1.2, top=False, right=False)
            ax.set_xlabel('Time [s]', fontsize=16)
            ax.set_ylabel('Droplet Radius [$\\mu$m]', fontsize=16)
            ax.set_xlim(0, 4*tau)
            cbar = fig.colorbar(im, ax=ax, shrink=0.9, fraction=0.06, pad=0.1)
            cbar.ax.set_title('$n(r)$ [cm$^{-3}$ $\\mu$m$^{-1}$]', pad=10, fontsize=12)
            cbar.ax.title.set_ha('center')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS4.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS4.png")

# ── Figure S5: Heatmap, 15tau, Normalized Time ──
def plot_figS5(datadir, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for row, gamma in enumerate([0.1, 10.0]):
        m1, m2 = load_summary(datadir, gamma)
        for col, (mlabel, d) in enumerate([('Model 1', m1), ('Model 2', m2)]):
            ax = axes[row, col]
            hist = d['hist_data']; tau = float(d['tau']); tt = float(d['total_time'])
            mask = np.where(hist > 0, 1, np.nan)
            im = ax.imshow(hist*mask, cmap='viridis', aspect='auto', origin='lower',
                           extent=[0, tt/tau, 0, HIST_RMAX], vmin=0, vmax=11)
            ax.text(0.02, 0.98, f'{mlabel} $\\gamma={gamma}$ $s^{{-1}}$',
                    transform=ax.transAxes, fontsize=12, va='top', ha='left',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            ax.tick_params(axis='both', which='both', direction='in',
                           labelsize=12, length=6, width=1.2, top=False, right=False)
            ax.set_xlabel('Normalized Time', fontsize=16)
            ax.set_ylabel('Droplet Radius [$\\mu$m]', fontsize=16)
            ax.set_xlim(0, 15); ax.set_xticks([1, 3, 5, 7, 9, 11, 13, 15])
            cbar = fig.colorbar(im, ax=ax, shrink=0.9, fraction=0.06, pad=0.1)
            cbar.ax.set_title('$n(r)$ [cm$^{-3}$ $\\mu$m$^{-1}$]', pad=10, fontsize=12)
            cbar.ax.title.set_ha('center')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS5.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS5.png")

# ── Figure 1: Mean/Std + eps/beta, Time [s], merged vertically ──
def plot_figure1(datadir, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row, gamma in enumerate([0.1, 10.0]):
        dash_pattern = (5, 3) if gamma == 0.1 else (5, 8)

        pattern = os.path.join(datadir, f"gamma{gamma}", f"summary_gamma{gamma}_seed*.npz")
        seed_files = sorted(glob(pattern))

        m1_mean_r_list = []; m2_mean_r_list = []
        m1_std_r_list  = []; m2_std_r_list  = []
        m1_eps_list    = []; m2_eps_list    = []
        m1_beta_list   = []; m2_beta_list   = []
        t1_full = None; t2_full = None; tau_val = None

        for path in seed_files:
            d = np.load(path, allow_pickle=True)
            m1 = {k[3:]: d[k] for k in d.files if k.startswith('m1_')}
            m2 = {k[3:]: d[k] for k in d.files if k.startswith('m2_')}
            t1_full  = m1['time_axis']
            t2_full  = m2['time_axis']
            tau_val  = float(m1['tau'])
            m1_mean_r_list.append(m1['mean_r'])
            m2_mean_r_list.append(m2['mean_r'])
            m1_std_r_list.append(m1['std_r'])
            m2_std_r_list.append(m2['std_r'])
            m1_eps_list.append(m1['eps'])
            m2_eps_list.append(m2['eps'])
            m1_beta_list.append(m1['beta'])
            m2_beta_list.append(m2['beta'])

        tau = tau_val
        s1 = np.searchsorted(t1_full, 8*tau)
        s2 = np.searchsorted(t2_full, 8*tau)

        t1 = t1_full[:s1]; t2 = t2_full[:s2]

        def ms(lst, s):
            arr = np.array([x[:s] for x in lst])
            return arr.mean(axis=0), arr.std(axis=0)

        m1_meanr, m1_meanr_std = ms(m1_mean_r_list, s1)
        m2_meanr, m2_meanr_std = ms(m2_mean_r_list, s2)
        m1_stdr,  m1_stdr_std  = ms(m1_std_r_list,  s1)
        m2_stdr,  m2_stdr_std  = ms(m2_std_r_list,  s2)
        m1_eps,   m1_eps_std   = ms(m1_eps_list,    s1)
        m2_eps,   m2_eps_std   = ms(m2_eps_list,    s2)
        m1_beta,  m1_beta_std  = ms(m1_beta_list,   s1)
        m2_beta,  m2_beta_std  = ms(m2_beta_list,   s2)

        ax_l = axes[row, 0]
        ax_l.set_xlabel('Time [s]', fontsize=16)
        ax_l.set_ylabel('Mean Radius [$\\mu$m]', fontsize=16, color='blue')
        ln1 = ax_l.plot(t1, m1_meanr, color='blue', lw=2, label='$\\bar{r}_1$', ls='-', alpha=0.9)
        ln2 = ax_l.plot(t2, m2_meanr, color='blue', lw=2, label='$\\bar{r}_2$', ls='--', dashes=dash_pattern, alpha=0.9)
        ax_l.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_l.set_ylim(4, 10); ax_l.grid(True, alpha=0.3)

        ax_r = ax_l.twinx()
        ax_r.set_ylabel('Standard Deviation of Radius [$\\mu$m]', fontsize=16, color='red')
        ln3 = ax_r.plot(t1, m1_stdr, color='red', lw=2, label='$\\sigma_{r_1}$', ls='-', alpha=0.9)
        ln4 = ax_r.plot(t2, m2_stdr, color='red', lw=2, label='$\\sigma_{r_2}$', ls='--', dashes=dash_pattern, alpha=0.9)
        ax_r.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r.set_ylim(0, 8)

        ax_l.text(0.98, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax_l.transAxes, fontsize=16,
                  va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        lines = ln1+ln2+ln3+ln4
        ax_l.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=14, framealpha=0.9)

        # Right panel
        ax_r2 = axes[row, 1]
        l1, = ax_r2.plot(t1, m1_eps, color='blue', lw=2, label=r'$\varepsilon_1$', ls='-', alpha=0.9)
        l2, = ax_r2.plot(t2, m2_eps, color='blue', lw=2, label=r'$\varepsilon_2$', ls='--', dashes=dash_pattern, alpha=0.9)

        ax_r2.set_xlabel('Time [s]', fontsize=16)
        ax_r2.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=16, color='b')
        ax_r2.set_ylim(0, 1.0)
        ax_r2.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r2.grid(True, which='both', linestyle=':', linewidth=0.8)

        ax_b = ax_r2.twinx()
        ax_b.set_ylim(1.0, 1.5)

        l3, = ax_b.plot(t1, m1_beta, color='red', lw=2, label=r'$\beta_1$', ls='-', alpha=0.9)
        l4, = ax_b.plot(t2, m2_beta, color='red', lw=2, label=r'$\beta_2$', ls='--', dashes=dash_pattern, alpha=0.9)

        ax_b.set_ylabel(r'Effective Radius Ratio $\beta$', fontsize=16, color='r')
        ax_b.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)

        ax_r2.text(0.98, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax_r2.transAxes, fontsize=16,
                   va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_r2.legend([l1,l2,l3,l4], [x.get_label() for x in [l1,l2,l3,l4]], loc='upper left', fontsize=14, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig1.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved fig1.png")

# ── Figure S1: Same as Fig 1 but Normalized Time ──
def plot_figS1(datadir, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row, gamma in enumerate([0.1, 10.0]):
        dash_pattern = (5, 3) if gamma == 0.1 else (5, 8)

        pattern = os.path.join(datadir, f"gamma{gamma}", f"summary_gamma{gamma}_seed*.npz")
        seed_files = sorted(glob(pattern))

        m1_mean_r_list = []; m2_mean_r_list = []
        m1_std_r_list  = []; m2_std_r_list  = []
        m1_eps_list    = []; m2_eps_list    = []
        m1_beta_list   = []; m2_beta_list   = []
        t1 = None; t2 = None

        for path in seed_files:
            d = np.load(path, allow_pickle=True)
            m1 = {k[3:]: d[k] for k in d.files if k.startswith('m1_')}
            m2 = {k[3:]: d[k] for k in d.files if k.startswith('m2_')}
            tau1 = float(m1['tau']); tau2 = float(m2['tau'])
            t1 = m1['time_axis'] / tau1
            t2 = m2['time_axis'] / tau2
            m1_mean_r_list.append(m1['mean_r'])
            m2_mean_r_list.append(m2['mean_r'])
            m1_std_r_list.append(m1['std_r'])
            m2_std_r_list.append(m2['std_r'])
            m1_eps_list.append(m1['eps'])
            m2_eps_list.append(m2['eps'])
            m1_beta_list.append(m1['beta'])
            m2_beta_list.append(m2['beta'])

        def ms(lst):
            arr = np.array(lst)
            return arr.mean(axis=0), arr.std(axis=0)

        m1_meanr, m1_meanr_std = ms(m1_mean_r_list)
        m2_meanr, m2_meanr_std = ms(m2_mean_r_list)
        m1_stdr,  m1_stdr_std  = ms(m1_std_r_list)
        m2_stdr,  m2_stdr_std  = ms(m2_std_r_list)
        m1_eps,   m1_eps_std   = ms(m1_eps_list)
        m2_eps,   m2_eps_std   = ms(m2_eps_list)
        m1_beta,  m1_beta_std  = ms(m1_beta_list)
        m2_beta,  m2_beta_std  = ms(m2_beta_list)

        ax_l = axes[row, 0]
        ax_l.set_xlabel('Normalized Time', fontsize=16)
        ax_l.set_ylabel('Mean Radius [$\\mu$m]', fontsize=16, color='blue')
        ln1 = ax_l.plot(t1, m1_meanr, color='blue', lw=2, label='$\\bar{r}_1$', ls='-', alpha=0.9)
        ln2 = ax_l.plot(t2, m2_meanr, color='blue', lw=2, label='$\\bar{r}_2$', ls='--', dashes=dash_pattern, alpha=0.9)
        ax_l.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_l.set_ylim(4, 10); ax_l.set_xlim(0, 15); ax_l.set_xticks([1,3,5,7,9,11,13,15])
        ax_l.grid(True, alpha=0.3)

        ax_r = ax_l.twinx()
        ax_r.set_ylabel('Standard Deviation of Radius [$\\mu$m]', fontsize=16, color='red')
        ln3 = ax_r.plot(t1, m1_stdr, color='red', lw=2, label='$\\sigma_{r_1}$', ls='-', alpha=0.9)
        ln4 = ax_r.plot(t2, m2_stdr, color='red', lw=2, label='$\\sigma_{r_2}$', ls='--', dashes=dash_pattern, alpha=0.9)
        ax_r.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r.set_ylim(0, 8.0); ax_r.set_xlim(0, 15); ax_r.set_xticks([1,3,5,7,9,11,13,15])

        ax_l.text(0.98, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax_l.transAxes, fontsize=16,
                  va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        lines = ln1+ln2+ln3+ln4; ax_l.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=14, framealpha=0.9)

        ax_r2 = axes[row, 1]
        l1, = ax_r2.plot(t1, m1_eps, color='blue', lw=2, label=r'$\varepsilon_1$', ls='-', alpha=0.9)
        l2, = ax_r2.plot(t2, m2_eps, color='blue', lw=2, label=r'$\varepsilon_2$', ls='--', dashes=dash_pattern, alpha=0.9)
        ax_r2.set_xlabel('Normalized Time', fontsize=16)
        ax_r2.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=16, color='b')
        ax_r2.set_ylim(0, 1.0); ax_r2.set_xlim(0, 15); ax_r2.set_xticks([1,3,5,7,9,11,13,15])
        ax_r2.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r2.grid(True, which='both', linestyle=':', linewidth=0.8)

        ax_b = ax_r2.twinx()
        ax_b.set_ylim(1.0, 1.5); ax_b.set_xlim(0, 15); ax_b.set_xticks([1,3,5,7,9,11,13,15])
        l3, = ax_b.plot(t1, m1_beta, color='red', lw=2, label=r'$\beta_1$', ls='-', alpha=0.9)
        l4, = ax_b.plot(t2, m2_beta, color='red', lw=2, label=r'$\beta_2$', ls='--', dashes=dash_pattern, alpha=0.9)
        ax_b.set_ylabel(r'Effective Radius Ratio $\beta$', fontsize=16, color='r')
        ax_b.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)

        ax_r2.text(0.98, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax_r2.transAxes, fontsize=16,
                   va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_r2.legend([l1,l2,l3,l4], [x.get_label() for x in [l1,l2,l3,l4]], loc='upper left', fontsize=14, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS1.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS1.png")

# ── Figure 2: Steady-state eps vs gamma ──
def plot_figure2(datadir, outdir):
    data = np.load(os.path.join(datadir, 'sweeps', 'gamma_sweep_RD.npz'), allow_pickle=True)
    gv = data['gamma_values']
    sigmaS = float(data['sigmaS']) if 'sigmaS' in data else 0.01
    gamma_star = np.sqrt(2) * kT * sigmaS / z0   # γ* = √2·k·σ_S / z₀  (manuscript line 183)

    R1_mean = np.empty(len(gv)); R1_std = np.empty(len(gv))
    R2_mean = np.empty(len(gv)); R2_std = np.empty(len(gv))
    for i, g in enumerate(gv):
        tag = f"g{g:.2f}".replace('.','p')
        rd1 = data[f"{tag}_RD1"]   # (n_seeds, n_steps)
        rd2 = data[f"{tag}_RD2"]
        s1 = find_steady_state_start(rd1.mean(axis=0))
        s2 = find_steady_state_start(rd2.mean(axis=0))
        ss1 = rd1[:, s1:].mean(axis=1)
        ss2 = rd2[:, s2:].mean(axis=1)
        R1_mean[i] = ss1.mean(); R1_std[i] = ss1.std()
        R2_mean[i] = ss2.mean(); R2_std[i] = ss2.std()

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gv, R1_mean, 'b-', label='Model 1', linewidth=2.0)
    ax.fill_between(gv, R1_mean - 5*R1_std, R1_mean + 5*R1_std, color='b', alpha=0.2)
    ax.plot(gv, R2_mean, 'r-', label='Model 2', linewidth=2.0)
    ax.fill_between(gv, R2_mean - 5*R2_std, R2_mean + 5*R2_std, color='r', alpha=0.2)
    ax.set_xlabel(r'Correlation Rate $\gamma$ [$s^{-1}$]', fontsize=18)
    ax.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=18)
    ax.set_xlim(0.1, 10.0); ax.set_ylim(0.28, 0.62)
    ax.legend(fontsize=12); ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16, direction='in')

    # Upper x-axis: γ / γ*
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xlabel(r'$\gamma / \gamma^*$', fontsize=18)
    ax2.tick_params(axis='x', which='major', labelsize=14, direction='in')
    tick_pos = ax.get_xticks()
    tick_pos = tick_pos[(tick_pos >= 0.1) & (tick_pos <= 10.0)]
    ax2.set_xticks(tick_pos)
    ax2.set_xticklabels([f'{v/gamma_star:.0f}' for v in tick_pos])

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig2.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved fig2.png")

# ── Figure 3: Steady-state eps vs sigmaS ──
def plot_figure3(datadir, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, gamma in [(ax1, 0.1), (ax2, 10.0)]:
        data = np.load(os.path.join(datadir, 'sweeps', f'sigmaS_sweep_gamma{gamma}.npz'), allow_pickle=True)
        sv = data['sigmaS_values']
        R1_mean = np.empty(len(sv)); R1_std = np.empty(len(sv))
        R2_mean = np.empty(len(sv)); R2_std = np.empty(len(sv))
        for i, s in enumerate(sv):
            tag = f"s{s:.3f}".replace('.','p')
            rd1 = data[f"{tag}_RD1"]   # (n_seeds, n_steps)
            rd2 = data[f"{tag}_RD2"]
            s1 = find_steady_state_start(rd1.mean(axis=0))
            s2 = find_steady_state_start(rd2.mean(axis=0))
            ss1 = rd1[:, s1:].mean(axis=1)
            ss2 = rd2[:, s2:].mean(axis=1)
            R1_mean[i] = ss1.mean(); R1_std[i] = ss1.std()
            R2_mean[i] = ss2.mean(); R2_std[i] = ss2.std()
        ax.plot(sv, R1_mean, 'b-', label='Model 1', linewidth=2.0)
        ax.fill_between(sv, R1_mean - 5*R1_std, R1_mean + 5*R1_std, color='b', alpha=0.2)
        ax.plot(sv, R2_mean, 'r-', label='Model 2', linewidth=2.0)
        ax.fill_between(sv, R2_mean - 5*R2_std, R2_mean + 5*R2_std, color='r', alpha=0.2)
        ax.set_xlabel(r'$\sigma_S$', fontsize=18)
        ax.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=18)
        ax.set_ylim(0.28, 0.8); ax.legend(fontsize=12); ax.grid(alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16, direction='in')
        ax.text(0.02, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax.transAxes, fontsize=16,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig3.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved fig3.png")

# ── Figure 4: Info flow vs time (left) + Info flow vs relative dispersion (right) ──
def plot_figure4(datadir, outdir, gamma=0.1):
    pattern = os.path.join(datadir, f"gamma{gamma}", f"summary_gamma{gamma}_seed*.npz")
    seed_files = sorted(glob(pattern))
    T_sz_list, T_zs_list, rho_list, eps_list = [], [], [], []
    idx_sz_last, idx_zs_last = None, None
    ta_if = None
    M_STEP_val = None

    for path in seed_files:
        d = np.load(path, allow_pickle=True)
        m2 = {k[3:]: d[k] for k in d.files if k.startswith('m2_')}

        # Reconstruct IF time axis from M_STEP * dt (length n_if, not n_steps)
        M_STEP     = int(m2['M_STEP'])
        M_STEP_val = M_STEP
        dt         = float(m2['dt'])
        n_if       = len(m2['C_SS'])
        ta_if      = np.arange(n_if) * M_STEP * dt

        idx_sz, T_sz_s = estimate_T_S_to_z(
            m2['C_SS'], m2['C_zz'], m2['C_zS'], m2['C_S_dz'], m_step=1)
        idx_zs, T_zs_s = estimate_T_z_to_S(
            m2['C_SS'], m2['C_zz'], m2['C_zS'], m2['C_z_dS'], m2['C_S_dS'], m_step=1)

        T_sz_list.append(T_sz_s)
        T_zs_list.append(T_zs_s)
        rho_list.append(m2['rho_zS'])
        eps_list.append(m2['eps'])
        idx_sz_last = idx_sz
        idx_zs_last = idx_zs

    # Truncate all quantities to minimum length across seeds before averaging
    min_sz  = min(len(x) for x in T_sz_list)
    min_zs  = min(len(x) for x in T_zs_list)
    min_rho = min(len(x) for x in rho_list)
    min_eps = min(len(x) for x in eps_list)

    T_sz = np.mean([x[:min_sz]  for x in T_sz_list], axis=0)
    T_zs = np.mean([x[:min_zs]  for x in T_zs_list], axis=0)
    rho  = np.mean([x[:min_rho] for x in rho_list],  axis=0)
    eps  = np.mean([x[:min_eps] for x in eps_list],  axis=0)

    # Truncate indices to min lengths
    idx_sz = np.array(idx_sz_last[:min_sz])
    idx_zs = np.array(idx_zs_last[:min_zs])
    ta_sz  = ta_if[idx_sz]
    ta_zs  = ta_if[idx_zs]

    T_ana = T_S_to_z_analytical(ta_sz, gamma)

    LABEL_SIZE = 26; TICK_SIZE = 22; LEGEND_SIZE = 18
    LINTHRESH = 1e-3

    C_Tsz = 'red'
    C_Tzs = 'blue'
    C_ana = 'black'
    C_rho = 'cyan'

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(22, 9))
    fig.subplots_adjust(wspace=0.45)

    # ── Left panel: info flow vs time ──
    l1, = ax_left.plot(ta_sz, T_sz, '-', color=C_Tsz,
                       label=r'$\hat T_{S\to z}$', linewidth=2.5)
    s1 = ax_left.scatter(ta_zs, T_zs, s=20.0, alpha=0.6, color=C_Tzs,
                         label=r'$\hat T_{z\to S}$')
    l3, = ax_left.plot(ta_sz, T_ana, '-', color=C_ana,
                       label=r'$T_{S\to z}$', linewidth=2.5)
    ax_left.set_xlabel('Time [s]', fontsize=LABEL_SIZE)
    ax_left.set_ylabel('Information Flow [Hz]', fontsize=LABEL_SIZE)
    ax_left.set_yscale('symlog', linthresh=LINTHRESH)
    ax_left.set_xlim(0, 200); ax_left.set_ylim(-0.5, 1e3)
    ax_left.tick_params(axis='both', which='both', direction='in',
                        labelsize=TICK_SIZE, length=9, width=1.8)
    ax_left.grid(alpha=0.3)

    ax_left_twin = ax_left.twinx()
    l2, = ax_left_twin.plot(ta_if[:min_rho], rho, color=C_rho,
                            label=r'$\rho_{zS}$', linewidth=2.5)
    ax_left_twin.set_ylabel('Correlation Coefficient', fontsize=LABEL_SIZE)
    ax_left_twin.set_yscale('symlog', linthresh=LINTHRESH)
    ax_left_twin.set_ylim(-0.5, 1e3)
    ax_left_twin.tick_params(axis='y', which='both', direction='in',
                             labelsize=TICK_SIZE, length=9, width=1.8)

    handles = [l1, l3, s1, l2]
    ax_left.legend(handles, [h.get_label() for h in handles], fontsize=LEGEND_SIZE,
                   loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray',
                   markerscale=2.0, handlelength=2.0, borderpad=0.6)

    # ── Right panel: info flow vs relative dispersion ──
    # convert IF sample indices back to full step indices to index into eps
    idx_sz_full = idx_sz * M_STEP_val
    valid_mask  = idx_sz_full < min_eps
    eps_for_sz  = eps[idx_sz_full[valid_mask]]
    T_sz_plot   = T_sz[valid_mask]

    # rho x-axis: subsample eps at M_STEP intervals to match rho length (n_if)
    eps_for_rho = eps[np.arange(min_rho) * M_STEP_val]

    r1 = ax_right.scatter(eps_for_sz, T_sz_plot, color=C_Tsz, s=20.0, alpha=0.7,
                          label=r'$\hat{T}_{S\to z}$')
    ax_right.set_xlabel(r'Relative Dispersion $\varepsilon$', fontsize=LABEL_SIZE)
    ax_right.set_ylabel('Information Flow [Hz]', fontsize=LABEL_SIZE)
    ax_right.set_yscale('log'); ax_right.set_ylim(1e-4, 1e3)
    ax_right.tick_params(axis='both', which='both', direction='in',
                         labelsize=TICK_SIZE, length=9, width=1.8)
    ax_right.grid(alpha=0.3)

    ax_right_twin = ax_right.twinx()
    r2 = ax_right_twin.scatter(eps_for_rho, rho[:min_rho], color=C_rho, s=20.0, alpha=0.7,
                               label=r'$\rho_{zS}$')
    ax_right_twin.set_ylabel('Correlation Coefficient', fontsize=LABEL_SIZE)
    ax_right_twin.set_yscale('log'); ax_right_twin.set_ylim(1e-4, 1e3)
    ax_right_twin.tick_params(axis='y', which='both', direction='in',
                              labelsize=TICK_SIZE, length=9, width=1.8)

    ax_right.legend([r1, r2], [r1.get_label(), r2.get_label()], fontsize=LEGEND_SIZE+2,
                    loc='best', frameon=True, framealpha=0.9, edgecolor='gray',
                    markerscale=2.0, handlelength=2.0, borderpad=0.6)

    plt.savefig(os.path.join(outdir, 'fig4.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved fig4.png")

# ── Figure S2: Steady-state distributions ──
def plot_figS2(datadir, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for ax, gamma in [(ax1, 0.1), (ax2, 10.0)]:
        m1, m2 = load_summary(datadir, gamma)
        bins = m1['radius_bins']; centers = 0.5*(bins[:-1]+bins[1:])
        ax.plot(centers, m1['hist_data'][:,-1], label='$n_1(r)$', color='blue', lw=1.5)
        ax.plot(centers, m2['hist_data'][:,-1], label='$n_2(r)$', color='red', lw=1.5)
        ax.set_xlabel('Droplet Radius [$\\mu$m]', fontsize=16)
        ax.set_ylabel(r'$n(r)\ [\mathrm{cm}^{-3}\ \mu\mathrm{m}^{-1}]$', fontsize=16)
        ax.set_ylim(0, 16); ax.legend(fontsize=16, framealpha=0.9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', direction='in', labelsize=14, length=6, width=1.2)
        ax.text(0.02, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax.transAxes, fontsize=16,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS2.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS2.png")

# ── Figure S3: Steady-state start time vs sigmaS ──
def plot_figS3(datadir, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    for col, gamma in enumerate([0.1, 10.0]):
        data = np.load(os.path.join(datadir, 'sweeps', f'sigmaS_sweep_gamma{gamma}.npz'), allow_pickle=True)
        sv = data['sigmaS_values']
        s1t = np.empty(len(sv)); s2t = np.empty(len(sv))
        s1n = np.empty(len(sv)); s2n = np.empty(len(sv))
        for i, s in enumerate(sv):
            tag = f"s{s:.3f}".replace('.','p')
            tau = compute_tau(s, gamma); dt = DT_FACTOR*tau
            ss1 = find_steady_state_start(data[f"{tag}_RD1"].mean(axis=0))
            ss2 = find_steady_state_start(data[f"{tag}_RD2"].mean(axis=0))
            s1t[i] = ss1*dt; s2t[i] = ss2*dt; s1n[i] = s1t[i]/tau; s2n[i] = s2t[i]/tau
        ax = axes[0, col]
        ax.semilogy(sv, s1t, 'b-o', label='Model 1', lw=2, ms=4)
        ax.semilogy(sv, s2t, 'r-o', label='Model 2', lw=2, ms=4)
        ax.set_xlabel(r'$\sigma_S$', fontsize=16); ax.set_ylabel('Steady-State Time [s]', fontsize=16)
        ax.set_ylim(5,1e7)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))   
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=12); ax.tick_params(axis='both', direction='in', labelsize=14)
        ax.text(0.02, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax.transAxes, fontsize=14,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax = axes[1, col]
        ax.semilogy(sv, s1n, 'b-o', label='Model 1', lw=2, ms=4)
        ax.semilogy(sv, s2n, 'r-o', label='Model 2', lw=2, ms=4)
        ax.set_xlabel(r'$\sigma_S$', fontsize=16); ax.set_ylabel('Normalized Steady-State Time', fontsize=16)
        ax.grid(alpha=0.3, which='both')
        ax.set_ylim(1e-1,15)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5)) 
        ax.legend(fontsize=12); ax.tick_params(axis='both', direction='in', labelsize=14)
        ax.text(0.02, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax.transAxes, fontsize=14,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS3.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS3.png")


# ── Figure S6: Analytical info flow with gamma limits ──
def plot_figS6(outdir, gamma=0.1):
    tau = compute_tau(0.01, gamma); t = np.linspace(0.1, 15*tau, 10000)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, T_S_to_z_analytical(t, gamma), 'r-', lw=2.5,
            label=rf'$T_{{S \to z}}(t)$, $\gamma={gamma}$ $s^{{-1}}$')
    ax.plot(t, T_S_to_z_limit_gamma_inf(t), 'c--', lw=2.5,
            label=r'$T_{S \to z}(t)=\frac{1}{2t}$, $\gamma \to \infty$')
    ax.plot(t, T_S_to_z_limit_gamma_zero(t), 'm--', lw=2.5,
            label=r'$T_{S \to z}(t)=\frac{3}{2t}$, $\gamma \to 0$')
    ax.set_yscale('log'); ax.set_xscale('log'); ax.set_ylim(1e-5, 10)
    ax.set_xlabel('Time [s]', fontsize=20); ax.set_ylabel('Information Flow [Hz]', fontsize=20)
    ax.tick_params(axis='both', which='both', direction='in', labelsize=16, length=8, width=1.5)
    ax.grid(True, alpha=0.3); ax.legend(fontsize=16, loc='best', frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS6.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS6.png")




def main():
    parser = argparse.ArgumentParser(description='Generate all paper figures.')
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--outdir', type=str, default='figures')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("Generating figures...")
    plot_figure1(args.datadir, args.outdir)
    plot_figure2(args.datadir, args.outdir)
    plot_figure3(args.datadir, args.outdir)
    plot_figure4(args.datadir, args.outdir)
    plot_figS1(args.datadir, args.outdir)
    plot_figS2(args.datadir, args.outdir)
    plot_figS3(args.datadir, args.outdir)
    plot_figS4(args.datadir, args.outdir)
    plot_figS5(args.datadir, args.outdir)
    plot_figS6(args.outdir)
    print("\nAll figures generated.")

if __name__ == '__main__':
    main()