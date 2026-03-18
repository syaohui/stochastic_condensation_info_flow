"""
plot_figures.py — Generate all figures for the paper from summary data.

Usage
-----
    python plot_figures.py --datadir data --outdir figures
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from config import (
    N_c, rho_w, kT, ND_eff, z0,
    compute_S_mean, compute_DZ, compute_tau, HIST_RMAX,
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
    path = os.path.join(datadir, f"gamma{gamma}", f"summary_gamma{gamma}.npz")
    data = np.load(path, allow_pickle=True)
    m1, m2 = {}, {}
    for key in data.files:
        if key.startswith('m1_'):
            m1[key[3:]] = data[key]
        elif key.startswith('m2_'):
            m2[key[3:]] = data[key]
    return m1, m2

# ── Figure S4: Heatmap, 4tau, Time [s] ──
def plot_figS4(datadir, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    for row, gamma in enumerate([0.3, 8.0]):
        m1, m2 = load_summary(datadir, gamma)
        for col, (mlabel, d) in enumerate([('Model 1', m1), ('Model 2', m2)]):
            ax = axes[row, col]
            hist = d['hist_data']; tau = float(d['tau']); ta = d['time_axis']
            s4 = np.searchsorted(ta, 4*tau)
            hp = hist[:, :s4]; mask = np.where(hp > 0, 1, np.nan)
            im = ax.imshow(hp*mask, cmap='jet', aspect='auto', origin='lower',
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
    for row, gamma in enumerate([0.3, 8.0]):
        m1, m2 = load_summary(datadir, gamma)
        for col, (mlabel, d) in enumerate([('Model 1', m1), ('Model 2', m2)]):
            ax = axes[row, col]
            hist = d['hist_data']; tau = float(d['tau']); tt = float(d['total_time'])
            mask = np.where(hist > 0, 1, np.nan)
            im = ax.imshow(hist*mask, cmap='jet', aspect='auto', origin='lower',
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
    for row, gamma in enumerate([0.3, 8.0]):
        m1, m2 = load_summary(datadir, gamma)

        t1 = m1['time_axis']; t2 = m2['time_axis']
        tau = float(m1['tau'])
        s1 = np.searchsorted(t1, 8*tau)
        s2 = np.searchsorted(t2, 8*tau)

        t1 = t1[:s1]; t2 = t2[:s2]

        ax_l = axes[row, 0]
        ax_l.set_xlabel('Time [s]', fontsize=16)
        ax_l.set_ylabel('Mean Radius [$\\mu$m]', fontsize=16, color='blue')
        ln1 = ax_l.plot(t1, m1['mean_r'][:s1], color='darkblue', lw=2, label='$\\bar{r}_1$', ls='-', alpha=0.9)
        ln2 = ax_l.plot(t2, m2['mean_r'][:s2], color='deepskyblue', lw=2, label='$\\bar{r}_2$', ls='--', dashes=(5,2), alpha=0.9)
        ax_l.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_l.set_ylim(6, 10); ax_l.grid(True, alpha=0.3)

        ax_r = ax_l.twinx()
        ax_r.set_ylabel('Standard Deviation of Radius [$\\mu$m]', fontsize=16, color='red')
        ln3 = ax_r.plot(t1, m1['std_r'][:s1], color='darkred', lw=2, label='$\\sigma_{r_1}$', ls='-', alpha=0.9)
        ln4 = ax_r.plot(t2, m2['std_r'][:s2], color='lightcoral', lw=2, label='$\\sigma_{r_2}$', ls='--', dashes=(5,2), alpha=0.9)
        ax_r.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r.set_ylim(0, 5)

        ax_l.text(0.98, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax_l.transAxes, fontsize=16,
                  va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        lines = ln1+ln2+ln3+ln4
        ax_l.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=16, framealpha=0.9)

        # Right panel
        ax_r2 = axes[row, 1]
        l1, = ax_r2.plot(t1, m1['eps'][:s1], color='darkblue', lw=2, label=r'$\varepsilon_1$', ls='-', alpha=0.9)
        l2, = ax_r2.plot(t2, m2['eps'][:s2], color='deepskyblue', lw=2, label=r'$\varepsilon_2$', ls='--', dashes=(5,2), alpha=0.9)

        ax_r2.set_xlabel('Time [s]', fontsize=16)
        ax_r2.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=16, color='b')
        ax_r2.set_ylim(0, 0.8)
        ax_r2.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r2.grid(True, which='both', linestyle=':', linewidth=0.8)

        ax_b = ax_r2.twinx()
        ax_b.set_ylim(1.0, 1.25)

        l3, = ax_b.plot(t1, m1['beta'][:s1], color='darkred', lw=2, label=r'$\beta_1$', ls='-', alpha=0.9)
        l4, = ax_b.plot(t2, m2['beta'][:s2], color='lightcoral', lw=2, label=r'$\beta_2$', ls='--', dashes=(5,2), alpha=0.9)

        ax_b.set_ylabel(r'Effective Radius Ratio $\beta$', fontsize=16, color='r')
        ax_b.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)

        ax_r2.text(0.98, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax_r2.transAxes, fontsize=16,
                   va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax_r2.legend([l1,l2,l3,l4], [x.get_label() for x in [l1,l2,l3,l4]], loc='lower right', fontsize=16, frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig1.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved fig1.png")

# ── Figure S1: Same as Fig 2 but Normalized Time ──
def plot_figS1(datadir, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    for row, gamma in enumerate([0.3, 8.0]):
        m1, m2 = load_summary(datadir, gamma)
        tau1 = float(m1['tau']); tau2 = float(m2['tau'])
        t1 = m1['time_axis']/tau1; t2 = m2['time_axis']/tau2
        ax_l = axes[row, 0]
        ax_l.set_xlabel('Normalized Time', fontsize=16)
        ax_l.set_ylabel('Mean Radius [$\\mu$m]', fontsize=16, color='blue')
        ln1 = ax_l.plot(t1, m1['mean_r'], color='darkblue', lw=2, label='$\\bar{r}_1$', ls='-', alpha=0.9)
        ln2 = ax_l.plot(t2, m2['mean_r'], color='deepskyblue', lw=2, label='$\\bar{r}_2$', ls='--', dashes=(5,2), alpha=0.9)
        ax_l.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_l.set_ylim(6, 10); ax_l.set_xlim(0, 15); ax_l.set_xticks([1,3,5,7,9,11,13,15])
        ax_l.grid(True, alpha=0.3)
        ax_r = ax_l.twinx()
        ax_r.set_ylabel('Standard Deviation of Radius [$\\mu$m]', fontsize=16, color='red')
        ln3 = ax_r.plot(t1, m1['std_r'], color='darkred', lw=2, label='$\\sigma_{r_1}$', ls='-', alpha=0.9)
        ln4 = ax_r.plot(t2, m2['std_r'], color='lightcoral', lw=2, label='$\\sigma_{r_2}$', ls='--', dashes=(5,2), alpha=0.9)
        ax_r.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r.set_ylim(0, 5); ax_r.set_xlim(0, 15); ax_r.set_xticks([1,3,5,7,9,11,13,15])
        ax_l.text(0.98, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax_l.transAxes, fontsize=16,
                  va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        lines = ln1+ln2+ln3+ln4; ax_l.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize=16, framealpha=0.9)
        ax_r2 = axes[row, 1]
        l1, = ax_r2.plot(t1, m1['eps'], color='darkblue', lw=2, label=r'$\varepsilon_1$', ls='-', alpha=0.9)
        l2, = ax_r2.plot(t2, m2['eps'], color='deepskyblue', lw=2, label=r'$\varepsilon_2$', ls='--', dashes=(5,2), alpha=0.9)
        ax_r2.set_xlabel('Normalized Time', fontsize=16)
        ax_r2.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=16, color='b')
        ax_r2.set_ylim(0, 0.8); ax_r2.set_xlim(0, 15); ax_r2.set_xticks([1,3,5,7,9,11,13,15])
        ax_r2.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r2.grid(True, which='both', linestyle=':', linewidth=0.8)
        ax_b = ax_r2.twinx()
        ax_b.set_ylim(1.0, 1.25); ax_b.set_xlim(0, 15); ax_b.set_xticks([1,3,5,7,9,11,13,15])
        l3, = ax_b.plot(t1, m1['beta'], color='darkred', lw=2, label=r'$\beta_1$', ls='-', alpha=0.9)
        l4, = ax_b.plot(t2, m2['beta'], color='lightcoral', lw=2, label=r'$\beta_2$', ls='--', dashes=(5,2), alpha=0.9)
        ax_b.set_ylabel(r'Effective Radius Ratio $\beta$', fontsize=16, color='r')
        ax_b.tick_params(axis='both', which='both', direction='in', labelsize=14, length=6, width=1.2)
        ax_r2.text(0.98, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax_r2.transAxes, fontsize=16,
                   va='top', ha='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax_r2.legend([l1,l2,l3,l4], [x.get_label() for x in [l1,l2,l3,l4]], loc='lower right', fontsize=16, frameon=True)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS1.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS1.png")

# ── Figure 2: Steady-state eps vs gamma ──
def plot_figure2(datadir, outdir):
    data = np.load(os.path.join(datadir, 'sweeps', 'gamma_sweep_RD.npz'), allow_pickle=True)
    gv = data['gamma_values']
    R1 = np.empty(len(gv)); R2 = np.empty(len(gv))
    for i, g in enumerate(gv):
        tag = f"g{g:.2f}".replace('.','p')
        s1 = find_steady_state_start(data[f"{tag}_RD1"]); s2 = find_steady_state_start(data[f"{tag}_RD2"])
        R1[i] = np.mean(data[f"{tag}_RD1"][s1:]); R2[i] = np.mean(data[f"{tag}_RD2"][s2:])
    plt.figure(figsize=(8, 5))
    plt.plot(gv, R1, 'b-', label='Model 1', linewidth=2.5)
    plt.plot(gv, R2, 'r-', label='Model 2', linewidth=2.5)
    plt.xlabel(r'Correlation Rate $\gamma$ [$s^{-1}$]', fontsize=18)
    plt.ylabel(r'Relative Dispersion $\varepsilon$', fontsize=18)
    plt.xlim(0.3, 8.0); plt.ylim(0.45, 0.54)
    plt.legend(fontsize=12); plt.grid(alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=16, direction='in')
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig2.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved fig2.png")

# ── Figure 3: Steady-state eps vs sigmaS ──
def plot_figure3(datadir, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, gamma in [(ax1, 0.3), (ax2, 8.0)]:
        data = np.load(os.path.join(datadir, 'sweeps', f'sigmaS_sweep_gamma{gamma}.npz'), allow_pickle=True)
        sv = data['sigmaS_values']
        R1 = np.empty(len(sv)); R2 = np.empty(len(sv))
        for i, s in enumerate(sv):
            tag = f"s{s:.3f}".replace('.','p')
            s1 = find_steady_state_start(data[f"{tag}_RD1"]); s2 = find_steady_state_start(data[f"{tag}_RD2"])
            R1[i] = np.mean(data[f"{tag}_RD1"][s1:]); R2[i] = np.mean(data[f"{tag}_RD2"][s2:])
        ax.plot(sv, R1, 'b-', label='Model 1', linewidth=2.5)
        ax.plot(sv, R2, 'r-', label='Model 2', linewidth=2.5)
        ax.set_xlabel(r'$\sigma_S$', fontsize=18)
        ax.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=18)
        ax.set_ylim(0.45, 0.54); ax.legend(fontsize=12); ax.grid(alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16, direction='in')
        ax.text(0.02, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax.transAxes, fontsize=16,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig3.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved fig3.png")

# ── Figure 4: Info flow vs time (left) + Info flow vs relative dispersion (right) ──
def plot_figure4(datadir, outdir, gamma=0.3):
    _, m2 = load_summary(datadir, gamma)
    ta = m2['time_axis']
    idx_sz, T_sz = estimate_T_S_to_z(m2['C_SS'], m2['C_zz'], m2['C_zS'], m2['C_S_dz'])
    idx_zs, T_zs = estimate_T_z_to_S(m2['C_SS'], m2['C_zz'], m2['C_zS'], m2['C_z_dS'], m2['C_S_dS'])
    T_ana = T_S_to_z_analytical(ta[idx_sz], gamma)
    rho = m2['rho_zS']
    eps = m2['eps']
    phi = 0.01
    LABEL_SIZE = 26; TICK_SIZE = 22; LEGEND_SIZE = 22

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(22, 9))

    # ── Left panel: info flow vs time ──
    l1, = ax_left.plot(ta[idx_sz], T_sz + phi, 'r-', label=r'$\hat T_{S\to z}+0.01$', linewidth=2.5)
    s1 = ax_left.scatter(ta[idx_zs], T_zs + phi, s=10.0, alpha=0.5, color='b', label=r'$\hat T_{z\to S}+0.01$')
    l3, = ax_left.plot(ta[idx_sz], T_ana + phi, 'k-', label=r'$T_{S\to z}+0.01$', linewidth=2.5)
    ax_left.set_xlabel('Time [s]', fontsize=LABEL_SIZE)
    ax_left.set_ylabel('Information Flow [Hz]', fontsize=LABEL_SIZE)
    ax_left.set_yscale('log'); ax_left.set_xlim(0, 200); ax_left.set_ylim(1e-3, 10)
    ax_left.tick_params(axis='both', which='both', direction='in', labelsize=TICK_SIZE, length=9, width=1.8)
    ax_left.grid(alpha=0.3)
    ax_left_twin = ax_left.twinx()
    l2, = ax_left_twin.plot(ta, rho + phi, color='cyan', label=r'$\rho_{zS}+0.01$', linewidth=2.5)
    ax_left_twin.set_ylabel('Correlation Coefficient', fontsize=LABEL_SIZE)
    ax_left_twin.set_yscale('log'); ax_left_twin.set_ylim(1e-3, 10)
    ax_left_twin.tick_params(axis='y', which='both', direction='in', labelsize=TICK_SIZE, length=9, width=1.8)
    handles = [l1, l3, s1, l2]
    ax_left.legend(handles, [h.get_label() for h in handles], fontsize=LEGEND_SIZE,
                   loc='upper right', frameon=True, framealpha=0.9, edgecolor='gray')

    # ── Right panel: info flow vs relative dispersion ──
    r1 = ax_right.scatter(eps[idx_sz], T_sz, color='r', s=2.0, alpha=0.8, label=r'$\hat{T}_{S\to z}$')
    ax_right.set_xlabel(r'Relative Dispersion $\varepsilon$', fontsize=LABEL_SIZE)
    ax_right.set_ylabel('Information Flow [Hz]', fontsize=LABEL_SIZE)
    ax_right.set_yscale('log'); ax_right.set_ylim(1e-4, 10)
    ax_right.tick_params(axis='both', which='both', direction='in', labelsize=TICK_SIZE, length=9, width=1.8)
    ax_right.grid(alpha=0.3)
    ax_right_twin = ax_right.twinx()
    r2 = ax_right_twin.scatter(eps, rho, color='g', s=2.0, alpha=0.8, label=r'$\rho_{zS}$')
    ax_right_twin.set_ylabel('Correlation Coefficient', fontsize=LABEL_SIZE)
    ax_right_twin.set_yscale('log'); ax_right_twin.set_ylim(1e-4, 10)
    ax_right_twin.tick_params(axis='y', which='both', direction='in', labelsize=TICK_SIZE, length=9, width=1.8)
    ax_right.legend([r1, r2], [r1.get_label(), r2.get_label()], fontsize=LEGEND_SIZE,
                    loc='best', frameon=True, framealpha=0.9, edgecolor='gray')

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig4.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved fig4.png")

# ── Figure S2: Steady-state distributions ──
def plot_figS2(datadir, outdir):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    for ax, gamma in [(ax1, 0.3), (ax2, 8.0)]:
        m1, m2 = load_summary(datadir, gamma)
        bins = m1['radius_bins']; centers = 0.5*(bins[:-1]+bins[1:])
        ax.plot(centers, m1['hist_data'][:,-1], label='$n_1(r)$', color='blue', lw=1.5)
        ax.plot(centers, m2['hist_data'][:,-1], label='$n_2(r)$', color='red', lw=1.5)
        ax.set_xlabel('Droplet Radius [$\\mu$m]', fontsize=16)
        ax.set_ylabel(r'$n(r)\ [\mathrm{cm}^{-3}\ \mu\mathrm{m}^{-1}]$', fontsize=16)
        ax.set_ylim(0, 12); ax.legend(fontsize=16, framealpha=0.9)
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
    for col, gamma in enumerate([0.3, 8.0]):
        data = np.load(os.path.join(datadir, 'sweeps', f'sigmaS_sweep_gamma{gamma}.npz'), allow_pickle=True)
        sv = data['sigmaS_values']
        s1t = np.empty(len(sv)); s2t = np.empty(len(sv))
        s1n = np.empty(len(sv)); s2n = np.empty(len(sv))
        for i, s in enumerate(sv):
            tag = f"s{s:.3f}".replace('.','p')
            tau = compute_tau(s, gamma); dt = 0.001*tau
            ss1 = find_steady_state_start(data[f"{tag}_RD1"]); ss2 = find_steady_state_start(data[f"{tag}_RD2"])
            s1t[i] = ss1*dt; s2t[i] = ss2*dt; s1n[i] = s1t[i]/tau; s2n[i] = s2t[i]/tau
        ax = axes[0, col]
        ax.semilogy(sv, s1t, 'b-o', label='Model 1', lw=2, ms=4)
        ax.semilogy(sv, s2t, 'r-o', label='Model 2', lw=2, ms=4)
        ax.set_xlabel(r'$\sigma_S$', fontsize=16); ax.set_ylabel('Steady-State Start Time [s]', fontsize=16)
        ax.grid(alpha=0.3, which='both')
        ax.legend(fontsize=12); ax.tick_params(axis='both', direction='in', labelsize=14)
        ax.text(0.02, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax.transAxes, fontsize=14,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        ax = axes[1, col]
        ax.semilogy(sv, s1n, 'b-o', label='Model 1', lw=2, ms=4)
        ax.semilogy(sv, s2n, 'r-o', label='Model 2', lw=2, ms=4)
        ax.set_xlabel(r'$\sigma_S$', fontsize=16); ax.set_ylabel('Steady-State Start Normalized Time', fontsize=16)
        ax.grid(alpha=0.3, which='both')
        ax.set_ylim(1,15)
        ax.legend(fontsize=12); ax.tick_params(axis='both', direction='in', labelsize=14)
        ax.text(0.02, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax.transAxes, fontsize=14,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS3.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS3.png")

# ── Figure S6: Analytical info flow with gamma limits ──
def plot_figS6(outdir, gamma=0.3):
    tau = compute_tau(0.01, gamma); t = np.linspace(0.1, 15*tau, 10000)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(t, T_S_to_z_analytical(t, gamma), 'r-', lw=2.5,
            label=rf'$T_{{S \to z}}(t)$, $\gamma={gamma}$ $s^{{-1}}$')
    ax.plot(t, T_S_to_z_limit_gamma_inf(t), 'c--', lw=2.5,
            label=r'$T_{S \to z}(t)=\frac{1}{2t}$, $\gamma \to \infty$')
    ax.plot(t, T_S_to_z_limit_gamma_zero(t), 'm--', lw=2.5,
            label=r'$T_{S \to z}(t)=\frac{3}{2t}$, $\gamma \to 0$')
    ax.set_yscale('log'); ax.set_ylim(1e-5, 10)
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