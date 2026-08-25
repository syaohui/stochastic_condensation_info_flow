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
from matplotlib.ticker import MaxNLocator

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

SS_DIAGNOSTIC = True    # print eps(t_ss) / eps(plateau) for each detection

def find_steady_state_start(history, dt_over_tau=DT_FACTOR, delta=0.0001,
                            grid=0.002, smooth=0.05, window=0.25,
                            fit_lo=0.15, fit_hi=0.95):
    """
    Steady-state onset from the local rate of change of the observable alone.

    If eps approaches a limit eps_inf with approach time theta, then
    d(eps)/dt = (eps_inf - eps) / theta, so the distance still to be covered is
    the current slope times theta.  Normalising by eps gives the fraction still
    ahead,

        R(t) = theta |d(eps)/dt| / eps(t),

    in which eps_inf cancels: no reference value, tail average or plateau
    estimate enters anywhere.  Steady state is where R falls below `delta`.

    Since the droplets start monodisperse, eps(0) = 0 and
    eps(t) = eps_inf (1 - exp(-t/theta)), for which R = exp(-t/theta) /
    (1 - exp(-t/theta)) = 1 / (exp(t/theta) - 1).  Setting R = delta gives the
    closed form

        t_ss = theta ln(1 + 1/delta),

    so delta = 0.01 places the onset where 1% of eps remains to be gained.
    Evaluating this rather than thresholding R pointwise keeps every estimate
    on the rise, where the signal is strong: near the plateau the true slope
    vanishes but the estimated slope does not, and |noise| has a positive mean,
    so a pointwise test carries a noise floor in R that grows as the ensemble
    shrinks and biases the onset late at small N.

    theta is obtained from the record.  Differentiating the same relation gives
    d/dt(d eps/dt) = -(1/theta)(d eps/dt), so the slope decays at rate 1/theta
    and ln|d eps/dt| is linear in t.  theta follows from a linear fit of that
    quantity over the rise, taken as the interval where |d eps/dt| lies between
    `fit_lo` and `fit_hi` of its maximum, i.e. where the decay is measurable
    above the noise.  The fit is weighted by |d eps/dt|, the inverse-variance
    weight for a logarithm.  The linearity of the fit is itself the test of the
    single-rate assumption: a curved ln|d eps/dt| indicates a slower second
    mode, in which case the onset is a lower bound.

    All widths and the tolerance are fixed in units of tau and nothing is
    defined in solver steps, so the detected physical time has a limit under
    refinement of the time step.

    `dt_over_tau` is the spacing of `history` in units of tau.  Resampling is a
    block average via reshape and the slope uses sliding sums, so the cost is
    linear in the number of recorded steps.

    With SS_DIAGNOSTIC set, the fraction of the plateau reached at the detected
    onset is printed.  The plateau there is the mean over the final half of the
    record; it is a diagnostic only and plays no part in the criterion.

    Returns the index into `history` at which steady state begins, or 0 if the
    record is too short or no relaxation is detected.
    """
    h = np.asarray(history, dtype=np.float64)
    n = h.size
    if n == 0 or not np.isfinite(dt_over_tau) or dt_over_tau <= 0.0:
        return 0

    # resample onto a grid fixed in tau: block average, O(n)
    k = max(1, int(round(grid / dt_over_tau)))
    ncell = n // k
    if ncell < 8:
        return 0
    g_dt = k * dt_over_tau                       # grid spacing actually used
    y = h[:ncell * k].reshape(ncell, k).mean(axis=1)
    t = (np.arange(ncell) + 0.5) * g_dt          # cell centres

    # centred moving average; a trailing average would displace the onset late
    # by half its width
    w = max(1, int(round(smooth / g_dt))) | 1
    if 1 < w < ncell:
        y = np.convolve(np.pad(y, w // 2, mode='edge'), np.ones(w) / w, 'valid')

    # slope by least squares on a centred window, which reduces to a
    # correlation of the signal with the centred ramp
    m = max(3, int(round(window / g_dt))) | 1
    if m >= ncell:
        return 0
    half = m // 2
    tw = (np.arange(m) - half) * g_dt
    slope = (np.correlate(np.pad(y, half, mode='edge'), tw, 'valid')
             / (tw ** 2).sum())

    # approach time from the decay of the slope, fitted over the rise
    a = np.abs(slope)
    amax = a.max()
    if not np.isfinite(amax) or amax <= 0.0:
        return 0
    i0 = int(np.argmax(a))
    seg = np.arange(i0, a.size)
    seg = seg[(a[seg] <= fit_hi * amax) & (a[seg] >= fit_lo * amax)]
    if seg.size < 10:
        return 0
    b = np.polyfit(t[seg], np.log(a[seg]), 1, w=a[seg])[0]
    if not np.isfinite(b) or b >= 0.0:
        return 0
    theta = -1.0 / b

    t_ss = theta * np.log(1.0 + 1.0 / delta)     # in units of tau
    idx = min(int(round(t_ss / dt_over_tau)), n - 1)

    if SS_DIAGNOSTIC:
        # fraction of the plateau reached at the detected onset.  Both values
        # come from the same smoothed grid, so the ratio is not affected by
        # the sampling noise of a single step.
        j = int(np.clip(round(t_ss / g_dt - 0.5), 0, y.size - 1))
        ref = float(y[y.size // 2:].mean())
        if np.isfinite(ref) and ref != 0.0:
            print(f"      [ss] theta={theta:.4f} tau  t_ss={t_ss:.3f} tau  "
                  f"eps(t_ss)/eps(plateau)={y[j] / ref:.5f}")

    return idx

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
        rd1 = data[f"{tag}_SR1"] / data[f"{tag}_MR1"]   # (n_seeds, n_steps)
        rd2 = data[f"{tag}_SR2"] / data[f"{tag}_MR2"]
        s1 = find_steady_state_start(rd1.mean(axis=0))
        s2 = find_steady_state_start(rd2.mean(axis=0))
        ss1 = rd1[:, s1:].mean(axis=1)
        ss2 = rd2[:, s2:].mean(axis=1)
        R1_mean[i] = ss1.mean(); R1_std[i] = ss1.std()
        R2_mean[i] = ss2.mean(); R2_std[i] = ss2.std()

    # Max across-seed standard deviation over the whole gamma range
    max_std = float(max(R1_std.max(), R2_std.max()))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(gv, R1_mean, 'b-', label='Model 1', linewidth=2.0)
    ax.plot(gv, R2_mean, 'r-', label='Model 2', linewidth=2.0)
    ax.set_xlabel(r'Correlation Rate $\gamma$ [$s^{-1}$]', fontsize=18)
    ax.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=18)
    ax.set_xlim(0.1, 10.0); ax.set_ylim(0.25, 0.55)

    # Drop the lowest y tick so it clears the leftmost x tick label
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5, prune='lower'))

    ax.legend(fontsize=12); ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=16, direction='in')
    ax.tick_params(axis='x', which='major', pad=6)
    ax.tick_params(axis='y', which='major', pad=6)

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
    plt.close()
    print(f"  Saved fig2.png   max across-seed std = {max_std:.2e}"
          f"  (M1 {R1_std.max():.2e}, M2 {R2_std.max():.2e})")
    return max_std


# ── Figure 3: Steady-state eps vs sigmaS ──
def plot_figure3(datadir, outdir):
    max_std = {}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, gamma in [(ax1, 0.1), (ax2, 10.0)]:
        data = np.load(os.path.join(datadir, 'sweeps', f'sigmaS_sweep_gamma{gamma}.npz'), allow_pickle=True)
        sv = data['sigmaS_values']
        R1_mean = np.empty(len(sv)); R1_std = np.empty(len(sv))
        R2_mean = np.empty(len(sv)); R2_std = np.empty(len(sv))
        for i, s in enumerate(sv):
            tag = f"s{s:.3f}".replace('.','p')
            rd1 = data[f"{tag}_SR1"] / data[f"{tag}_MR1"]   # (n_seeds, n_steps)
            rd2 = data[f"{tag}_SR2"] / data[f"{tag}_MR2"]
            s1 = find_steady_state_start(rd1.mean(axis=0))
            s2 = find_steady_state_start(rd2.mean(axis=0))
            ss1 = rd1[:, s1:].mean(axis=1)
            ss2 = rd2[:, s2:].mean(axis=1)
            R1_mean[i] = ss1.mean(); R1_std[i] = ss1.std()
            R2_mean[i] = ss2.mean(); R2_std[i] = ss2.std()

        # Max across-seed standard deviation over the whole sigmaS range
        max_std[gamma] = float(max(R1_std.max(), R2_std.max()))

        ax.plot(sv, R1_mean, 'b-', label='Model 1', linewidth=2.0)
        ax.plot(sv, R2_mean, 'r-', label='Model 2', linewidth=2.0)
        ax.set_xlabel(r'$\sigma_S$', fontsize=18)
        ax.set_ylabel(r'Relative Dispersion $\varepsilon$', fontsize=18)
        ax.set_ylim(0.25, 0.55); ax.legend(fontsize=12); ax.grid(alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=16, direction='in')
        ax.text(0.02, 0.98, f'$\\gamma={gamma}$ $s^{{-1}}$', transform=ax.transAxes, fontsize=16,
                va='top', ha='left', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'fig3.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved fig3.png   max across-seed std: "
          + ", ".join(f"gamma={g}: {v:.2e}" for g, v in max_std.items()))
    return max_std

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
            ss1 = find_steady_state_start((data[f"{tag}_SR1"] / data[f"{tag}_MR1"]).mean(axis=0))
            ss2 = find_steady_state_start((data[f"{tag}_SR2"] / data[f"{tag}_MR2"]).mean(axis=0))
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


# ── Figure S7: Boundary-condition diagnostics ──
# Values are the per-step quantities, on the same scale as the time series
# they replace.  Delta_L/L is signed and fluctuates about zero, so its mean
# absolute value is reported; n_neg/N is non-negative and is averaged directly.
_BC_MODELS = [('m1', 'Model 1', 'tab:blue'),
              ('m2', 'Model 2', 'tab:red')]
# (key stem, axis label, panel tag, take absolute value before averaging)
_BC_ROWS = [('dL_over_L',  r'$|\Delta L / L|$',     '(a)', True),
            ('n_neg_frac', r'$n_\mathrm{neg} / N$', '(b)', False)]
 
 
def plot_figS7(datadir, outdir):
    """Steady-state boundary diagnostics, from the run_simulation.py summaries.
 
    Each seed file is read individually so the spread across seeds can be
    shown.  The steady-state window is detected on the seed-averaged relative
    dispersion, so the same criterion is applied here as to the reported eps.
    """
    gammas = [0.1, 10.0]
    store = {}                       # (stem, model, gamma) -> (mean, std)
 
    for gamma in gammas:
        pattern = os.path.join(datadir, f"gamma{gamma}",
                               f"summary_gamma{gamma}_seed*.npz")
        files = sorted(glob(pattern))
        if not files:
            raise FileNotFoundError(f"No seed files found: {pattern}")
 
        eps_seeds = {suf: [] for suf, _, _ in _BC_MODELS}
        diag_seeds = {(suf, stem): []
                      for suf, _, _ in _BC_MODELS
                      for stem, _, _, _ in _BC_ROWS}
 
        for path in files:
            d = np.load(path, allow_pickle=True)
            for suf, _, _ in _BC_MODELS:
                eps_seeds[suf].append(np.asarray(d[f"{suf}_eps"], dtype=float))
                for stem, _, _, _ in _BC_ROWS:
                    key = f"{suf}_{stem}"
                    if key not in d.files:
                        raise KeyError(
                            f"{os.path.basename(path)} has no '{key}'. "
                            f"Rerun run_simulation.py with the boundary "
                            f"diagnostics enabled.")
                    diag_seeds[(suf, stem)].append(
                        np.asarray(d[key], dtype=float))
 
        for suf, _, _ in _BC_MODELS:
            start = find_steady_state_start(np.mean(eps_seeds[suf], axis=0))
            for stem, _, _, take_abs in _BC_ROWS:
                v = np.asarray(diag_seeds[(suf, stem)])      # (n_seeds, n_steps)
                if take_abs:
                    v = np.abs(v)
                per_seed = v[:, start:].mean(axis=1)
                mean = per_seed.mean()
                std = per_seed.std(ddof=1) if per_seed.size > 1 else 0.0
                store[(stem, suf, gamma)] = (mean, std)
                print(f"    {stem:12s} {suf}  gamma={gamma:<5g}  "
                      f"mean {mean:.6e}  s.d. across {per_seed.size} seeds "
                      f"{std:.6e}  ({100 * std / mean:.2f} %)")
 
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(gammas), dtype=float)
    width = 0.34
 
    for ax, (stem, ylabel, panel, _) in zip(axes, _BC_ROWS):
        # Common decimal scale for the panel, folded into the axis label, so
        # the values printed on the bars are plain numbers and matplotlib's
        # detached "1e-5" offset text above the axis is not used.
        peak = max(float(store[(stem, suf, g)][0])
                   for suf, _, _ in _BC_MODELS for g in gammas)
        exp10 = int(np.floor(np.log10(peak))) if peak > 0 else 0
        scale = 10.0 ** exp10
 
        vmax = 0.0
        for j, (suf, label, colour) in enumerate(_BC_MODELS):
            vals = np.array([store[(stem, suf, g)][0] for g in gammas]) / scale
            errs = np.array([store[(stem, suf, g)][1] for g in gammas]) / scale
            vmax = max(vmax, float((vals + errs).max()))
            pos = x + (j - 0.5) * width
            ax.bar(pos, vals, width,
                   color=colour, alpha=0.85, label=label,
                   edgecolor='black', linewidth=0.6)
            for xi, v in zip(pos, vals):
                ax.text(xi, v, f"{v:.2f}", ha='center', va='bottom',
                        fontsize=10)
 
        # linear axis: bar length is only meaningful measured from zero
        ax.set_ylim(0.0, vmax * 1.30)
        ax.ticklabel_format(axis='y', style='plain')
        ax.set_xticks(x)
        ax.set_xticklabels([rf'$\gamma = {g}$ $s^{{-1}}$' for g in gammas],
                           fontsize=14)
        ax.set_ylabel(rf'{ylabel}  ($\times 10^{{{exp10}}}$)', fontsize=16)
        ax.tick_params(axis='both', which='both', direction='in',
                       labelsize=13, right=True)
        ax.grid(alpha=0.3, axis='y')
 
    axes[0].legend(fontsize=12, loc='upper left', framealpha=0.9)
 
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS7.png'), dpi=300, bbox_inches='tight')
    plt.close(); print("  Saved figS7.png")

# ── Figure S8: convergence in time step and ensemble size ──
# Panel (a) sweeps Delta_t/tau at fixed N with the x axis
# reversed, so refinement runs left to right; panel (b) sweeps N at the
# manuscript time step.  Each point is the mean over seeds of the steady-state
# relative dispersion, with error bars giving the standard deviation across
# seeds.  Reads the archive written by run_convergence_test.py.
_CONV_SERIES = [
    # (model_id, gamma, label, colour)
    (1, 10.0,  r'Model 1, $\gamma = 10$',   'k'),
    (2, 0.1,  r'Model 2, $\gamma = 0.1$',  'tab:orange'),
    (2, 10.0, r'Model 2, $\gamma = 10$',   'tab:blue'),
]
CONV_DT_MARK = 5e-6
CONV_N_MARK = 100_000


def _conv_group(d, model_id, gamma, panel):
    """Group the convergence runs of one series by the swept variable.

    The steady-state value is obtained exactly as in Figures 2, 3 and S7: the
    onset index is detected on the seed-averaged eps history, each seed is
    then averaged in time from that index onward, and the plotted point is the
    mean of those per-seed values with the spread across seeds as its error.

    Returns (x, mean over seeds, std over seeds), sorted by x.
    """
    mask = (d['model_id'] == model_id) & np.isclose(d['gamma'], gamma)
    mask &= d['in_panel_a'] if panel == 'a' else d['in_panel_b']
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"no convergence runs for model {model_id}, "
                         f"gamma {gamma}, panel {panel}")

    xall = (d['dt_factor'][idx] if panel == 'a'
            else d['N_droplets'][idx].astype(float))

    xs = np.unique(xall)
    mean = np.empty(xs.size)
    std = np.empty(xs.size)
    for i, xv in enumerate(xs):
        rows = idx[xall == xv]
        # eps_<row> holds the per-step history of that run; seeds of one
        # configuration share a time step and therefore a length
        h = np.vstack([d[f"eps_{r}"] for r in rows])
        start = find_steady_state_start(h.mean(axis=0, dtype=np.float64),
                        dt_over_tau=float(d['dt_factor'][rows[0]]))
        per_seed = h[:, start:].mean(axis=1, dtype=np.float64)
        mean[i] = per_seed.mean()
        std[i] = per_seed.std()
    return xs, mean, std


def plot_figS8(datadir, outdir, ylim=None):
    """Steady-state eps versus time step and ensemble size."""
    path = os.path.join(datadir, 'convergence', 'convergence_test.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"convergence archive not found: {path}")
    d = np.load(path, allow_pickle=True)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13, 5.5), sharey=True,
        gridspec_kw={'width_ratios': [1.9, 1.0], 'wspace': 0.06})

    for ax, panel in ((ax_a, 'a'), (ax_b, 'b')):
        for model_id, gamma, label, colour in _CONV_SERIES:
            x, mean, std = _conv_group(d, model_id, gamma, panel)
            ax.errorbar(x, mean, yerr=std, fmt='o-', color=colour,
                        label=label, linewidth=2.0, markersize=6,
                        capsize=4, elinewidth=1.4)
        ax.set_xscale('log')
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=15,
                       direction='in', top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in',
                       top=True, right=True)

    # panel (a): time step, refinement to the right
    ax_a.invert_xaxis()
    ax_a.set_xlabel(r'Normalized Time Step $\Delta t / \tau$', fontsize=18)
    ax_a.set_ylabel(r'Relative Dispersion $\varepsilon$',
                    fontsize=18)
    ax_a.axvline(CONV_DT_MARK, color='0.45', linestyle=':', linewidth=1.5,
                 zorder=0)
    ax_a.text(CONV_DT_MARK, 0.02, r'$\Delta t = 5\times10^{-6}\,\tau$',
              transform=ax_a.get_xaxis_transform(), rotation=90,
              va='bottom', ha='right', fontsize=12, color='0.35')
    ax_a.text(0.98, 0.97, r'at $N = 10^{5}$', transform=ax_a.transAxes,
              fontsize=12, color='0.35', va='top', ha='right')
    ax_a.legend(fontsize=13, frameon=True, loc='best')

    # panel (b): ensemble size
    ax_b.set_xlabel(r'Ensemble Size $N$', fontsize=18)
    ax_b.axvline(CONV_N_MARK, color='0.45', linestyle=':', linewidth=1.5,
                 zorder=0)
    ax_b.text(CONV_N_MARK, 0.02, r'$N = 10^{5}$',
              transform=ax_b.get_xaxis_transform(), rotation=90,
              va='bottom', ha='right', fontsize=12, color='0.35')
    ax_b.text(0.97, 0.97, r'at $\Delta t = 5\times10^{-6}\,\tau$',
              transform=ax_b.transAxes, fontsize=12, color='0.35',
              va='top', ha='right')

    if ylim is not None:
        ax_a.set_ylim(*ylim)
    else:
        lo = float((d['eps_ss']).min())
        hi = float((d['eps_ss']).max())
        pad = max(0.02, 0.35 * (hi - lo))
        ax_a.set_ylim(lo - pad, hi + pad)

    plt.savefig(os.path.join(outdir, 'figS8.png'), dpi=300,
                bbox_inches='tight')
    plt.close(); print("  Saved figS8.png")

    # console table: mean +/- seed s.d., and the spread over the refined end
    seeds = np.unique(d['seed'])
    print(f"    seeds {list(seeds)}, sigma_S {float(d['sigmaS'])}, "
          f"eps averaged from the detected steady-state onset to "
          f"{float(d['total_tau']):g} tau")
    for panel, title, fmt in (
            ('a', 'dt/tau at N = 100,000', '{:.0e}'),
            ('b', 'N at dt/tau = 5e-06', '{:.0f}')):
        print(f"    -- {title} --")
        for model_id, gamma, label, _ in _CONV_SERIES:
            x, mean, std = _conv_group(d, model_id, gamma, panel)
            if panel == 'a':
                x, mean, std = x[::-1], mean[::-1], std[::-1]
            cells = "  ".join(f"{fmt.format(xi)}:{m:.4f}+-{sd:.4f}"
                              for xi, m, sd in zip(x, mean, std))
            plain = label.replace('$', '').replace('\\gamma', 'gamma')
            spread = mean[-3:].max() - mean[-3:].min()
            print(f"       {plain:22s} {cells}")
            print(f"       {'':22s} spread over last 3: {spread:.4f} "
                  f"({100 * spread / mean[-3:].mean():.2f} %), "
                  f"mean seed s.d. {std.mean():.4f}")

# ── Figure S9: steady-state time versus time step and ensemble size ──
# The detector returns an index; the corresponding physical time is
# index * dt, so the normalised time is simply index * (dt / tau) =
# index * dt_factor. Unlike the other figures the onset is detected separately for
# each seed, since the detected time is itself the plotted quantity and its
# spread across seeds is the error bar.


def _conv_group_time(d, model_id, gamma, panel):
    """Normalised steady-state time of one series, grouped by the swept axis.

    Returns (x, mean over seeds, std over seeds) of t_steady / tau.
    """
    mask = (d['model_id'] == model_id) & np.isclose(d['gamma'], gamma)
    mask &= d['in_panel_a'] if panel == 'a' else d['in_panel_b']
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        raise ValueError(f"no convergence runs for model {model_id}, "
                         f"gamma {gamma}, panel {panel}")

    xall = (d['dt_factor'][idx] if panel == 'a'
            else d['N_droplets'][idx].astype(float))

    xs = np.unique(xall)
    mean = np.empty(xs.size)
    std = np.empty(xs.size)
    for i, xv in enumerate(xs):
        rows = idx[xall == xv]
        t_norm = np.empty(rows.size)
        for j, r in enumerate(rows):
            start = find_steady_state_start(
            np.asarray(d[f"eps_{r}"], dtype=np.float64),
            dt_over_tau=float(d['dt_factor'][r]))
            # t / tau = index * dt / tau = index * dt_factor
            t_norm[j] = start * float(d['dt_factor'][r])
        mean[i] = t_norm.mean()
        std[i] = t_norm.std()
    return xs, mean, std


def plot_figS9(datadir, outdir, ylim=(None, 10.0)):
    """Normalised steady-state time versus time step and ensemble size."""
    path = os.path.join(datadir, 'convergence', 'convergence_test.npz')
    if not os.path.exists(path):
        raise FileNotFoundError(f"convergence archive not found: {path}")
    d = np.load(path, allow_pickle=True)

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(13, 5.5), sharey=True,
        gridspec_kw={'width_ratios': [1.9, 1.0], 'wspace': 0.06})

    for ax, panel in ((ax_a, 'a'), (ax_b, 'b')):
        for model_id, gamma, label, colour in _CONV_SERIES:
            x, mean, std = _conv_group_time(d, model_id, gamma, panel)
            ax.errorbar(x, mean, yerr=std, fmt='o-', color=colour,
                        label=label, linewidth=2.0, markersize=6,
                        capsize=4, elinewidth=1.4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.grid(alpha=0.3, which='both')
        ax.tick_params(axis='both', which='major', labelsize=15,
                       direction='in', top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in',
                       top=True, right=True)

    ax_a.invert_xaxis()
    ax_a.set_xlabel(r'Normalized Time Step $\Delta t / \tau$', fontsize=18)
    ax_a.set_ylabel('Normalized Steady-State Time', fontsize=18)
    ax_a.axvline(CONV_DT_MARK, color='0.45', linestyle=':', linewidth=1.5,
                 zorder=0)
    ax_a.text(CONV_DT_MARK, 0.02, r'$\Delta t = 5\times10^{-6}\,\tau$',
              transform=ax_a.get_xaxis_transform(), rotation=90,
              va='bottom', ha='right', fontsize=12, color='0.35')
    ax_a.text(0.98, 0.97, r'at $N = 10^{5}$', transform=ax_a.transAxes,
              fontsize=12, color='0.35', va='top', ha='right')
    ax_a.legend(fontsize=13, frameon=True, loc='best')

    ax_b.set_xlabel(r'Ensemble Size $N$', fontsize=18)
    ax_b.axvline(CONV_N_MARK, color='0.45', linestyle=':', linewidth=1.5,
                 zorder=0)
    ax_b.text(CONV_N_MARK, 0.02, r'$N = 10^{5}$',
              transform=ax_b.get_xaxis_transform(), rotation=90,
              va='bottom', ha='right', fontsize=12, color='0.35')
    ax_b.text(0.97, 0.97, r'at $\Delta t = 5\times10^{-6}\,\tau$',
              transform=ax_b.transAxes, fontsize=12, color='0.35',
              va='top', ha='right')

    if ylim is not None:
        ax_a.set_ylim(*ylim)

    plt.savefig(os.path.join(outdir, 'figS9.png'), dpi=300,
                bbox_inches='tight')
    plt.close(); print("  Saved figS9.png")

    seeds = np.unique(d['seed'])
    print(f"    seeds {list(seeds)}, sigma_S {float(d['sigmaS'])}, "
          f"t_steady / tau, onset detected per seed")
    for panel, title, fmt in (
            ('a', 'dt/tau at N = 100,000', '{:.0e}'),
            ('b', 'N at dt/tau = 5e-06', '{:.0f}')):
        print(f"    -- {title} --")
        for model_id, gamma, label, _ in _CONV_SERIES:
            x, mean, std = _conv_group_time(d, model_id, gamma, panel)
            if panel == 'a':
                x, mean, std = x[::-1], mean[::-1], std[::-1]
            cells = "  ".join(f"{fmt.format(xi)}:{m:.3f}+-{sd:.3f}"
                              for xi, m, sd in zip(x, mean, std))
            plain = label.replace('$', '').replace('\\gamma', 'gamma')
            spread = mean.max() - mean.min()
            print(f"       {plain:22s} {cells}")
            print(f"       {'':22s} full-range spread: {spread:.3f} "
                  f"({100 * spread / mean.mean():.1f} %), "
                  f"mean seed s.d. {std.mean():.3f}")

# ── Figure S10: collapse of the steady-state dispersion against sigma_S / gamma ──
# Pools every steady-state point of Figures 2 and 3 and plots it against the
# single combination sigma_S / gamma.  Same archives, same detector, same
# per-seed averaging as those figures, so each point here is a point there.
# (a) eps for both models; (b) the model difference eps_2 - eps_1.
_S10_M1, _S10_M2 = '#0072B2', '#D55E00'          # Model 1, Model 2
_S10_DIFF = '#6A3D9A'                            # panel (b), single accent
_S10_SOURCES = [
    # (marker, size, label) — marker denotes the data source in BOTH panels
    ('o', 8.5, r'$\gamma$ sweep ($\sigma_S = 0.01$)'),
    ('s', 8.5, r'$\sigma_S$ sweep ($\gamma = 0.1$)'),
    ('v', 8.5, r'$\sigma_S$ sweep ($\gamma = 10$)'),
]

def plot_figS10(datadir, outdir):
    """Collapse of the steady-state relative dispersion against sigma_S/gamma."""
    series = []          # (ratio, e1, e1_sd, e2, e2_sd, d, d_sd) per source

    for src in range(3):
        if src == 0:
            data = np.load(os.path.join(datadir, 'sweeps', 'gamma_sweep_RD.npz'),
                           allow_pickle=True)
            sigmaS = float(data['sigmaS']) if 'sigmaS' in data else 0.01
            tags = [(f"g{g:.2f}".replace('.', 'p'), sigmaS / float(g))
                    for g in data['gamma_values']]
        else:
            gamma = 0.1 if src == 1 else 10.0
            data = np.load(os.path.join(datadir, 'sweeps',
                                        f'sigmaS_sweep_gamma{gamma}.npz'),
                           allow_pickle=True)
            tags = [(f"s{s:.3f}".replace('.', 'p'), float(s) / gamma)
                    for s in data['sigmaS_values']]

        n = len(tags)
        ratio = np.empty(n)
        e1 = np.empty(n); e1_sd = np.empty(n)
        e2 = np.empty(n); e2_sd = np.empty(n)
        dd = np.empty(n); dd_sd = np.empty(n)
        for i, (tag, r) in enumerate(tags):
            # identical to Figures 2 and 3: onset on the seed-averaged history,
            # then a per-seed time average from that index onward
            rd1 = data[f"{tag}_SR1"] / data[f"{tag}_MR1"]   # (n_seeds, n_steps)
            rd2 = data[f"{tag}_SR2"] / data[f"{tag}_MR2"]
            s1 = find_steady_state_start(rd1.mean(axis=0))
            s2 = find_steady_state_start(rd2.mean(axis=0))
            ss1 = rd1[:, s1:].mean(axis=1)
            ss2 = rd2[:, s2:].mean(axis=1)
            delta = ss2 - ss1                    # paired by seed
            ratio[i] = r
            e1[i] = ss1.mean(); e1_sd[i] = ss1.std()
            e2[i] = ss2.mean(); e2_sd[i] = ss2.std()
            dd[i] = delta.mean(); dd_sd[i] = delta.std()

        o = np.argsort(ratio)
        series.append((ratio[o], e1[o], e1_sd[o], e2[o], e2_sd[o],
                       dd[o], dd_sd[o]))

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(14, 5.5))

    def _pts(ax, x, y, yerr, colour, marker, ms):
        # open markers: overlapping outlines still read as separate points,
        # whereas filled ones merge into a band where the sweeps are dense
        ax.errorbar(x, y, yerr=yerr, fmt=marker, ms=ms, linestyle='none',
                    color=colour, markerfacecolor='none',
                    markeredgecolor=colour, markeredgewidth=1.3,
                    ecolor=mcolors.to_rgba(colour, 0.5), elinewidth=0.9,
                    capsize=2.0, alpha=0.85, zorder=3)

    # panel (a): colour = model, marker = data source.  Drawn in reverse so
    # the dense sigma_S sweeps sit beneath the sparser gamma sweep.
    for (marker, ms, _), s in zip(reversed(_S10_SOURCES), reversed(series)):
        _pts(ax_a, s[0], s[1], s[2], _S10_M1, marker, ms)
        _pts(ax_a, s[0], s[3], s[4], _S10_M2, marker, ms)

    # panel (b): one quantity, so one colour; marker still = data source
    for (marker, ms, _), s in zip(reversed(_S10_SOURCES), reversed(series)):
        _pts(ax_b, s[0], s[5], s[6], _S10_DIFF, marker, ms)
    ax_b.axhline(0.0, color='0.35', lw=0.9, ls='--', alpha=0.6, zorder=1)

    for ax, ylabel, head in (
            (ax_a, r'Relative Dispersion $\varepsilon$', 1.18),
            (ax_b, r'Relative Dispersion Difference '
                   r'$\varepsilon_2 - \varepsilon_1$', 1.06)):
        ax.set_xscale('log')
        ax.set_xlabel(r'$\sigma_S / \gamma$  [s]', fontsize=18)
        ax.set_ylabel(ylabel, fontsize=18)
        ax.grid(alpha=0.3, which='both')
        ax.tick_params(axis='both', which='major', labelsize=15,
                       direction='in', top=True, right=True)
        ax.tick_params(axis='both', which='minor', direction='in',
                       top=True, right=True)
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, lo + head * (hi - lo))

    # model legend: panel (a) only, since colour means nothing in (b)
    h_model = [ax_a.plot([], [], marker='o', linestyle='none', ms=8.0,
                         markerfacecolor='none', markeredgecolor=c,
                         markeredgewidth=1.5, label=lbl)[0]
               for c, lbl in ((_S10_M1, 'Model 1'), (_S10_M2, 'Model 2'))]
    ax_a.legend(handles=h_model, loc='upper left', fontsize=12,
                framealpha=0.9, borderpad=0.6, handletextpad=0.5)

    # source legend: shared by both panels, neutral colour, one row below
    h_src = [ax_a.plot([], [], marker=mk, linestyle='none', ms=ms + 0.5,
                       markerfacecolor='none', markeredgecolor='0.30',
                       markeredgewidth=1.5, label=lbl)[0]
             for mk, ms, lbl in _S10_SOURCES]
    fig.legend(handles=h_src, loc='lower center', ncol=3, fontsize=12,
               frameon=False, bbox_to_anchor=(0.5, -0.04),
               title='Data source', title_fontsize=12,
               handletextpad=0.5, columnspacing=2.2)

    plt.tight_layout()
    plt.savefig(os.path.join(outdir, 'figS10.png'), dpi=300,
                bbox_inches='tight')
    plt.close(); print("  Saved figS10.png")

    # console table: worst departure from a single curve, per model
    allr = np.concatenate([s[0] for s in series])
    print(f"    {allr.size} points from 3 sources, "
          f"sigma_S/gamma in [{allr.min():.2e}, {allr.max():.2e}] s")
    for name, i_m, i_s in (('Model 1', 1, 2), ('Model 2', 3, 4),
                           ('eps2 - eps1', 5, 6)):
        vals = np.concatenate([s[i_m] for s in series])
        sds = np.concatenate([s[i_s] for s in series])
        print(f"       {name:12s} range {vals.min():.4f} to {vals.max():.4f}, "
              f"max seed s.d. {sds.max():.2e}")

def main():
    parser = argparse.ArgumentParser(description='Generate all paper figures.')
    parser.add_argument('--datadir', type=str, default='data')
    parser.add_argument('--outdir', type=str, default='figures')
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    print("Generating figures...")
    #plot_figure1(args.datadir, args.outdir)
    #plot_figure2(args.datadir, args.outdir)
    #plot_figure3(args.datadir, args.outdir)
    #plot_figure4(args.datadir, args.outdir)
    #plot_figS1(args.datadir, args.outdir)
    #plot_figS2(args.datadir, args.outdir)
    #plot_figS3(args.datadir, args.outdir)
    #plot_figS4(args.datadir, args.outdir)
    #plot_figS5(args.datadir, args.outdir)
    #plot_figS6(args.outdir)
    #plot_figS7(args.datadir, args.outdir)
    #plot_figS8(args.datadir, args.outdir)
    #plot_figS9(args.datadir, args.outdir)
    plot_figS10(args.datadir, args.outdir)
    print("\nAll figures generated.")

if __name__ == '__main__':
    main()