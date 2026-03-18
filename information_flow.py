"""
information_flow.py — Liang information-flow estimators for Model 2.

Implements:
  - Eq. 8 (reduced estimator) for T_{S→z}
  - Eq. 9 (full two-regressor estimator) for T_{z→S}
  - Eq. 10a (analytical expression) for validation
  - Correlation coefficient ρ_{zS}

All estimators work on the pre-computed ensemble covariances stored in
the Model 2 summary file, avoiding the need to reload 20 GB of raw data.
"""

import numpy as np


# ============================================================
# Numerical estimators (from pre-computed covariances)
# ============================================================

def estimate_T_S_to_z(C_SS, C_zz, C_zS, C_S_dz):
    """
    Reduced information-flow estimator S → z  (Eq. 8).

    T_hat_{S→z}(t) = [C_{S,dz}(t) / C_{SS}(t)] * [C_{zS}(t) / C_{zz}(t)]

    Parameters
    ----------
    C_SS   : array, shape (T,)      Var(S) at each time step
    C_zz   : array, shape (T,)      Var(z) at each time step
    C_zS   : array, shape (T,)      Cov(z, S) at each time step
    C_S_dz : array, shape (T-1,)    Cov(S_t, dz/dt) at each time step

    Returns
    -------
    time_indices : list of int    Valid time indices (no NaN, no zero)
    T_hat        : list of float  Information flow values at those indices
    """
    T_minus1 = len(C_S_dz)
    time_indices = []
    T_hat = []

    for t in range(T_minus1):
        css  = C_SS[t]
        czz  = C_zz[t]
        czs  = C_zS[t]
        csdz = C_S_dz[t]

        # Guard against zero denominators
        if css == 0.0 or czz == 0.0:
            continue

        # Guard against zero numerators (e.g. t=0 initial condition)
        if abs(csdz) < 1e-10 or abs(czs) < 1e-10:
            continue

        val = (csdz / css) * (czs / czz)

        # Skip NaN, Inf, or zero values
        if np.isfinite(val):
            time_indices.append(t)
            T_hat.append(val)

    return time_indices, np.array(T_hat)


def estimate_T_z_to_S(C_SS, C_zz, C_zS, C_z_dS, C_S_dS):
    """
    Full two-regressor information-flow estimator z → S  (Eq. 9).

    T_hat_{z→S}(t) = [C_{SS} C_{Sz} C_{z,dS} - C_{Sz}^2 C_{S,dS}]
                    / [C_{SS}^2 C_{zz} - C_{SS} C_{Sz}^2]

    Parameters
    ----------
    C_SS   : array, shape (T,)      Var(S)
    C_zz   : array, shape (T,)      Var(z)
    C_zS   : array, shape (T,)      Cov(z, S)   [= C_{Sz}]
    C_z_dS : array, shape (T-1,)    Cov(z_t, dS/dt)
    C_S_dS : array, shape (T-1,)    Cov(S_t, dS/dt)

    Returns
    -------
    time_indices : list of int
    T_hat        : list of float
    """
    T_minus1 = len(C_z_dS)
    time_indices = []
    T_hat = []

    for t in range(T_minus1):
        css = C_SS[t]
        czz = C_zz[t]
        csz = C_zS[t]        # Cov(S, z) = Cov(z, S)
        czds = C_z_dS[t]
        csds = C_S_dS[t]

        denom = css**2 * czz - css * csz**2
        if denom == 0.0:
            continue

        numer = css * csz * czds - csz**2 * csds
        val = numer / denom

        if np.isfinite(val):
            time_indices.append(t)
            T_hat.append(val)

    return time_indices, np.array(T_hat)


# ============================================================
# Analytical expression  (Eq. 10a)
# ============================================================

def T_S_to_z_analytical(t, gamma):
    """
    Analytical information flow from S to z  (Eq. 10a).

    T_{S→z}(t) = γ(1 - e^{-γt})^2 / [2γt + 4e^{-γt} - e^{-2γt} - 3]

    Parameters
    ----------
    t     : array-like, time [s]  (must be > 0)
    gamma : float, correlation rate [s^{-1}]
    """
    t = np.asarray(t, dtype=np.float64)
    egt  = np.exp(-gamma * t)
    e2gt = np.exp(-2.0 * gamma * t)

    numer = gamma * (1.0 - egt)**2
    denom = 2.0 * gamma * t + 4.0 * egt - e2gt - 3.0

    # Avoid division by zero at t=0
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.where(np.abs(denom) > 1e-30, numer / denom, np.nan)
    return result


def T_S_to_z_limit_gamma_inf(t):
    """Asymptotic limit γ → ∞:  T_{S→z} = 1/(2t)."""
    return 1.0 / (2.0 * np.asarray(t, dtype=np.float64))


def T_S_to_z_limit_gamma_zero(t):
    """Asymptotic limit γ → 0:  T_{S→z} = 3/(2t)."""
    return 3.0 / (2.0 * np.asarray(t, dtype=np.float64))


# ============================================================
# Convenience: compute everything from a summary dict
# ============================================================

def compute_all_info_flow(summary_m2):
    """
    Given a Model 2 summary dict, compute all information-flow quantities.

    Returns a dict with keys:
      T_S_to_z_idx, T_S_to_z      — numerical S→z
      T_z_to_S_idx, T_z_to_S      — numerical z→S
      T_S_to_z_ana                 — analytical S→z at same times
      rho_zS                       — correlation coefficient
    """
    C_SS   = summary_m2['C_SS']
    C_zz   = summary_m2['C_zz']
    C_zS   = summary_m2['C_zS']
    C_S_dz = summary_m2['C_S_dz']
    C_z_dS = summary_m2['C_z_dS']
    C_S_dS = summary_m2['C_S_dS']
    time_axis = summary_m2['time_axis']
    gamma  = float(summary_m2['gamma'])

    # Numerical estimators
    idx_sz, T_sz = estimate_T_S_to_z(C_SS, C_zz, C_zS, C_S_dz)
    idx_zs, T_zs = estimate_T_z_to_S(C_SS, C_zz, C_zS, C_z_dS, C_S_dS)

    # Analytical at the same time points
    T_sz_ana = T_S_to_z_analytical(time_axis[idx_sz], gamma)

    return dict(
        T_S_to_z_idx=np.array(idx_sz),
        T_S_to_z=T_sz,
        T_z_to_S_idx=np.array(idx_zs),
        T_z_to_S=T_zs,
        T_S_to_z_ana=T_sz_ana,
        rho_zS=summary_m2['rho_zS'],
    )
