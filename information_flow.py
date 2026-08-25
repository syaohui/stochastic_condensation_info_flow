"""
information_flow.py — Liang information-flow estimators for Model 2.

Implements:
  - Eq. 8 (reduced estimator) for T_{S→z}
  - Eq. 9 (full two-regressor estimator) for T_{z→S}
  - Eq. 10a (analytical expression) for validation
  - Correlation coefficient ρ_{zS}

All estimators work on the pre-computed ensemble covariances stored in
the Model 2 summary file. Covariances are sampled every M_STEP=100 steps
in run_simulation.py, so m_step=1 is used here (no further striding needed).
"""

import numpy as np


# ============================================================
# Numerical estimators (from pre-computed covariances)
# ============================================================

def estimate_T_S_to_z(C_SS, C_zz, C_zS, C_S_dz, m_step=1):
    """
    Reduced information-flow estimator S → z  (Eq. 8).

    T_hat_{S→z}(t) = [C_{S,dz}(t) / C_{SS}(t)] * [C_{zS}(t) / C_{zz}(t)]

    Covariances are already sampled every M_STEP=100 simulation steps,
    so m_step=1 evaluates every saved point. The physical time window
    M_STEP*dt is baked into C_S_dz by run_simulation.py, ensuring
    stability as dt→0.

    Parameters
    ----------
    C_SS   : array, shape (n_if,)      Var(S) at each sample point
    C_zz   : array, shape (n_if,)      Var(z) at each sample point
    C_zS   : array, shape (n_if,)      Cov(z, S) at each sample point
    C_S_dz : array, shape (n_if-1,)    Cov(S_t, dz/dt) at each sample point
    m_step : int                        Step stride (default: 1)

    Returns
    -------
    time_indices : list of int    Valid sample indices
    T_hat        : np.array       Information flow values at those indices
    """
    T_minus1 = len(C_S_dz)
    time_indices = []
    T_hat = []

    for t in range(m_step - 1, T_minus1, m_step):
        css  = C_SS[t]
        czz  = C_zz[t]
        czs  = C_zS[t]
        csdz = C_S_dz[t]

        # guard against zero denominators
        if css == 0.0 or czz == 0.0:
            continue

        # guard against zero numerators
        if abs(csdz) < 1e-10 or abs(czs) < 1e-10:
            continue

        val = (csdz / css) * (czs / czz)

        if np.isfinite(val):
            time_indices.append(t)
            T_hat.append(val)

    return time_indices, np.array(T_hat)


def estimate_T_z_to_S(C_SS, C_zz, C_zS, C_z_dS, C_S_dS, m_step=1):
    """
    Full two-regressor information-flow estimator z → S  (Eq. 9).

    T_hat_{z→S}(t) = [C_{SS} C_{Sz} C_{z,dS} - C_{Sz}^2 C_{S,dS}]
                    / [C_{SS}^2 C_{zz} - C_{SS} C_{Sz}^2]

    Covariances are already sampled every M_STEP=100 simulation steps.

    Parameters
    ----------
    C_SS   : array, shape (n_if,)      Var(S)
    C_zz   : array, shape (n_if,)      Var(z)
    C_zS   : array, shape (n_if,)      Cov(z, S)   [= C_{Sz}]
    C_z_dS : array, shape (n_if-1,)    Cov(z_t, dS/dt)
    C_S_dS : array, shape (n_if-1,)    Cov(S_t, dS/dt)
    m_step : int                        Step stride (default: 1)

    Returns
    -------
    time_indices : list of int
    T_hat        : np.array
    """
    T_minus1 = len(C_z_dS)
    time_indices = []
    T_hat = []

    for t in range(m_step - 1, T_minus1, m_step):
        css  = C_SS[t]
        czz  = C_zz[t]
        csz  = C_zS[t]
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
# Convenience: compute everything from a single seed file
# ============================================================

def compute_all_info_flow(seed_file, m_step=1):
    """
    Load one per-seed npz file produced by run_simulation.py and compute
    all information-flow quantities.

    Parameters
    ----------
    seed_file : str
        Path to a single summary_gamma*_seed*.npz file.
    m_step : int
        Step stride (default: 1 — covariances already subsampled every
        M_STEP=100 steps in run_simulation.py).

    Returns a dict with keys:
      T_S_to_z_idx, T_S_to_z      — numerical S→z
      T_z_to_S_idx, T_z_to_S      — numerical z→S
      T_S_to_z_ana                 — analytical S→z at same times
      rho_zS                       — correlation coefficient (sampled)
    """
    data = np.load(seed_file, allow_pickle=False)

    C_SS      = data['m2_C_SS'].astype(np.float64)
    C_zz      = data['m2_C_zz'].astype(np.float64)
    C_zS      = data['m2_C_zS'].astype(np.float64)
    C_S_dz    = data['m2_C_S_dz'].astype(np.float64)
    C_z_dS    = data['m2_C_z_dS'].astype(np.float64)
    C_S_dS    = data['m2_C_S_dS'].astype(np.float64)
    gamma     = float(data['m2_gamma'])
    dt        = float(data['m2_dt'])
    M_STEP    = int(data['m2_M_STEP'])

    # Reconstruct time axis for IF sample points
    n_if      = len(C_SS)
    time_axis = np.arange(n_if) * M_STEP * dt

    idx_sz, T_sz = estimate_T_S_to_z(C_SS, C_zz, C_zS, C_S_dz, m_step=m_step)
    idx_zs, T_zs = estimate_T_z_to_S(C_SS, C_zz, C_zS, C_z_dS, C_S_dS, m_step=m_step)

    T_sz_ana = T_S_to_z_analytical(time_axis[idx_sz], gamma)

    return dict(
        T_S_to_z_idx=np.array(idx_sz),
        T_S_to_z=T_sz,
        T_z_to_S_idx=np.array(idx_zs),
        T_z_to_S=T_zs,
        T_S_to_z_ana=T_sz_ana,
        rho_zS=data['m2_rho_zS'],
        time_axis=time_axis,
    )
