"""
config.py — Shared physical constants and simulation parameters.

References
----------
Shu et al. (2026), "Stochastic Condensation with Information Flow
for Causality Analysis", Geophys. Res. Lett.

McGraw & Liu (2006), "Brownian drift-diffusion model for evolution of
droplet size distributions in turbulent clouds", Geophys. Res. Lett.
"""

import numpy as np

# ============================================================
# Physical constants  (Table S1 / Text S2 of the paper)
# ============================================================
N_c   = 1.0e2        # Droplet number concentration  [cm^{-3}]
rho_w = 1.0           # Liquid water density           [g cm^{-3}]
kT    = 167.8          # Condensational growth rate coefficient  [μm^2 s^{-1}]

# Unit conversion factor: μm^3 → cm^3
UM3_TO_CM3 = 1.0e-12

# Derived combined parameter used throughout
# (replaces the former "ND" variable)
# ND_eff ≡ N_c * rho_w * UM3_TO_CM3
# This appears whenever r is in μm and L is in g cm^{-3}.
ND_eff = N_c * rho_w * UM3_TO_CM3   # = 1.0e-10

# ============================================================
# Default simulation parameters
# ============================================================
N_droplets  = 100_000       # Number of Monte Carlo droplets
sigmaS_default = 0.01       # Std dev of saturation ratio fluctuation
gamma_default  = 0.3         # Correlation rate  [s^{-1}]
RANDOM_SEED    = 21          # For reproducibility

# ============================================================
# Derived initial conditions (monodisperse at r0)
# ============================================================
# Initial liquid water content  L = (4π/3) r0^3 N_c ρ_w
#   where r0 = 10 μm = 10e-4 cm
# In code-units (r in μm):
#   L = (4π/3) * r0_um^3 * ND_eff
r0_um = 10.0                                             # [μm]
initial_L = (4.0 * np.pi / 3.0) * r0_um**3 * ND_eff     # [g cm^{-3}]
# ≈ 4.18879e-7 g cm^{-3}

# Initial squared radius
z0 = (3.0 / (4.0 * np.pi))**(2.0/3.0) * (initial_L / ND_eff)**(2.0/3.0)
# z0 = r0^2 = 100 μm^2

# ============================================================
# Helper functions
# ============================================================
def compute_S_mean(sigmaS, gamma):
    """Steady-state mean saturation ratio (Eq. S5)."""
    return 1.0 - np.pi * (ND_eff / initial_L)**(2.0/3.0) * kT * sigmaS**2 / gamma


def compute_DZ(sigmaS, gamma):
    """Diffusion coefficient in z-coordinate (Eq. 2b)."""
    return kT**2 * sigmaS**2 / gamma


def compute_tau(sigmaS, gamma):
    """Relaxation time τ = z0² / (2 D_Z)."""
    DZ = compute_DZ(sigmaS, gamma)
    return z0**2 / (2.0 * DZ)


# ============================================================
# Histogram parameters (shared across all plotting scripts)
# ============================================================
HIST_RMAX   = 30       # Maximum radius for histograms [μm]
HIST_NBINS  = 500      # Number of histogram bins
