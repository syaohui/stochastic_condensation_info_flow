# Stochastic Condensation with Information Flow for Causality Analysis

*DOI will be assigned upon publication.*

Data repository for:

> Shu, Y., Liu, Y., Zhang, T., McGraw, R., & Li, X. (2026). Stochastic Condensation with Information Flow for Causality Analysis. *Geophysical Research Letters*.

## Overview

This repository provides the **summary data** needed to reproduce all figures in the paper. Raw per-droplet simulation data (~5–10 GB per case) is excluded and can be regenerated from the model code.

The study presents Monte Carlo simulations for two stochastic condensation models:

- **Model 1** (McGraw & Liu, 2006): Brownian drift–diffusion model where the saturation ratio fluctuation is treated as white noise.
- **Model 2** (this study): Generalized model where the saturation ratio *S* follows an Ornstein–Uhlenbeck process, coupled to droplet growth.

An information-flow framework (Liang, 2014) is embedded into Model 2 to quantify the direction and strength of causal influence between *S* and droplet size.

## Repository Contents

Only summary `.npz` files are tracked here. Raw data files (`*_raw.npz`) are excluded via `.gitignore`.

```
├── data/
│   ├── gamma0.3/
│   │   └── summary_gamma0.3.npz        # ~15 MB
│   ├── gamma8.0/
│   │   └── summary_gamma8.0.npz        # ~15 MB
│   └── sweeps/
│       ├── gamma_sweep_RD.npz
│       ├── sigmaS_sweep_gamma0.3.npz
│       └── sigmaS_sweep_gamma8.0.npz
├── .gitignore
└── README.md
```
## Code Availability

The full simulation and analysis code is included in this repository.

## Physical Parameters

| Symbol | Value | Unit | Description |
|--------|-------|------|-------------|
| N<sub>c</sub> | 1.0 × 10² | cm⁻³ | Droplet number concentration |
| ρ<sub>w</sub> | 1.0 | g cm⁻³ | Liquid water density |
| k | 167.8 | μm² s⁻¹ | Condensational growth rate coefficient |
| σ<sub>S</sub> | 0.01 | — | Std dev of saturation ratio fluctuation |
| r<sub>0</sub> | 10 | μm | Initial monodisperse droplet radius |
| N | 100000 | — | Number of Monte Carlo droplets |

## Figures Reproduced from This Data

| File | Paper Figure | Description |
|------|-------------|-------------|
| `fig1.png` | Fig. 1 | Mean/std/ε/β vs time (γ = 0.3 s⁻¹)(γ = 8.0 s⁻¹) |
| `fig2.png` | Fig. 2 | Steady-state ε vs γ |
| `fig3.png` | Fig. 3 | Steady-state ε vs σ_S |
| `fig4.png` | Fig. 4 | Information flow vs time & Information flow vs relative dispersion|
| `figS1.png` | Fig. S1 | Statistics (normalized time) |
| `figS2.png` | Fig. S2 | Steady-state size distributions |
| `figS3.png` | Fig. S3 | Steady-state start time vs σ_S |
| `figS4.png` | Fig. S4 | Droplet size distribution heatmaps (Time [s]) |
| `figS5.png` | Fig. S5 | Heatmaps (normalized time) |
| `figS6.png` | Fig. S6 | Analytical information flow with γ limits |


## Data Availability

Raw data files will be archived on Zenodo upon publication.

## References

- McGraw, R., & Liu, Y. (2006). Brownian drift–diffusion model for evolution of droplet size distributions in turbulent clouds. *Geophys. Res. Lett.*, 33(3). https://doi.org/10.1029/2005GL023545
- Liang, X. S. (2014). Unraveling the cause-effect relation between time series. *Phys. Rev. E*, 90(5). https://doi.org/10.1103/PhysRevE.90.052150

## License

[To be decided]

## Acknowledgments

This study is jointly supported by the U.S. Department of Energy's Atmospheric System Research (ASR) Program and the Brookhaven National Laboratory Directed Research and Development Program (LDRD-25-008).
