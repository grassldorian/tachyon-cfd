"""Validation: converging-diverging nozzle vs 1D isentropic theory.

Generates a smooth axisymmetric CD nozzle (area ratio 4) at its design
pressure ratio (pe ~ p_amb: shock-free, converges to steady state) and
checks three things:
  1. choked mass flow vs  mdot = p0 A* Gamma(g) / sqrt(R T0)   (within 3 %)
  2. mass conservation: mdot agrees across 4 nozzle planes      (within 6 %)
  3. centerline exit Mach vs the area-ratio relation, one-sided band
     [-15 %, +3 %].  The deficit is a KNOWN solver bias: ambient pressure
     creeps upstream through the numerically smeared subsonic wall layer
     and recompresses the expansion (measured exit-Mach bias: -2.5 % at
     5 kPa ambient, -9 % at 30 kPa, -19 % at 98.6 kPa on this geometry
     family), plus MUSCL/minmod dissipation along the expansion.  Wall
     functions / finer near-wall treatment (REALISM.md item 4) should
     shrink it; this test pins the current envelope so regressions and
     improvements both show up.
Run with turbulence OFF: isentropic theory is loss-free, and the SST inlet
eddy viscosity (mut/mu = 50) alone costs ~5-8 % of mass flow.
"""
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig
from rocketcfd.cuda_kernels import axis_j
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

# ---------------------------------------------------------------- geometry
W, H = 800, 400
CY = H / 2 - 0.5                      # axis through the image center
R_T, R_E, R_C = 30.0, 60.0, 80.0      # px: throat, exit (eps=4), chamber
X0, X_CONV, X_TH, X_EXIT = 40, 200, 300, 700
WALL_PX = 8

img = np.full((H, W, 3), 255, dtype=np.uint8)
yy = np.abs(np.arange(H)[:, None] - CY)
xs = np.arange(W)

half = np.full(W, -1.0)
m = (xs >= X0) & (xs < X_CONV)
half[m] = R_C
m = (xs >= X_CONV) & (xs < X_TH)                       # cosine contraction
t = (xs[m] - X_CONV) / (X_TH - X_CONV)
half[m] = R_C + (R_T - R_C) * 0.5 * (1 - np.cos(np.pi * t))
m = (xs >= X_TH) & (xs <= X_EXIT)                      # smooth bell to eps=4
t = (xs[m] - X_TH) / (X_EXIT - X_TH)
half[m] = R_T + (R_E - R_T) * (1.5 * t ** 2 - 0.5 * t ** 3)

hh = half[None, :]
img[(hh >= 0) & (yy >= hh) & (yy < hh + WALL_PX)] = (0, 0, 0)
back = (xs[None, :] >= X0 - 2 * WALL_PX) & (xs[None, :] < X0) & (yy < R_C + WALL_PX)
img[np.broadcast_to(back, (H, W))] = (0, 0, 0)
inlet = (xs[None, :] >= X0 - WALL_PX) & (xs[None, :] < X0) & (yy < R_C * 0.85)
img[np.broadcast_to(inlet, (H, W))] = (0, 80, 255)
img[:WALL_PX, :] = img[-WALL_PX:, :] = img[:, -WALL_PX:] = (255, 40, 30)

png = os.path.join(tempfile.gettempdir(), "isentropic_cd.png")
Image.fromarray(img).save(png)

# ---------------------------------------------------------------- config
cfg = SimConfig()
cfg.meters_per_pixel = 0.001
cfg.axisymmetric = True
cfg.axis_location = "center"
cfg.gamma, cfg.R_gas = 1.4, 287.0
cfg.inlet_p0, cfg.inlet_T0 = 2.0e6, 600.0
cfg.farfield_p = 59800.0              # design condition pe ~ p_amb for eps=4
cfg.farfield_T = 288.0                #   at p0=2 MPa: shock-free and steady
cfg.wall_type = "slip"
cfg.viscous = True                    # laminar only: k-omega SST inlet eddy
cfg.turbulence = False                #   viscosity (mut/mu=50) adds total-
                                      #   pressure loss; theory is loss-free
cfg.inlet_ramp_steps = 3000           # soft start: weaker startup blast

mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                 smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                 mesh_scale=cfg.mesh_scale)
sol = GPUSolver(mask, cfg)
AXIS_ROW = axis_j(cfg, H) - 2.0       # half-integer interior row of the axis


def plane_integrals(snap, col):
    """Choked-plane integrals at interior column `col` (one revolved half),
    restricted to INSIDE the nozzle contour — the same column also crosses
    the ambient outside the wall, whose huge annulus areas (2*pi*r*dx grows
    with r) would otherwise swamp the throat flux with entrainment noise.
    Returns (mdot, mass-weighted Mach)."""
    rho = snap["fields"]["Density [kg/m^3]"][:, col]
    u = snap["fields"]["Velocity u [m/s]"][:, col]
    mach = snap["fields"]["Mach"][:, col]
    j = np.arange(rho.size)
    r = np.abs(j - AXIS_ROW) * cfg.meters_per_pixel
    inside = (j > AXIS_ROW) & (r < half[col] * cfg.meters_per_pixel)
    flux = np.where(np.isfinite(rho * u) & inside,
                    rho * u * 2.0 * np.pi * r * cfg.meters_per_pixel, 0.0)
    mdot = float(np.sum(flux))
    m_avg = float(np.sum(np.where(np.isfinite(mach), mach, 0.0) * flux)
                  / max(mdot, 1e-30))
    return mdot, m_avg


# the throat flow is choked and steady even while the chamber volume still
# "breathes" acoustically, so measure there and average a few samples
COLS = [X_TH, X_TH + 60, 460, 560, X_EXIT - 8]
mdots, machs, planes = [], [], []
for i in range(24):
    sol.step(1000)
    snap_i = sol.snapshot()
    mds = [plane_integrals(snap_i, c)[0] for c in COLS]
    jc = int(AXIS_ROW + 0.5)
    mcl = [float(np.nanmean(snap_i["fields"]["Mach"][jc - 1:jc + 3, c]))
           for c in COLS]
    print(f"step {sol.step_count:6d}  res {sol.residual:.2e}  " +
          "  ".join(f"x{c}: {m:6.3f}kg/s M={mm:5.3f}"
                    for c, m, mm in zip(COLS, mds, mcl)))
    if sol.step_count > 16000:
        mdots.append(mds[0])
        planes.append(mds)
        # centerline Mach: the wall-adjacent annuli carry the most area and
        # are smeared by the cut-cell zone, so mass-weighted averages read
        # low; the axis value is the clean quasi-1D comparison
        machs.append(mcl[-1])
print(f"steps {sol.step_count}, residual {sol.residual:.2e}")

perf = {"mdot": float(np.mean(mdots))}
M_exit_sim = float(np.mean(machs))
plane_means = np.mean(np.array(planes), axis=0)

# ---------------------------------------------------------------- theory
g, R = cfg.gamma, cfg.R_gas
A_star = np.pi * (R_T * cfg.meters_per_pixel) ** 2
Gamma = np.sqrt(g) * (2.0 / (g + 1.0)) ** ((g + 1.0) / (2.0 * (g - 1.0)))
mdot_th = cfg.inlet_p0 * A_star * Gamma / np.sqrt(R * cfg.inlet_T0)

eps = (half[X_EXIT - 8] / R_T) ** 2             # local area ratio at probe
Ms = np.linspace(1.01, 6.0, 20000)              # invert area-ratio relation
area = (1.0 / Ms) * ((2.0 / (g + 1.0)) * (1.0 + 0.5 * (g - 1.0) * Ms ** 2)) \
       ** ((g + 1.0) / (2.0 * (g - 1.0)))
M_exit_th = float(Ms[np.argmin(np.abs(area - eps))])

err_mdot = (perf["mdot"] - mdot_th) / mdot_th
err_mach = (M_exit_sim - M_exit_th) / M_exit_th
cons = float((np.max(plane_means) - np.min(plane_means))
             / np.mean(plane_means))
print(f"mass flow : sim {perf['mdot']:.4g} kg/s  theory {mdot_th:.4g} kg/s "
      f"({err_mdot*100:+.2f} %)")
print(f"conservation: plane-mean mdot spread {cons*100:.2f} % over "
      f"{len(COLS)} planes")
print(f"exit Mach : sim {M_exit_sim:.3f} (centerline)  "
      f"theory {M_exit_th:.3f}  ({err_mach*100:+.2f} %)")

assert abs(err_mdot) < 0.03, f"mass-flow error {err_mdot*100:.1f}% > 3%"
assert cons < 0.06, f"mass-conservation spread {cons*100:.1f}% > 6%"
assert -0.15 < err_mach < 0.03, \
    f"exit-Mach error {err_mach*100:.1f}% outside [-15%, +3%]"
print("isentropic validation OK")
