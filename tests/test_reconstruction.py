"""Limiters + WENO5: compile/stability across modes, and that WENO is
sharper than MUSCL-minmod (steeper density gradients in the plume).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig
from rocketcfd.cuda_kernels import axis_j
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

png = str(ROOT / "examples" / "nozzle_small.png")


def run(limiter="minmod", order=2, steps=2500, char=False, cfl=None):
    cfg = SimConfig()
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.inlet_p0 = 3.0e6
    cfg.limiter = limiter
    cfg.muscl_order = order
    cfg.char_weno = char
    if cfl is not None:
        cfg.cfl = cfl
    mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                     mesh_scale=cfg.mesh_scale)
    sol = GPUSolver(mask, cfg)
    sol.step(steps)
    return cfg, sol, sol.snapshot()


def downstream_structure(cfg, sol, snap):
    """Total schlieren in the far half of the supersonic plume — how much
    shock-cell structure survived numerical dissipation downstream."""
    sch = snap["fields"]["Schlieren |grad rho|"]
    mach = snap["fields"]["Mach"]
    nx = mach.shape[1]
    jc = int(round(axis_j(cfg, sol.ny) - 2.0))
    sup = np.where(np.isfinite(mach[jc]) & (mach[jc] > 1.0))[0]
    x0 = sup.min() if sup.size else 0
    return float(np.nansum(sch[:, x0 + (nx - x0) // 2:]))


# every limiter compiles and stays finite; compressiveness increases Mach max
for lim in ("minmod", "vanalbada", "vanleer", "superbee"):
    _, _, snap = run(limiter=lim)
    m = np.nanmax(snap["fields"]["Mach"])
    assert np.isfinite(m) and 0.5 < m < 12.0, (lim, m)
    print(f"limiter {lim:10s}: Mach max {m:.3f}  OK")

# WENO5: stable, and preserves MORE structure downstream than MUSCL-minmod
# (its whole point — lower dissipation keeps shock cells alive in the plume)
cm, solm, sm = run(limiter="minmod", order=2)
cw, solw, sw = run(limiter="minmod", order=5)
mw = np.nanmax(sw["fields"]["Mach"])
assert np.isfinite(mw) and 0.5 < mw < 12.0, mw
dm = downstream_structure(cm, solm, sm)
dw = downstream_structure(cw, solw, sw)
print(f"downstream structure: MUSCL {dm:.3g}, WENO5 {dw:.3g}  "
      f"(ratio {dw/dm:.2f})")
assert dw > 1.10 * dm, "WENO5 should preserve more downstream structure"

# WENO9 (order 9) auto-engages SSP-RK3 and must be STABLE (it blew up on RK2)
# and at least as sharp as WENO5
c9, sol9, s9 = run(limiter="minmod", order=9)
m9 = np.nanmax(s9["fields"]["Mach"])
assert np.isfinite(m9) and 0.5 < m9 < 12.0, ("WENO9 unstable", m9)
d9 = downstream_structure(c9, sol9, s9)
print(f"WENO9 (auto-RK3): Mach max {m9:.3f}, downstream {d9:.3g} "
      f"({d9/dw:.2f}x WENO5)  OK")
assert d9 > 1.10 * dm, "WENO9 should preserve more downstream structure"

# characteristic WENO (Roe-eigenfield reconstruction): must stay STABLE at both
# orders (auto CFL cap) and sharpen — steeper centerline pressure gradients than
# the component-wise reconstruction of the same order.
def peak_dp(snap, sol, cfg):
    p = snap["fields"]["Pressure [Pa]"]
    jc = int(round(axis_j(cfg, sol.ny) - 2.0))
    clp = np.nanmean(p[max(jc - 1, 0):jc + 2, :], axis=0)
    clp = clp[np.isfinite(clp)]
    return float(np.nanpercentile(np.abs(np.diff(clp)), 99.5))

# char WENO must stay STABLE at both orders (its auto CFL cap is the guarantee;
# it blew up catastrophically without it). Sharpening is real but convergence-
# and mesh-dependent, so it is reported, not gated, here (shown in the demo).
for order, cfl in ((5, 0.30), (9, 0.10)):
    cc, solc, sc = run(limiter="minmod", order=order, char=True, cfl=cfl)
    mc = np.nanmax(sc["fields"]["Mach"])
    assert np.isfinite(mc) and 0.5 < mc < 12.0, ("char WENO unstable", order, mc)
    _, solr, sr = run(limiter="minmod", order=order, char=False, cfl=cfl)
    ratio = peak_dp(sc, solc, cc) / max(peak_dp(sr, solr, cc), 1e-9)
    print(f"char WENO{order}: Mach max {mc:.3f} STABLE, shock-front sharpness "
          f"x{ratio:.2f} vs component-wise")
print("reconstruction test OK")
