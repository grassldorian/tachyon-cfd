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


def run(limiter="minmod", order=2, steps=2500):
    cfg = SimConfig()
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.inlet_p0 = 3.0e6
    cfg.limiter = limiter
    cfg.muscl_order = order
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
print("reconstruction test OK")
