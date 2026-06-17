"""Downstream mesh stretching: the engine is on the uniform grid, so thrust
and mass flow must be (nearly) unchanged when stretching is enabled, while
the plume domain physically extends. Also checks stability.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

png = str(ROOT / "examples" / "nozzle_small.png")


def run(stretch, steps=5000):
    cfg = SimConfig()
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.inlet_p0 = 4.0e6
    cfg.plume_stretch = stretch
    mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                     mesh_scale=cfg.mesh_scale)
    sol = GPUSolver(mask, cfg)
    sol.step(steps)
    snap = sol.snapshot()
    return sol, snap


sol0, s0 = run(1.0)
sol1, s1 = run(1.03)

p0 = s0["meta"]["performance"]
p1 = s1["meta"]["performance"]
m0 = np.nanmax(s0["fields"]["Mach"])
m1 = np.nanmax(s1["fields"]["Mach"])
assert np.isfinite(m1) and 0.5 < m1 < 12.0, m1     # stable with stretching

L0 = float(sol0.x_centers[-1])
L1 = float(sol1.x_centers[-1])
dF = abs(p1["F"] - p0["F"]) / abs(p0["F"])
dm = abs(p1["mdot"] - p0["mdot"]) / abs(p0["mdot"])

print(f"stretch off: F {p0['F']:.5g} N, mdot {p0['mdot']:.4g} kg/s, "
      f"domain {L0*1e3:.0f} mm, Mach max {m0:.3f}")
print(f"stretch on : F {p1['F']:.5g} N, mdot {p1['mdot']:.4g} kg/s, "
      f"domain {L1*1e3:.0f} mm, Mach max {m1:.3f}")
print(f"thrust change {dF*100:.2f} %, mass-flow change {dm*100:.2f} %, "
      f"domain grew {L1/L0:.2f}x")

# engine sits on the uniform grid -> performance must be essentially unchanged
assert dF < 0.03, f"thrust moved {dF*100:.1f}% with stretching (should not)"
assert dm < 0.03, f"mass flow moved {dm*100:.1f}% with stretching (should not)"
# the plume domain must physically extend
assert L1 > 1.2 * L0, f"plume did not extend ({L1/L0:.2f}x)"
print("stretch test OK")
