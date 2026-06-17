"""Radiation heat-transfer validation + flux-scheme smoke test."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

png = str(ROOT / "examples" / "nozzle_small.png")


def run(scheme="hllc", wall_T=0.0, emiss=0.0, steps=3000):
    cfg = SimConfig()
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.inlet_p0 = 4.0e6
    cfg.flux_scheme = scheme
    cfg.wall_type = "noslip"
    cfg.wall_T = wall_T
    cfg.wall_emissivity = emiss
    mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma)
    sol = GPUSolver(mask, cfg)
    sol.step(steps)
    return sol.snapshot()


# every shipped flux scheme gives a sane supersonic field
for sch in ("hllc", "hll", "roe", "ausm"):
    snap = run(scheme=sch)
    m = np.nanmax(snap["fields"]["Mach"])
    assert np.isfinite(m) and 1.0 < m < 12.0, (sch, m)
    print(f"scheme {sch:7s}: Mach max {m:.3f}  OK")

# radiation increases the wall heat flux (adds q_rad on top of convective)
s_conv = run(wall_T=800.0, emiss=0.0)
s_rad = run(wall_T=800.0, emiss=0.3)
q_conv = np.nanmax(s_conv["fields"]["Wall heat flux [W/m^2]"])
q_rad = np.nanmax(s_rad["fields"]["Wall heat flux [W/m^2]"])
print(f"peak wall flux: convective {q_conv/1e6:.1f} MW/m^2, "
      f"with radiation {q_rad/1e6:.1f} MW/m^2")
assert q_rad > q_conv, "radiation should add to the wall heat flux"
print("radiation test OK")
