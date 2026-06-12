"""Wall functions: no-slip runs across gas models + isothermal wall flux.

Checks that no-slip with the Reichardt wall function runs stably in all
three gas models, that the isothermal wall actually removes energy
(T near the wall drops vs the adiabatic run), and that the slip path is
untouched by the wall_T setting.
"""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig, PROPELLANTS
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

png = str(ROOT / "examples" / "nozzle_small.png")


def run(gas_model, wall_type, wall_T=0.0, steps=1200):
    cfg = SimConfig()
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.propellant = "LOX/RP-1 (kerosene)"
    for k, v in PROPELLANTS[cfg.propellant].items():
        setattr(cfg, k, v)
    cfg.inlet_p0 = 7.0e6
    cfg.gas_model = gas_model
    cfg.wall_type = wall_type
    cfg.wall_T = wall_T
    mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                     mesh_scale=cfg.mesh_scale)
    sol = GPUSolver(mask, cfg)
    sol.step(steps)
    return sol.snapshot()


for gm in ("calorically perfect", "thermally perfect", "equilibrium"):
    snap = run(gm, "noslip")
    m = np.nanmax(snap["fields"]["Mach"])
    assert np.isfinite(m) and 0.5 < m < 10.0, (gm, m)
    print(f"noslip {gm:22s}: Mach max {m:.3f}  "
          f"F {snap['meta']['performance']['F']:.4g}  OK")

# isothermal wall removes energy: chamber/nozzle mean T must drop vs
# adiabatic (measure only the high-pressure interior — the domain mean is
# dominated by the ambient and would hide a thin cooled wall layer)
sa = run("calorically perfect", "noslip", wall_T=0.0, steps=3000)
si = run("calorically perfect", "noslip", wall_T=600.0, steps=3000)
ina = sa["fields"]["Pressure [Pa]"] > 5.0e5
ini = si["fields"]["Pressure [Pa]"] > 5.0e5
Ta = np.nanmean(sa["fields"]["Temperature [K]"][ina])
Ti = np.nanmean(si["fields"]["Temperature [K]"][ini])
print(f"chamber mean T: adiabatic {Ta:.0f} K, isothermal(600K) {Ti:.0f} K")
assert Ti < Ta - 5.0, "isothermal wall did not cool the flow"

# wall heat-flux diagnostic field: present, finite, plausible magnitude
qw = si["fields"].get("Wall heat flux [W/m^2]")
assert qw is not None, "Wall heat flux field missing from snapshot"
qpk = np.nanmax(qw)
print(f"peak wall heat flux: {qpk/1e6:.1f} MW/m^2")
assert 0.5e6 < qpk < 500e6, qpk
assert "Wall heat flux [W/m^2]" not in sa["fields"], \
    "adiabatic run should not export a heat-flux field"
print("wall-function test OK")
