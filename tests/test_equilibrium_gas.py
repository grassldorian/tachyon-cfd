"""Equilibrium gas model: kernel compile + short GPU run smoke test."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig, PROPELLANTS
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

cfg = SimConfig()
cfg.axisymmetric = True
cfg.axis_location = "center"
cfg.propellant = "LOX/RP-1 (kerosene)"
for k, v in PROPELLANTS[cfg.propellant].items():
    setattr(cfg, k, v)
cfg.inlet_p0 = 7.0e6
cfg.gas_model = "equilibrium"

mask = load_mask(str(ROOT / "examples" / "nozzle_small.png"),
                 cfg.meters_per_pixel, cfg.svg_raster_px,
                 smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma,
                 mesh_scale=cfg.mesh_scale)
sol = GPUSolver(mask, cfg)
print("equilibrium kernels compiled, tables uploaded")
for i in range(4):
    sol.step(250)
    snap = sol.snapshot()
    f = snap["fields"]
    print(f"step {sol.step_count:5d}  res {sol.residual:.2e}  "
          f"Mach max {np.nanmax(f['Mach']):.3f}  "
          f"T [{np.nanmin(f['Temperature [K]']):.0f}, "
          f"{np.nanmax(f['Temperature [K]']):.0f}] K  "
          f"F {snap['meta']['performance']['F']:.4g}")
    assert np.isfinite(np.nanmax(f["Mach"])), "non-finite fields"

# ambient sanity: quiescent corner should sit at farfield conditions
T = snap["fields"]["Temperature [K]"]
p = snap["fields"]["Pressure [Pa]"]
corner_T = np.nanmedian(T[:30, -60:])
corner_p = np.nanmedian(p[:30, -60:])
print(f"ambient corner: T={corner_T:.0f} K (288 expected)  "
      f"p={corner_p:.0f} Pa (101325 expected)")
assert abs(corner_T - 288.15) < 30, corner_T
assert abs(corner_p / 101325.0 - 1) < 0.1, corner_p
print("equilibrium smoke test OK")
