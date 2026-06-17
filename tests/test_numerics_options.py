"""Carbuncle cure + compressibility correction: both stay stable and behave
as expected (carbuncle barely changes a shock-free run; compressibility
correction lowers eddy viscosity in the high-Mach plume shear layer)."""
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rocketcfd.config import SimConfig
from rocketcfd.mask import load_mask
from rocketcfd.solver import GPUSolver

png = str(ROOT / "examples" / "nozzle_small.png")


def run(carb, comp, steps=4000):
    cfg = SimConfig()
    cfg.axisymmetric = True
    cfg.axis_location = "center"
    cfg.inlet_p0 = 4.0e6
    cfg.carbuncle_fix = carb
    cfg.compressibility_correction = comp
    mask = load_mask(png, cfg.meters_per_pixel, cfg.svg_raster_px,
                     smooth=cfg.smooth_boundary, sigma=cfg.boundary_sigma)
    sol = GPUSolver(mask, cfg)
    sol.step(steps)
    return sol.snapshot()


# all four combinations compile and stay finite
for carb in (False, True):
    for comp in (False, True):
        snap = run(carb, comp)
        m = np.nanmax(snap["fields"]["Mach"])
        assert np.isfinite(m) and 0.5 < m < 12.0, (carb, comp, m)
        print(f"carbuncle={carb!s:5} compcorr={comp!s:5}: Mach max {m:.3f}  OK")

# compressibility correction lowers eddy viscosity in the plume (it adds
# dilatational dissipation of k at high turbulent Mach number)
s_off = run(True, False)
s_on = run(True, True)
mut_off = np.nanmean(s_off["fields"]["Eddy viscosity ratio mu_t/mu [-]"])
mut_on = np.nanmean(s_on["fields"]["Eddy viscosity ratio mu_t/mu [-]"])
print(f"mean mu_t/mu: compcorr off {mut_off:.1f}, on {mut_on:.1f}")
assert mut_on < mut_off, "compressibility correction should lower eddy viscosity"
print("numerics-options test OK")
